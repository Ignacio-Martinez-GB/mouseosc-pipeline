#!/usr/bin/env python3
"""
===============================================================================
run.py — CLI / orquestador del pipeline mouseosc
===============================================================================

Subcomandos
-----------
  scan-folder <carpeta>      genera una plantilla de manifiesto (manifest.csv)
  inspect <archivo>          lista las variables/canales de un archivo
  validate [--config c.yaml] corre SOLO la capa de checks (no produce resultados)
  run      [--config c.yaml] pipeline completo: carga → preproceso → espectro →
                             bandas → PAC → stats → figuras → reporte de salud

Ejemplos
--------
  python run.py scan-folder Datos/            # crea manifest.csv para rellenar
  python run.py validate                      # ¿están sanos los datos?
  python run.py run                           # corrida completa

DISEÑO: run.py NO contiene análisis "pesado". Carga el config y llama a los
módulos de mouseosc en orden, acumulando una fila de métricas por registro y
una lista de checks por registro. Si checks.stop_on_error=true, se detiene ante
el primer registro con veredicto rojo.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

# Aviso temprano de versión de Python: el stack científico fijado (numpy/scipy)
# tiene wheels para 3.10–3.12. Con 3.13+ pip intentaría compilar desde fuente.
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    print(f"AVISO: estás usando Python {sys.version_info.major}.{sys.version_info.minor}. "
          "Este pipeline está probado en Python 3.10–3.12. Con versiones más nuevas "
          "la instalación de numpy/scipy puede fallar (intenta compilar desde fuente). "
          "Crea el entorno con Python 3.12.")

from mouseosc import (io, preprocessing as pp, spectral, bands, pac, bursts,
                      stats, checks, report, viz, export, noise)


# Secciones válidas del config (para detectar typos del usuario).
_SECCIONES = {"project", "analysis_band", "dataset", "preprocessing", "spectral",
              "bands", "relative_power", "noise", "pac", "bursts", "statistics",
              "groups", "plotting", "descriptivo", "comparisons", "output", "checks"}


def load_config(path):
    """Carga el config, avisa de secciones desconocidas (typos) y aplica el
    rango global de análisis."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    desconocidas = set(cfg) - _SECCIONES
    if desconocidas:
        print(f"⚠ AVISO: secciones no reconocidas en {path}: {sorted(desconocidas)}")
        print("  (¿un typo? se ignorarán). Secciones válidas:", ", ".join(sorted(_SECCIONES)))
    # Recorta TODO el análisis al rango global analysis_band (si está definido).
    return bands.apply_analysis_band(cfg)


# ---------------------------------------------------------------------------
def cmd_scan_folder(args):
    out = Path(args.out)
    cfg = load_config(args.config) if getattr(args, "config", None) and Path(args.config).exists() else {}
    rep = io.scan_folder(Path(args.folder), out, cfg)
    print(f"\nManifiesto escrito en {out}  ({rep['n']} archivos, modo: {rep['modo']}).")
    print(f"Columnas: {rep['columnas']}")
    if rep["modo"] == "plantilla":
        print("\n⚠ No definiste dataset.scan.factores en el config, así que se")
        print("  escribieron columnas genéricas (nivel_1, nivel_2…). Revisa los")
        print("  valores y define el diccionario de factores para clasificar solo.")
        return
    if rep["conteos"]:
        print("\nConteo por combinación de factores:")
        for combo, n in sorted(rep["conteos"].items()):
            print(f"  {combo}: {n}")
    if rep["no_reconocidos"]:
        print("\n⚠ Segmentos de carpeta NO reconocidos (posibles typos o factores"
              " faltantes en el diccionario):")
        for seg, n in sorted(rep["no_reconocidos"].items(), key=lambda kv: -kv[1]):
            print(f"  '{seg}'  ({n} archivos)")
    if rep["ambiguos"]:
        print("\nℹ Carpetas que aportan a MÁS de un factor (normal si una carpeta"
              " codifica dos cosas, p. ej. '9 DIAS TRANSPLANTE' → día + cirugía):")
        print(f"  {rep['ambiguos']}")
    print("\nRevisa el manifiesto; si algo no cuadra, ajusta el diccionario y vuelve a escanear.")


def cmd_inspect(args):
    import scipy.io
    fp = Path(args.file)
    if fp.suffix.lower() == ".mat":
        mat = scipy.io.loadmat(fp)
        print("Variables en el .mat:")
        for k, v in mat.items():
            if not k.startswith("__"):
                print(f"  {k}: shape={getattr(v,'shape',None)} dtype={getattr(v,'dtype',None)}")
    else:
        print("inspect actualmente soporta .mat. Para otros formatos usa el loader correspondiente.")


def _process_base(rec, cfg):
    """
    Procesamiento COMÚN de un registro (una sola vez): carga, preprocesa (con el
    notch de 60 Hz por defecto), calcula PSD y detecta contaminación por ruido.
    Devuelve (base|None, checks). El `base` alimenta los 3 análisis sin recalcular.
    """
    chk = []
    signal, fs = io.load_signal(rec, cfg)
    chk += checks.check_signal(signal, fs, cfg)
    if checks.worst_level(chk) == "error":
        return None, chk
    fs = cfg["preprocessing"]["fs"]
    pp_res = pp.full_pipeline(signal, fs, cfg["preprocessing"])
    chk += checks.check_preprocessing(pp_res, cfg)
    if checks.worst_level(chk) == "error":
        return None, chk
    freqs, psd = spectral.compute_psd_welch(
        pp_res["epochs"], fs,
        window_s=cfg["spectral"].get("primary_window_s", 2.0),
        overlap=cfg["spectral"]["welch"].get("overlap", 0.5),
        freq_min=cfg["spectral"]["welch"].get("freq_min", 0.5),
        freq_max=cfg["spectral"]["welch"].get("freq_max", 500.0))
    ninfo = {"flag": False}
    if cfg.get("noise", {}).get("enabled", False):
        ninfo = noise.detect_contamination(freqs, psd, pp_res["epochs"], fs, cfg)
    base = {"rec": rec, "fs": fs, "freqs": freqs, "psd": psd,
            "signal": pp_res["signal_clean"], "noise": ninfo}
    return base, chk


def _metrics_row(rec, freqs, psd, signal, fs, cfg):
    """Construye la fila de métricas a partir de un PSD y una señal dados
    (así el análisis 3 puede usar el PSD corregido / la señal con notch)."""
    row = bands.compute_all_metrics(freqs, psd, signal, fs, cfg)
    sp_cfg = cfg.get("spectral", {}).get("specparam", {})
    if sp_cfg.get("enabled", False):
        try:
            r = spectral.fit_specparam(freqs, psd, sp_cfg)
            row["aperiodic_offset"] = r["aperiodic_params"][0]
            row["aperiodic_exponent"] = r["aperiodic_params"][-1]
            row["specparam_r2"] = r["r_squared"]
        except Exception:
            pass
    if cfg.get("pac", {}).get("enabled", False):
        for pr in pac.run_pac_analysis(signal, fs, cfg):
            row[f"pac_{pr['pair']}_mi"] = pr["mi"]
            row[f"pac_{pr['pair']}_mvl"] = pr["mvl"]
            row[f"pac_{pr['pair']}_p"] = pr["p_value"]
    if cfg.get("bursts", {}).get("enabled", False):
        row.update(bursts.run_burst_analysis(signal, fs, cfg))
    row.update({"rec_id": rec.rec_id, "group": rec.group, "animal_id": rec.animal_id})
    row.update(rec.meta)
    return row


def _print_plan(cfg, n_recs):
    """
    Imprime el PLAN de la corrida antes de empezar: qué análisis se harán, cuáles
    están desactivados y con qué método. Así el usuario ve de un vistazo qué va a
    obtener (y qué no) sin tener que leer el config.
    """
    sp = cfg.get("spectral", {}).get("specparam", {}).get("enabled", False)
    pc = cfg.get("pac", {}).get("enabled", False)
    bu = cfg.get("bursts", {}).get("enabled", False)
    nz = cfg.get("noise", {})
    st = cfg.get("statistics", {})
    fa = st.get("factorial", {}) or {}
    on = lambda b: "SÍ" if b else "no"
    ab = cfg.get("analysis_band")

    print("\n" + "─" * 66)
    print(f"PLAN DE LA CORRIDA  ({n_recs} registros)")
    print("─" * 66)
    print(f"  Rango de análisis     : {ab[0]}–{ab[1]} Hz" if ab else "  Rango de análisis     : completo")
    print(f"  Bandas                : {', '.join(cfg.get('bands', {}))}")
    print(f"  Notch de línea        : {on(cfg['preprocessing'].get('notch_default',{}).get('enabled'))}"
          f" ({cfg['preprocessing'].get('notch_default',{}).get('hz','-')} Hz)")
    print(f"  PSD (Welch)           : SÍ (ventana {cfg['spectral'].get('primary_window_s')} s)")
    print(f"  specparam (1/f)       : {on(sp)}")
    print(f"  PAC                   : {on(pc)}"
          f"{'  (' + str(cfg['pac'].get('n_surrogates')) + ' subrogados)' if pc else ''}")
    print(f"  Bursts                : {on(bu)}")
    if nz.get("enabled"):
        print(f"  Esquema de ruido      : SÍ → análisis {nz.get('analyses')}, "
              f"corrección '{nz.get('metodo_correccion')}' sobre {nz.get('fundamental_hz')} Hz")
    else:
        print(f"  Esquema de ruido      : no (un solo análisis)")
    print(f"  Prueba estadística    : {st.get('metodo','auto')}  |  post-hoc: {st.get('posthoc','auto')}"
          f"  |  corrección: {st.get('correction','holm')}")
    print(f"  Factorial             : {on(fa.get('enabled'))}"
          f"{'  (' + ' × '.join(fa.get('factores', [])) + ')' if fa.get('enabled') else ''}")
    comps = [c["name"] for c in cfg.get("comparisons", []) if c.get("enabled")]
    print(f"  Descriptivo           : {on(cfg.get('descriptivo',{}).get('enabled'))}"
          f"  |  Comparaciones: {', '.join(comps) if comps else 'ninguna'}")
    if not cfg.get("descriptivo", {}).get("enabled") and not comps:
        print("  ⚠ AVISO: sin descriptivo ni comparaciones NO se generarán figuras ni")
        print("           estadística; solo el reporte de salud y metrics_all.csv.")
    print("─" * 66 + "\n")


def emit_analysis(df, psd_store, freqs, out_dir, cfg):
    """Genera el descriptivo + comparaciones por pares + factorial para UN
    conjunto de datos. Devuelve el nº total de comparaciones significativas."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    export._save_csv(df, out_dir / "metrics_all.csv", cfg)
    total_sig = 0

    # ---- FACTORIAL: efectos principales + interacciones (opcional) ----
    fcfg = cfg.get("statistics", {}).get("factorial", {}) or {}
    if fcfg.get("enabled", False):
        factores = [f for f in fcfg.get("factores", []) if f in df.columns]
        if len(factores) >= 2:
            mcols = [c for c in df.columns
                     if c.endswith(("_abs", "_rel", "_rms", "_mi", "_mvl"))
                     or c in ("aperiodic_exponent", "aperiodic_offset", "median_freq",
                              "spectral_entropy", "spectral_edge_95")
                     or c.startswith("burst_")]
            mcols = [c for c in mcols if c != "sum_bands_abs"]
            fdf = stats.factorial_all(df, mcols, factores, cfg)
            if len(fdf):
                export._save_csv(fdf, out_dir / "factorial" / "efectos_e_interacciones.csv", cfg)
                total_sig += int(fdf["significant"].sum())
                # POST-HOC de celdas (por defecto solo donde la interacción es signif.)
                ph = None
                if fcfg.get("posthoc_celdas", True):
                    ph = stats.factorial_posthoc_all(
                        df, mcols, factores, cfg,
                        solo_si_interaccion=fcfg.get("posthoc_solo_si_interaccion", True),
                        factorial_df=fdf)
                    if len(ph):
                        export._save_csv(ph, out_dir / "factorial" / "posthoc_celdas.csv", cfg)
                # FIGURAS del cruce de celdas (sexo × dieta × condición en un panel)
                if fcfg.get("figuras_celdas", True) and cfg.get("output", {}).get("save_figures", True):
                    n_fig = export.export_factorial_figures(
                        df, fcfg.get("factores_figura") or factores,
                        out_dir / "factorial", cfg, posthoc_df=ph)
                    if n_fig:
                        print(f"    figuras de celdas: {n_fig} → {out_dir/'factorial'/'figuras_celdas'}")
        elif fcfg.get("factores"):
            print(f"  AVISO: factorial omitido (faltan columnas: {fcfg.get('factores')}).")
    desc = cfg.get("descriptivo", {})
    if desc.get("enabled", True):
        by = desc.get("by", cfg["statistics"]["group_col"])
        if by in df.columns:
            total_sig += export.export_analyses(df, psd_store, freqs, by,
                                                out_dir / "descriptivo", cfg,
                                                label="(todos)", bandpower_kind="box")
    for comp in cfg.get("comparisons", []):
        if not comp.get("enabled", False):
            continue
        by, within, name = comp["by"], comp.get("within"), comp["name"]
        if by not in df.columns:
            print(f"  AVISO: comparación '{name}' omitida (falta columna '{by}').")
            continue
        if within and within in df.columns:
            estratos = [(f"{within}={lv}", df[df[within] == lv])
                        for lv in sorted(df[within].dropna().unique())]
        else:
            estratos = [("", df)]
        for est_label, df_est in estratos:
            for a, b in combinations(sorted(df_est[by].dropna().unique()), 2):
                pair = df_est[df_est[by].isin([a, b])]
                parts = [name] + ([est_label] if est_label else []) + [f"{a}_vs_{b}"]
                total_sig += export.export_analyses(
                    pair, psd_store, freqs, by, out_dir / "comparaciones" / Path(*parts),
                    cfg, label=f"({a} vs {b})", bandpower_kind="box")
    return total_sig


def cmd_run(args, validate_only=False):
    cfg = load_config(args.config)
    out_dir = Path(cfg.get("output", {}).get("dir", "resultados"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = cfg["checks"].get("stop_on_error", True)

    recs = io.read_manifest(cfg)
    # Los grupos REALES salen siempre del manifiesto (columna 'group'):
    found = sorted(set(r.group for r in recs))
    print(f"Grupos detectados en el manifiesto: {found}")
    # config.groups.expected es OPCIONAL: solo sirve para avisar si aparece un
    # grupo que no esperabas (control de calidad), nunca limita el análisis.
    expected = set(cfg.get("groups", {}).get("expected", []) or [])
    if expected and set(found) - expected:
        print(f"AVISO: grupos no declarados en groups.expected: {set(found)-expected}")

    _print_plan(cfg, len(recs))

    bases, check_records = [], []

    # Verificación de configuración (una vez): definición de bandas.
    cfg_checks = checks.check_band_definitions(cfg)
    check_records.append({"rec_id": "[config]", "group": "—",
                          "verdict": checks.worst_level(cfg_checks), "checks": cfg_checks})
    if checks.worst_level(cfg_checks) == "error" and stop:
        print("DETENIDO: definición de bandas inválida (ver reporte).")
        report.write_reports(check_records, cfg, out_dir)
        return 1

    for rec in tqdm(recs, desc="procesando"):
        try:
            base, chk = _process_base(rec, cfg)
        except Exception as e:
            chk = [checks.Check("carga", "excepcion", "error", f"{type(e).__name__}: {e}")]
            base = None
        verdict = checks.worst_level(chk)
        check_records.append({"rec_id": rec.rec_id, "group": rec.group,
                              "verdict": verdict, "checks": chk})
        if verdict == "error" and stop:
            print(f"\nDETENIDO por check rojo en {rec.rec_id}. "
                  f"Pon checks.stop_on_error=false para continuar.")
            report.write_reports(check_records, cfg, out_dir)
            return 1
        if base is not None and not validate_only:
            bases.append(base)

    html = report.write_reports(check_records, cfg, out_dir)
    print(f"\nReporte de salud: {html}")
    if validate_only:
        print("Modo validate: no se generaron resultados.")
        return 0
    if not bases:
        print("Ningún registro pasó los checks; no hay métricas que exportar.")
        return 1

    freqs = bases[0]["freqs"]
    viz.style.apply_style()

    ncfg = cfg.get("noise", {})
    if not ncfg.get("enabled", False):
        # ---- comportamiento clásico: un solo análisis en la raíz ----
        rows = [_metrics_row(b["rec"], freqs, b["psd"], b["signal"], b["fs"], cfg) for b in bases]
        df = pd.DataFrame(rows)
        psd_store = {b["rec"].rec_id: b["psd"] for b in bases}
        n = emit_analysis(df, psd_store, freqs, out_dir, cfg)
        print(f"Análisis completo en {out_dir}  ({n} comparaciones significativas)")
        return 0

    # =====================================================================
    # ESQUEMA DE RUIDO: hasta 3 análisis aislados
    # =====================================================================
    analyses = set(ncfg.get("analyses", [1, 2, 3]))
    n_flagged = sum(b["noise"]["flag"] for b in bases)
    print(f"\nRuido: {n_flagged}/{len(bases)} registros marcados como contaminados "
          f"({ncfg.get('fundamental_hz',10)} Hz + armónicos).")

    # Reporte de detección (siempre que el esquema de ruido esté activo)
    det_rows = []
    for b in bases:
        r = {"rec_id": b["rec"].rec_id, "group": b["rec"].group,
             "contaminado": b["noise"]["flag"], "n_armonicos_hit": b["noise"].get("n_hits", 0)}
        for h in b["noise"].get("harmonics", []):
            r[f"snr_{h['f']:g}Hz"] = round(h["snr"], 2) if h["snr"] == h["snr"] else np.nan
            r[f"persist_{h['f']:g}Hz"] = round(h["persistencia"], 2)
        det_rows.append(r)
    export._save_csv(pd.DataFrame(det_rows), out_dir / "deteccion_ruido" / "contaminacion.csv", cfg)
    print(f"Detección de ruido: {out_dir/'deteccion_ruido'/'contaminacion.csv'}")

    # Métricas base (análisis 1) — se calculan una vez y se reutilizan
    rows1 = {b["rec"].rec_id: _metrics_row(b["rec"], freqs, b["psd"], b["signal"], b["fs"], cfg)
             for b in bases}
    for b in bases:
        rows1[b["rec"].rec_id]["ruido_contaminado"] = b["noise"]["flag"]
    psd1 = {b["rec"].rec_id: b["psd"] for b in bases}

    # --- Análisis 1: NORMAL ---
    if 1 in analyses:
        df1 = pd.DataFrame(list(rows1.values()))
        n = emit_analysis(df1, psd1, freqs, out_dir / "analisis_1_normal", cfg)
        print(f"  [1] normal → {out_dir/'analisis_1_normal'} ({n} sig.)")

    # --- Análisis 2: SIN RUIDO (excluye contaminados) ---
    if 2 in analyses:
        ok_ids = [b["rec"].rec_id for b in bases if not b["noise"]["flag"]]
        df2 = pd.DataFrame([rows1[i] for i in ok_ids])
        psd2 = {i: psd1[i] for i in ok_ids}
        n = emit_analysis(df2, psd2, freqs, out_dir / "analisis_2_sin_ruido", cfg)
        print(f"  [2] sin ruido ({len(ok_ids)}/{len(bases)}) → {out_dir/'analisis_2_sin_ruido'} ({n} sig.)")

    # --- Análisis 3: CORREGIDO (resta espectral + notch para PAC/bursts) ---
    if 3 in analyses:
        metodo = ncfg.get("metodo_correccion", "interpolacion")
        ref_f, ref_p = _load_noise_reference(cfg)
        if ref_f is None and metodo != "interpolacion":
            print(f"  [3] OMITIDO: el método '{metodo}' necesita referencia de ruido "
                  f"(revisa noise.ruido) o usa metodo_correccion: interpolacion.")
        else:
            f0 = ncfg.get("fundamental_hz", 10.0); nh = ncfg.get("n_armonicos", 6)
            fmax = ncfg.get("freq_max", 200.0); Q = ncfg.get("notch_Q", 30.0)
            rows3, psd3 = [], {}
            for b in bases:
                rid = b["rec"].rec_id
                if b["noise"]["flag"]:
                    psd_c = noise.correct_psd(freqs, b["psd"], ref_f, ref_p, cfg)
                    sig_c = pp.notch_harmonics(b["signal"], b["fs"], f0, nh, fmax, Q)
                    row = _metrics_row(b["rec"], freqs, psd_c, sig_c, b["fs"], cfg)
                    row["ruido_contaminado"] = True; row["ruido_corregido"] = True
                    psd3[rid] = psd_c
                else:
                    row = dict(rows1[rid]); row["ruido_corregido"] = False
                    psd3[rid] = b["psd"]
                rows3.append(row)
            df3 = pd.DataFrame(rows3)
            n = emit_analysis(df3, psd3, freqs, out_dir / "analisis_3_corregido", cfg)
            print(f"  [3] corregido → {out_dir/'analisis_3_corregido'} ({n} sig.)")
    return 0


def _load_noise_reference(cfg):
    """Carga la referencia de ruido: del CSV precalculado si existe, o la calcula
    a partir de la carpeta de ruido (promedio de todos los archivos)."""
    ncfg = cfg.get("noise", {}); rcfg = ncfg.get("ruido", {})
    csv = rcfg.get("reference_csv")
    if csv and Path(csv).exists():
        d = pd.read_csv(csv, comment="#")
        return d["freq_hz"].values, d["psd_medio"].values
    ruido_dir = rcfg.get("dir")
    if not ruido_dir or not Path(ruido_dir).exists():
        return None, None
    # cataloga los .mat/.csv de la carpeta de ruido como "recordings" mínimos
    recs = []
    for fp in sorted(Path(ruido_dir).rglob("*")):
        if fp.suffix.lower() in io._EXT2FMT:
            recs.append(io.Recording(file_path=fp, group="ruido", animal_id=fp.stem))
    return noise.build_noise_reference(recs, cfg, io.load_signal, pp.full_pipeline)


def cmd_demo(args):
    """Demo de un comando: genera datos sintéticos, escribe un config temporal
    apuntando a ellos y corre el pipeline completo. Sirve para confirmar en
    segundos que la instalación funciona, antes de tocar datos reales."""
    import subprocess
    gen = PROJ / "examples" / "make_synthetic_data.py"
    subprocess.run([sys.executable, str(gen)], check=True)
    cfg = load_config(PROJ / "config.yaml")
    cfg["dataset"].update({"manifest": "manifest.csv",
                           "root": str(PROJ / "examples" / "datos_sinteticos"),
                           "format": "csv"})
    cfg["spectral"]["specparam"]["enabled"] = False  # demo no exige specparam
    cfg["output"]["dir"] = str(PROJ / "resultados_demo")
    demo_cfg = PROJ / "_config_demo.yaml"
    yaml.safe_dump(cfg, open(demo_cfg, "w"), allow_unicode=True)

    class A:  # objeto mínimo con .config para reutilizar cmd_run
        config = str(demo_cfg)
    rc = cmd_run(A())
    print(f"\nDemo lista. Abre el reporte: {cfg['output']['dir']}/report.html")
    return rc


_GUIA = """
╔══════════════════════════════════════════════════════════════════╗
║  mouseosc — pipeline de actividad oscilatoria en ratón            ║
╚══════════════════════════════════════════════════════════════════╝

No pasaste ningún comando. Tienes DOS formas de usar el pipeline:

  ▶ FÁCIL (sin terminal): abre el archivo  INICIO.py , edita las 2
    variables de arriba (MODO y CONFIG) y dale al botón Run de PyCharm.

  ▶ TERMINAL (avanzado): usa uno de estos comandos:
        python run.py demo                 prueba con datos sintéticos
        python run.py scan-folder Datos/   crea la plantilla manifest.csv
        python run.py inspect archivo.mat  ve las variables de un .mat
        python run.py validate             revisa la salud de los datos
        python run.py run                  análisis completo

El orden normal con datos reales es:  inspect → scan-folder → (rellenar
manifest.csv) → ajustar config.yaml → validate → run.
"""


def main():
    ap = argparse.ArgumentParser(
        description="mouseosc — pipeline de oscilaciones en ratón")
    # required=False: si lo corren sin comando (p. ej. ▶ en PyCharm sobre
    # run.py) NO falla; mostramos una guía clara en su lugar.
    sub = ap.add_subparsers(dest="cmd", required=False)

    p = sub.add_parser("scan-folder"); p.add_argument("folder")
    p.add_argument("--out", default="manifest.csv"); p.add_argument("--config", default="config.yaml")
    p = sub.add_parser("inspect"); p.add_argument("file")
    p = sub.add_parser("validate"); p.add_argument("--config", default="config.yaml")
    p = sub.add_parser("run"); p.add_argument("--config", default="config.yaml")
    p = sub.add_parser("demo")

    args = ap.parse_args()
    if args.cmd is None:
        print(_GUIA)
        return 0
    if args.cmd == "demo":
        return cmd_demo(args)
    if args.cmd == "scan-folder":
        return cmd_scan_folder(args)
    if args.cmd == "inspect":
        return cmd_inspect(args)
    if args.cmd == "validate":
        return cmd_run(args, validate_only=True)
    if args.cmd == "run":
        return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
