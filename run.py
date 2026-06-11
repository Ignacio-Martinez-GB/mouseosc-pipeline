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
                      stats, checks, report, viz, export)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
def cmd_scan_folder(args):
    out = Path(args.out)
    df = io.scan_folder(Path(args.folder), out)
    print(f"Plantilla de manifiesto escrita en {out} con {len(df)} archivos.")
    print("Rellena las columnas 'group' y 'animal_id' antes de correr el pipeline.")


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


def _process_one(rec, cfg):
    """Procesa un registro: devuelve (fila_métricas|None, lista_checks, spec|None)."""
    chk = []
    signal, fs = io.load_signal(rec, cfg)
    chk += checks.check_signal(signal, fs, cfg)
    if checks.worst_level(chk) == "error":
        return None, chk, None

    fs = cfg["preprocessing"]["fs"]   # tras validar, el análisis usa el fs del config
    pp_res = pp.full_pipeline(signal, fs, cfg["preprocessing"])
    chk += checks.check_preprocessing(pp_res, cfg)
    if checks.worst_level(chk) == "error":
        return None, chk, None

    spec = spectral.analyze_recording(pp_res["epochs"], fs, cfg)
    spec["_signal_clean"] = pp_res["signal_clean"]   # para el comodulograma (opcional)
    chk += checks.check_spectral(spec, cfg)

    row = bands.compute_all_metrics(spec["freqs"], spec["psd"],
                                    pp_res["signal_clean"], fs, cfg)
    chk += checks.check_energy_conservation(spec["freqs"], spec["psd"], row, cfg)

    if cfg.get("pac", {}).get("enabled", False):
        for pr in pac.run_pac_analysis(pp_res["signal_clean"], fs, cfg):
            row[f"pac_{pr['pair']}_mi"] = pr["mi"]
            row[f"pac_{pr['pair']}_mvl"] = pr["mvl"]
            row[f"pac_{pr['pair']}_p"] = pr["p_value"]

    if cfg.get("bursts", {}).get("enabled", False):
        row.update(bursts.run_burst_analysis(pp_res["signal_clean"], fs, cfg))

    if spec.get("specparam_ok"):
        row["aperiodic_offset"] = spec["aperiodic_params"][0]
        row["aperiodic_exponent"] = spec["aperiodic_params"][-1]
        row["specparam_r2"] = spec["r_squared"]

    row.update({"rec_id": rec.rec_id, "group": rec.group, "animal_id": rec.animal_id})
    row.update(rec.meta)
    return row, chk, spec


def cmd_run(args, validate_only=False):
    cfg = load_config(args.config)
    out_dir = Path(cfg.get("output", {}).get("dir", "resultados"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = cfg["checks"].get("stop_on_error", True)

    recs = io.read_manifest(cfg)
    expected = set(cfg.get("groups", {}).get("expected", []))
    found = set(r.group for r in recs)
    if expected and found - expected:
        print(f"AVISO: grupos en el manifiesto no declarados en config.groups.expected: {found-expected}")

    rows, check_records = [], []
    psd_store = {}               # {rec_id: psd}  → reconstruimos cualquier subconjunto
    freqs_ref = [None]           # vector de frecuencias (común a todos los registros)
    _como_signal = [None]        # guarda una señal limpia para el comodulograma

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
            row, chk, spec = _process_one(rec, cfg)
        except Exception as e:
            chk = [checks.Check("carga", "excepcion", "error", f"{type(e).__name__}: {e}")]
            row, spec = None, None
        verdict = checks.worst_level(chk)
        check_records.append({"rec_id": rec.rec_id, "group": rec.group,
                              "verdict": verdict, "checks": chk})
        if verdict == "error" and stop:
            print(f"\nDETENIDO por check rojo en {rec.rec_id}. "
                  f"Revisa el reporte o pon checks.stop_on_error=false para continuar.")
            report.write_reports(check_records, cfg, out_dir)
            return 1
        if row is not None and not validate_only:
            rows.append(row)
            psd_store[rec.rec_id] = spec["psd"]
            freqs_ref[0] = spec["freqs"]
            if _como_signal[0] is None:
                _como_signal[0] = spec["_signal_clean"]

    html = report.write_reports(check_records, cfg, out_dir)
    print(f"\nReporte de salud: {html}")

    if validate_only:
        print("Modo validate: no se generaron resultados.")
        return 0

    if not rows:
        print("Ningún registro pasó los checks; no hay métricas que exportar.")
        return 1

    df = pd.DataFrame(rows)
    freqs = freqs_ref[0]
    viz.style.apply_style()      # apariencia consistente en todas las figuras

    # Maestro global (una fila por registro, todas las métricas)
    export._save_csv(df, out_dir / "metrics_all.csv", cfg)
    print(f"Maestro: {out_dir/'metrics_all.csv'}  ({len(df)} registros)")

    # ---------- BLOQUE DESCRIPTIVO (todos los grupos juntos) ----------
    desc = cfg.get("descriptivo", {})
    if desc.get("enabled", True):
        by = desc.get("by", cfg["statistics"]["group_col"])
        if by in df.columns:
            n = export.export_analyses(df, psd_store, freqs, by,
                                       out_dir / "descriptivo", cfg, label="(todos)")
            print(f"Descriptivo (por '{by}'): {out_dir/'descriptivo'}  ({n} sig.)")

    # ---------- COMPARACIONES por pares de 2 grupos ----------
    comp_root = out_dir / "comparaciones"
    for comp in cfg.get("comparisons", []):
        if not comp.get("enabled", False):
            continue
        by, within, name = comp["by"], comp.get("within"), comp["name"]
        if by not in df.columns:
            print(f"AVISO: comparación '{name}' omitida: falta la columna '{by}' en el manifiesto.")
            continue
        # estratos: o bien el dataset completo, o un subconjunto por cada nivel de `within`
        if within and within in df.columns:
            estratos = [(f"{within}={lv}", df[df[within] == lv]) for lv in sorted(df[within].dropna().unique())]
        else:
            estratos = [("", df)]
        for est_label, df_est in estratos:
            levels = sorted(df_est[by].dropna().unique())
            for a, b in combinations(levels, 2):     # todas las parejas
                pair = df_est[df_est[by].isin([a, b])]
                parts = [name] + ([est_label] if est_label else []) + [f"{a}_vs_{b}"]
                sub = comp_root.joinpath(*parts)
                n = export.export_analyses(pair, psd_store, freqs, by, sub, cfg,
                                           label=f"({a} vs {b})")
                print(f"  {name}: {a} vs {b}{' ['+est_label+']' if est_label else ''} → {n} sig.")

    # ---------- Comodulograma global (opcional, caro) ----------
    if cfg.get("pac", {}).get("comodulogram", {}).get("enabled", False) and _como_signal[0] is not None:
        d = out_dir / "descriptivo" / "pac"; d.mkdir(parents=True, exist_ok=True)
        ph, am, mi = pac.compute_comodulogram(_como_signal[0], cfg["preprocessing"]["fs"], cfg)
        viz.plot_comodulogram(ph, am, mi, d / "comodulograma.png", cfg)
        export._save_csv(pd.DataFrame(mi, index=am, columns=ph),
                         d / "comodulograma_datos.csv", cfg, index=True)
    return 0


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

    p = sub.add_parser("scan-folder"); p.add_argument("folder"); p.add_argument("--out", default="manifest.csv")
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
