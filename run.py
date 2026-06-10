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

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

from mouseosc import (io, preprocessing as pp, spectral, bands, pac, bursts,
                      stats, checks, report, viz)
from mouseosc.provenance import header_text


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
    psd_by_group = {}
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
            psd_by_group.setdefault(rec.group, []).append(spec["psd"])
            if _como_signal[0] is None:
                _como_signal[0] = spec["_signal_clean"]
            if rec.group not in getattr(cmd_run, "_freqs", {}):
                cmd_run.__dict__.setdefault("_freqs", {})[rec.group] = spec["freqs"]

    html = report.write_reports(check_records, cfg, out_dir)
    print(f"\nReporte de salud: {html}")

    if validate_only:
        print("Modo validate: no se generaron resultados.")
        return 0

    if not rows:
        print("Ningún registro pasó los checks; no hay métricas que exportar.")
        return 1

    df = pd.DataFrame(rows)
    metrics_csv = out_dir / "metrics_all.csv"
    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write(header_text(cfg) + "\n")
        df.to_csv(f, index=False)
    print(f"Métricas por registro: {metrics_csv}  ({len(df)} registros)")

    # Estadística de grupos sobre las métricas numéricas relevantes.
    metric_cols = [c for c in df.columns if c.endswith(("_abs", "_rel", "_rms"))
                   or c in ("aperiodic_exponent", "median_freq", "spectral_entropy")]
    if df[cfg["statistics"]["group_col"]].nunique() >= 2:
        stats_df = stats.compare_all(df, metric_cols, cfg)
        stats_df.to_csv(out_dir / "stats_comparisons.csv", index=False)
        print(f"Comparaciones de grupos: {out_dir/'stats_comparisons.csv'}")

    # Figuras.
    if cfg.get("output", {}).get("save_figures", True):
        fig_dir = out_dir / "figuras"
        fig_dir.mkdir(exist_ok=True)
        pbg = {g: (cmd_run._freqs[g], psds) for g, psds in psd_by_group.items()}
        viz.plot_group_psd(pbg, fig_dir / "psd_por_grupo.png", cfg)
        for m in ("gamma_lo_abs", "aperiodic_exponent"):
            if m in df.columns:
                viz.plot_band_box(df, m, cfg, fig_dir / f"box_{m}.png")
        # Comodulograma (caro): se calcula UNA vez sobre el registro guardado.
        if cfg.get("pac", {}).get("comodulogram", {}).get("enabled", False) and _como_signal[0] is not None:
            ph, am, mi = pac.compute_comodulogram(_como_signal[0], cfg["preprocessing"]["fs"], cfg)
            viz.plot_comodulogram(ph, am, mi, fig_dir / "comodulograma.png", cfg)
        print(f"Figuras en {fig_dir}")
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


def main():
    ap = argparse.ArgumentParser(description="mouseosc — pipeline de oscilaciones en ratón")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan-folder"); p.add_argument("folder"); p.add_argument("--out", default="manifest.csv")
    p = sub.add_parser("inspect"); p.add_argument("file")
    p = sub.add_parser("validate"); p.add_argument("--config", default="config.yaml")
    p = sub.add_parser("run"); p.add_argument("--config", default="config.yaml")
    p = sub.add_parser("demo")

    args = ap.parse_args()
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
