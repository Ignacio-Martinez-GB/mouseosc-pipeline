"""
===============================================================================
EXPORTACIÓN DE RESULTADOS — una carpeta por análisis
===============================================================================

Organiza las salidas para que cada tipo de análisis tenga lo suyo:
  - su(s) figura(s),
  - los DATOS DETRÁS de cada figura (lo que se graficó, no solo la imagen),
  - los datos COMPLETOS del análisis (formato largo),
  - CSVs en formato GraphPad Prism (columnas = grupos, filas = valores por
    sujeto) listos para copiar y pegar.

Estructura generada en `resultados/`:

  metrics_all.csv            ← maestro: una fila por registro, todas las métricas
  espectro/
    psd_por_grupo.png
    psd_grupo_media_sem.csv  ← datos detrás de la figura (freq, media, sem por grupo)
    psd_por_sujeto.csv       ← PSD completo: una columna por registro
  bandas/
    bandpower_abs.png, bandpower_rel.png
    box_<banda>_<abs|rel>.png
    bandas_largo.csv         ← datos completos (formato largo)
    prism/<banda>_<abs|rel|rms>.csv
  specparam/   (si está activo)
    box_aperiodic_exponent.png, box_aperiodic_offset.png
    prism/aperiodic_*.csv
  pac/         (si está activo)
    box_pac_<par>_<mi|mvl>.png
    prism/pac_<par>_<mi|mvl>.csv
  bursts/      (si está activo)
    box_burst_*.png ; prism/*.csv
  estadistica/
    stats_comparisons.csv    ← omnibus + pares con corrección múltiple

Formato Prism: cada columna es un grupo; cada fila, el valor de un sujeto. Las
columnas de distinto largo se rellenan con celdas vacías. Pegable directo en una
tabla "Column" de GraphPad Prism.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from . import viz
from .provenance import header_text


# ---------------------------------------------------------------------------
# Utilidades de formato
# ---------------------------------------------------------------------------
def _save_csv(df, path, cfg, index=False):
    """Guarda un CSV con cabecera de procedencia (1ª línea como comentario)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header_text(cfg) + "\n")
        df.to_csv(f, index=index, na_rep="")


def prism_wide(df, metric, group_col):
    """Tabla ancha estilo Prism: una columna por grupo, filas = valores por sujeto.
    Las columnas de distinto largo quedan rellenas con vacío (NaN→'')."""
    groups = sorted(df[group_col].dropna().unique())
    series = {g: df.loc[df[group_col] == g, metric].dropna().reset_index(drop=True)
              for g in groups}
    return pd.DataFrame(series)   # alinea por índice y rellena con NaN


def _prism_for_metrics(df, metrics, group_col, out_dir, cfg):
    """Escribe un CSV Prism por cada métrica que exista en df.
    SIN cabecera de procedencia: la 1ª fila son los nombres de grupo, para que
    se pueda copiar/pegar directo en una tabla de GraphPad Prism."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for m in metrics:
        if m in df.columns and df[m].notna().any():
            prism_wide(df, m, group_col).to_csv(out_dir / f"{m}.csv",
                                                index=False, na_rep="")
            written.append(m)
    return written


# ---------------------------------------------------------------------------
# Maestro
# ---------------------------------------------------------------------------
def export_master(df, out_dir, cfg):
    _save_csv(df, out_dir / "metrics_all.csv", cfg)


# ---------------------------------------------------------------------------
# Espectro: figura PSD por grupo + datos detrás + PSD por sujeto
# ---------------------------------------------------------------------------
def export_spectral(psd_by_group, freqs_by_group, rec_ids_by_group, out_dir, cfg):
    d = out_dir / "espectro"; d.mkdir(parents=True, exist_ok=True)
    save_fig = cfg.get("output", {}).get("save_figures", True)

    # Figura
    if save_fig:
        pbg = {g: (freqs_by_group[g], psds) for g, psds in psd_by_group.items()}
        viz.plot_group_psd(pbg, d / "psd_por_grupo.png", cfg)

    # Datos DETRÁS de la figura: freq, <grupo>_media, <grupo>_sem
    any_g = next(iter(freqs_by_group))
    out = {"freq_hz": freqs_by_group[any_g]}
    for g, psds in psd_by_group.items():
        arr = np.array(psds)
        out[f"{g}_media"] = arr.mean(0)
        out[f"{g}_sem"] = arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros(arr.shape[1])
    _save_csv(pd.DataFrame(out), d / "psd_grupo_media_sem.csv", cfg)

    # PSD COMPLETO por sujeto: freq + una columna por registro
    full = {"freq_hz": freqs_by_group[any_g]}
    for g, psds in psd_by_group.items():
        for rid, psd in zip(rec_ids_by_group[g], psds):
            full[rid] = psd
    _save_csv(pd.DataFrame(full), d / "psd_por_sujeto.csv", cfg)


# ---------------------------------------------------------------------------
# Bandas: barras abs/rel + boxplots por banda + prism + largo
# ---------------------------------------------------------------------------
def export_bands(df, out_dir, cfg):
    d = out_dir / "bandas"; d.mkdir(parents=True, exist_ok=True)
    gcol = cfg["statistics"]["group_col"]
    bands = list(cfg.get("bands", {}).keys())
    save_fig = cfg.get("output", {}).get("save_figures", True)

    # Barras de potencia por banda (abs y rel)
    if save_fig:
        for suf in ("abs", "rel"):
            if any(f"{b}_{suf}" in df.columns for b in bands):
                viz.plot_bandpower_bars(df, cfg, d / f"bandpower_{suf}.png", suffix=suf)
        # Boxplot por banda (abs y rel): vista por animal
        for b in bands:
            for suf in ("abs", "rel"):
                m = f"{b}_{suf}"
                if m in df.columns:
                    viz.plot_band_box(df, m, cfg, d / f"box_{m}.png",
                                      ylabel=f"{b} ({suf})", title=f"{b} ({suf}) por grupo")

    # CSVs Prism (abs, rel, rms por banda) — son también los datos de los boxplots
    metrics = [f"{b}_{s}" for b in bands for s in ("abs", "rel", "rms")]
    _prism_for_metrics(df, metrics, gcol, d / "prism", cfg)

    # Datos COMPLETOS en formato largo
    long_cols = ["rec_id", gcol, "animal_id"] + [m for m in metrics if m in df.columns]
    long = df[[c for c in long_cols if c in df.columns]].melt(
        id_vars=[c for c in ["rec_id", gcol, "animal_id"] if c in df.columns],
        var_name="metrica", value_name="valor")
    _save_csv(long, d / "bandas_largo.csv", cfg)


# ---------------------------------------------------------------------------
# Specparam (1/f)
# ---------------------------------------------------------------------------
def export_specparam(df, out_dir, cfg):
    cols = [c for c in ("aperiodic_exponent", "aperiodic_offset", "specparam_r2")
            if c in df.columns]
    if not cols:
        return
    d = out_dir / "specparam"; d.mkdir(parents=True, exist_ok=True)
    gcol = cfg["statistics"]["group_col"]
    if cfg.get("output", {}).get("save_figures", True):
        for m in cols:
            viz.plot_band_box(df, m, cfg, d / f"box_{m}.png", ylabel=m, title=f"{m} por grupo")
    _prism_for_metrics(df, cols, gcol, d / "prism", cfg)


# ---------------------------------------------------------------------------
# PAC
# ---------------------------------------------------------------------------
def export_pac(df, out_dir, cfg):
    cols = [c for c in df.columns if c.startswith("pac_") and c.endswith(("_mi", "_mvl"))]
    if not cols:
        return
    d = out_dir / "pac"; d.mkdir(parents=True, exist_ok=True)
    gcol = cfg["statistics"]["group_col"]
    if cfg.get("output", {}).get("save_figures", True):
        for m in cols:
            viz.plot_band_box(df, m, cfg, d / f"box_{m}.png", ylabel=m, title=f"{m} por grupo")
    _prism_for_metrics(df, cols, gcol, d / "prism", cfg)


# ---------------------------------------------------------------------------
# Bursts
# ---------------------------------------------------------------------------
def export_bursts(df, out_dir, cfg):
    cols = [c for c in df.columns if c.startswith("burst_")]
    if not cols:
        return
    d = out_dir / "bursts"; d.mkdir(parents=True, exist_ok=True)
    gcol = cfg["statistics"]["group_col"]
    if cfg.get("output", {}).get("save_figures", True):
        # solo boxplot de las métricas resumidas más interpretables
        for m in [c for c in cols if c.endswith(("_n_bursts", "_burst_rate_hz",
                                                 "_mean_duration_ms", "_burst_fraction"))]:
            viz.plot_band_box(df, m, cfg, d / f"box_{m}.png", ylabel=m, title=f"{m} por grupo")
    _prism_for_metrics(df, cols, gcol, d / "prism", cfg)


# ---------------------------------------------------------------------------
# Estadística
# ---------------------------------------------------------------------------
def export_stats(stats_df, out_dir, cfg):
    d = out_dir / "estadistica"; d.mkdir(parents=True, exist_ok=True)
    _save_csv(stats_df, d / "stats_comparisons.csv", cfg)
