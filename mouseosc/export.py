"""
===============================================================================
EXPORTACIÓN POR SEGMENTO DE ANÁLISIS
===============================================================================

Un MISMO motor (`export_analyses`) exporta cualquier subconjunto de datos
agrupado por una columna (group, sex, …). Lo usan tanto:
  - el bloque DESCRIPTIVO (todos los grupos juntos), como
  - cada COMPARACIÓN por pares de 2 grupos (todas las combinaciones).

Por cada segmento se crea una carpeta con:
  espectro/   PSD por grupo (log-log) + PSD con bandas + PSD por banda (zoom)
              + datos detrás (media/sem) + PSD por sujeto.
  bandas/     barras abs/rel + boxplots por banda (con pie estadístico) +
              prism/ (CSV columnas=grupos) + datos largos.
  specparam/  (si hay) boxplots + prism.
  pac/        (si hay) boxplots + prism.
  bursts/     (si hay) boxplots + prism.
  estadistica/ stats_comparisons.csv (2 grupos o omnibus+pares).
  metrics.csv  maestro del segmento (filas = registros del subconjunto).

Los CSV de prism/ NO llevan cabecera: 1ª fila = nombres de grupo → pegables en
GraphPad Prism.
"""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import pandas as pd

from . import viz, stats, style
from .provenance import header_text


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def _save_csv(df, path, cfg, index=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header_text(cfg) + "\n")
        df.to_csv(f, index=index, na_rep="")


def _prism(df, metric, gcol, path):
    """CSV Prism (columnas=grupo, filas=valor por sujeto), sin cabecera."""
    groups = sorted(df[gcol].dropna().unique())
    series = {g: df.loc[df[gcol] == g, metric].dropna().reset_index(drop=True)
              for g in groups}
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(series).to_csv(path, index=False, na_rep="")


def _omnibus_pf(df, metric, gcol, cfg):
    """Devuelve (nombre_prueba, p) del omnibus del subconjunto, o (None, nan)."""
    paired = cfg.get("statistics", {}).get("paired", False)
    data = [df.loc[df[gcol] == g, metric].dropna().values
            for g in sorted(df[gcol].dropna().unique())]
    if len(data) < 2 or any(len(d) < 2 for d in data):
        return None, float("nan")
    try:
        name, _stat, p = stats._omnibus(data, paired)
        return name, p
    except Exception:
        return None, float("nan")


def _footer(df, metric, gcol, cfg):
    """Texto del pie: prueba estadística usada y p-valor."""
    name, p = _omnibus_pf(df, metric, gcol, cfg)
    if name is None:
        return "n insuficiente para prueba estadística"
    return f"{name}: p = {p:.4f}" + ("" if p >= 0.05 else "  (*)")


def _pvalue(df, metric, gcol, cfg):
    """Solo el p-valor del omnibus (para marcar significancia por banda)."""
    return _omnibus_pf(df, metric, gcol, cfg)[1]


def _psd_by_group(df, psd_store, freqs, gcol):
    out, ids = {}, {}
    for g in sorted(df[gcol].dropna().unique()):
        recs = [r for r in df.loc[df[gcol] == g, "rec_id"] if r in psd_store]
        if recs:
            out[g] = (freqs, [psd_store[r] for r in recs])
            ids[g] = recs
    return out, ids


# ---------------------------------------------------------------------------
# motor único
# ---------------------------------------------------------------------------
def export_analyses(df, psd_store, freqs, gcol, out_dir, cfg, label="", bandpower_kind="bar"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig = cfg.get("output", {}).get("save_figures", True)
    scale = (cfg.get("plotting", {}) or {}).get("psd_scale", "loglog")
    scales = ["loglog", "semilog"] if scale == "both" else [scale]
    bands_cfg = cfg.get("bands", {})

    # maestro del segmento
    _save_csv(df, out_dir / "metrics.csv", cfg)

    # ---------------- ESPECTRO ----------------
    pbg, ids = _psd_by_group(df, psd_store, freqs, gcol)
    if pbg:
        d = out_dir / "espectro"; d.mkdir(exist_ok=True)
        ttl = f"PSD por grupo {label}".strip()
        if save_fig:
            for sc in scales:
                suf = "" if len(scales) == 1 else f"_{sc}"
                viz.plot_group_psd(pbg, d / f"psd_por_grupo{suf}.png", cfg, scale=sc, title=ttl)
            viz.plot_psd_bands(pbg, d / "psd_con_bandas.png", cfg, scale=scales[0])
            # PSD por banda con zoom (sub-bandas en detalle)
            db = d / "por_banda"; db.mkdir(exist_ok=True)
            for name, rng in bands_cfg.items():
                viz.plot_band_psd_zoom(pbg, name, rng, db / f"psd_{name}.png", cfg)
        # datos detrás de la figura
        any_g = next(iter(pbg)); fr = pbg[any_g][0]
        mean_sem = {"freq_hz": fr}
        for g, (_, psds) in pbg.items():
            arr = np.asarray(psds)
            mean_sem[f"{g}_media"] = arr.mean(0)
            mean_sem[f"{g}_sem"] = arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros(arr.shape[1])
        _save_csv(pd.DataFrame(mean_sem), d / "psd_media_sem.csv", cfg)
        full = {"freq_hz": fr}
        for g, (_, psds) in pbg.items():
            for rid, psd in zip(ids[g], psds):
                full[rid] = psd
        _save_csv(pd.DataFrame(full), d / "psd_por_sujeto.csv", cfg)

    # ---------------- BANDAS ----------------
    d = out_dir / "bandas"; d.mkdir(exist_ok=True)
    if save_fig:
        for suf in ("abs", "rel"):
            if any(f"{b}_{suf}" in df.columns for b in bands_cfg):
                sig = {b: _pvalue(df, f"{b}_{suf}", gcol, cfg)
                       for b in bands_cfg if f"{b}_{suf}" in df.columns}
                viz.plot_bandpower(df, gcol, cfg, d / f"bandpower_{suf}.png",
                                   suffix=suf, kind=bandpower_kind, sig_by_band=sig)
        for b in bands_cfg:
            for suf in ("abs", "rel"):
                m = f"{b}_{suf}"
                if m in df.columns:
                    titulo = style.band_label(b, bands_cfg[b]).replace("\n", " ") + f" — {suf}"
                    viz.plot_box(df, m, gcol, cfg, d / f"box_{m}.png",
                                 ylabel=f"{b} ({suf})", title=titulo,
                                 stat_text=_footer(df, m, gcol, cfg),
                                 sig_pairs=stats.significant_pairs(df, m, gcol, cfg))
    band_metrics = [f"{b}_{s}" for b in bands_cfg for s in ("abs", "rel", "rms")]
    for m in band_metrics:
        if m in df.columns and df[m].notna().any():
            _prism(df, m, gcol, d / "prism" / f"{m}.csv")
    idv = [c for c in ("rec_id", gcol, "animal_id") if c in df.columns]
    long = df[idv + [m for m in band_metrics if m in df.columns]].melt(
        id_vars=idv, var_name="metrica", value_name="valor")
    _save_csv(long, d / "bandas_largo.csv", cfg)

    # ---------------- SPECPARAM / PAC / BURSTS ----------------
    _export_metric_group(df, gcol, cfg, out_dir / "specparam",
                         [c for c in ("aperiodic_exponent", "aperiodic_offset", "specparam_r2") if c in df.columns], save_fig)
    _export_metric_group(df, gcol, cfg, out_dir / "pac",
                         [c for c in df.columns if c.startswith("pac_") and c.endswith(("_mi", "_mvl"))], save_fig)
    _export_metric_group(df, gcol, cfg, out_dir / "bursts",
                         [c for c in df.columns if c.startswith("burst_")], save_fig,
                         box_only_suffixes=("_n_bursts", "_burst_rate_hz", "_mean_duration_ms", "_burst_fraction"))

    # ---------------- ESTADÍSTICA ----------------
    if df[gcol].nunique() >= 2:
        cfg2 = copy.deepcopy(cfg); cfg2["statistics"]["group_col"] = gcol
        metric_cols = [c for c in df.columns
                       if c.endswith(("_abs", "_rel", "_rms", "_mi", "_mvl"))
                       or c in ("aperiodic_exponent", "aperiodic_offset", "median_freq",
                                "spectral_entropy", "spectral_edge_95")
                       or c.startswith("burst_")]
        metric_cols = [c for c in metric_cols if c != "sum_bands_abs"]
        sdf = stats.compare_all(df, metric_cols, cfg2)
        _save_csv(sdf, out_dir / "estadistica" / "stats_comparisons.csv", cfg)
        return int(sdf["significant"].sum()) if len(sdf) else 0
    return 0


def export_factorial_figures(df, factors, out_dir, cfg, metrics=None, posthoc_df=None):
    """
    Figuras del CRUCE de factores (p. ej. sexo × dieta × condición): una por
    métrica, con todas las celdas en un panel, colores por familia del factor
    externo y corchetes de significancia del post-hoc de celdas.
    """
    out_dir = Path(out_dir) / "figuras_celdas"
    out_dir.mkdir(parents=True, exist_ok=True)
    bands_cfg = cfg.get("bands", {})
    if metrics is None:
        metrics = ([f"{b}_{s}" for b in bands_cfg for s in ("abs", "rel")] +
                   [c for c in ("aperiodic_exponent", "median_freq", "spectral_entropy")
                    if c in df.columns] +
                   [c for c in df.columns if c.startswith("pac_") and c.endswith("_mi")])
    hechas = 0
    for m in metrics:
        if m not in df.columns or not df[m].notna().any():
            continue
        sig = []
        if posthoc_df is not None and len(posthoc_df):
            sub = posthoc_df[(posthoc_df["metric"] == m) & posthoc_df["significant"]]
            for _, r in sub.iterrows():
                a, b = str(r["comparison"]).split(" vs ")
                sig.append((a, b, float(r["p_corrected"])))
        titulo = m
        base = m.rsplit("_", 1)[0]
        if base in bands_cfg:
            titulo = style.band_label(base, bands_cfg[base]).replace("\n", " ") + \
                     f" — {m.rsplit('_', 1)[1]}"
        viz.plot_factorial_cells(df, m, factors, cfg, out_dir / f"celdas_{m}.png",
                                 title=titulo, sig_pairs=sig)
        hechas += 1
    return hechas


def _export_metric_group(df, gcol, cfg, out_dir, cols, save_fig, box_only_suffixes=None):
    """Exporta un grupo de métricas (specparam/pac/bursts): boxplots + prism."""
    if not cols:
        return
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    box_cols = cols if box_only_suffixes is None else \
        [c for c in cols if c.endswith(box_only_suffixes)]
    if save_fig:
        for m in box_cols:
            viz.plot_box(df, m, gcol, cfg, out_dir / f"box_{m}.png",
                         ylabel=m, title=m, stat_text=_footer(df, m, gcol, cfg),
                         sig_pairs=stats.significant_pairs(df, m, gcol, cfg))
    for m in cols:
        if df[m].notna().any():
            _prism(df, m, gcol, out_dir / "prism" / f"{m}.csv")
