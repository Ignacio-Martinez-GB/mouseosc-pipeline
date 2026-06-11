"""
===============================================================================
FIGURAS — apariencia consistente, color por grupo, pie estadístico
===============================================================================

Todas las figuras:
  - usan la MISMA paleta por grupo (mouseosc.style.color_map),
  - llevan cabecera de procedencia y, cuando aplica, el método estadístico al pie,
  - autoescalan a la dispersión de los datos,
  - etiquetan las bandas con su rango en Hz.

PSD por defecto en log-log (resalta la forma 1/f). Hay además un PSD "por banda"
con zoom y escala propia para mirar sub-bandas en detalle.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import style
from .provenance import header_text


def _save(fig, path, cfg, footer_text=None):
    fig.text(0.005, 0.005, header_text(cfg), fontsize=5, color="#bbb")
    style.footer(fig, footer_text)
    fig.savefig(path, dpi=cfg.get("output", {}).get("dpi", 150), bbox_inches="tight")
    plt.close(fig)


def _order(levels, cfg):
    """Orden estable de grupos (alfabético) y su mapa de color."""
    levels = sorted(levels)
    return levels, style.color_map(levels, cfg)


# ---------------------------------------------------------------------------
# PSD por grupo (media ± SEM, sombreado)
# ---------------------------------------------------------------------------
def plot_group_psd(psd_by_group, out_path, cfg, scale="loglog", title=None,
                   xlim=None, footer_text=None):
    """PSD medio ± SEM por grupo. psd_by_group: {grupo: (freqs, [psd...])}."""
    levels, cmap = _order(list(psd_by_group), cfg)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for g in levels:
        freqs, psds = psd_by_group[g]
        arr = np.asarray(psds)
        m = arr.mean(0)
        sem = arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(m)
        ax.plot(freqs, m, color=cmap[g], lw=1.8, label=f"{g} (n={len(arr)})")
        ax.fill_between(freqs, m - sem, m + sem, color=cmap[g], alpha=0.22, linewidth=0)
    if scale == "loglog":
        ax.set(xscale="log", yscale="log")
    elif scale == "semilog":
        ax.set(yscale="log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set(xlabel="Frecuencia (Hz)", ylabel="PSD (amplitud²/Hz)",
           title=title or "PSD por grupo (media ± SEM)")
    ax.legend()
    _save(fig, out_path, cfg, footer_text)


def plot_psd_bands(psd_by_group, out_path, cfg, scale="loglog", title=None):
    """PSD por grupo con las bandas sombreadas de fondo (referencia visual)."""
    levels, cmap = _order(list(psd_by_group), cfg)
    bands = cfg.get("bands", {})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # franjas de banda al fondo
    for i, (name, (lo, hi)) in enumerate(bands.items()):
        ax.axvspan(lo, hi, color="#000", alpha=0.04 if i % 2 == 0 else 0.0)
        ax.text(np.sqrt(lo * hi) if scale == "loglog" else (lo + hi) / 2,
                0.98, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color="#777", rotation=90)
    for g in levels:
        freqs, psds = psd_by_group[g]
        arr = np.asarray(psds); m = arr.mean(0)
        sem = arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(m)
        ax.plot(freqs, m, color=cmap[g], lw=1.8, label=f"{g} (n={len(arr)})")
        ax.fill_between(freqs, m - sem, m + sem, color=cmap[g], alpha=0.22, linewidth=0)
    if scale == "loglog":
        ax.set(xscale="log", yscale="log")
    elif scale == "semilog":
        ax.set(yscale="log")
    ax.set(xlabel="Frecuencia (Hz)", ylabel="PSD (amplitud²/Hz)",
           title=title or "PSD con bandas")
    ax.legend()
    _save(fig, out_path, cfg)


def plot_band_psd_zoom(psd_by_group, band_name, rng, out_path, cfg):
    """PSD con ZOOM a una banda concreta y escala propia (autoescala a esa banda).
    Útil para ver sub-bandas que en el panorama completo quedan aplastadas."""
    lo, hi = rng
    levels, cmap = _order(list(psd_by_group), cfg)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ymax = 0
    for g in levels:
        freqs, psds = psd_by_group[g]
        freqs = np.asarray(freqs); arr = np.asarray(psds)
        mask = (freqs >= lo) & (freqs <= hi)
        m = arr.mean(0); sem = arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(m)
        ax.plot(freqs[mask], m[mask], color=cmap[g], lw=1.8, label=f"{g} (n={len(arr)})")
        ax.fill_between(freqs[mask], (m - sem)[mask], (m + sem)[mask],
                        color=cmap[g], alpha=0.22, linewidth=0)
        ymax = max(ymax, np.nanmax((m + sem)[mask]) if mask.any() else 0)
    ax.set_xlim(lo, hi)
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.1)   # autoescala a la dispersión de la banda
    ax.set(xlabel="Frecuencia (Hz)", ylabel="PSD (amplitud²/Hz)",
           title=style.band_label(band_name, rng).replace("\n", " "))
    ax.legend()
    _save(fig, out_path, cfg)


# ---------------------------------------------------------------------------
# Boxplot por grupo (color + puntos + pie estadístico)
# ---------------------------------------------------------------------------
def plot_box(df, metric, gcol, cfg, out_path, ylabel=None, title=None,
             footer_text=None):
    levels, cmap = _order(df[gcol].dropna().unique(), cfg)
    data = [df.loc[df[gcol] == g, metric].dropna().values for g in levels]
    fig, ax = plt.subplots(figsize=(1.6 * len(levels) + 2, 4))
    bp = ax.boxplot(data, labels=levels, showmeans=True, patch_artist=True,
                    widths=0.6, medianprops=dict(color="#222"))
    for patch, g in zip(bp["boxes"], levels):
        patch.set_facecolor(cmap[g]); patch.set_alpha(0.45)
    rng = np.random.default_rng(0)
    for i, (g, vals) in enumerate(zip(levels, data), start=1):
        ax.plot(rng.normal(i, 0.06, len(vals)), vals, "o", ms=4,
                color=cmap[g], alpha=0.8, markeredgecolor="white", markeredgewidth=0.4)
    ax.set(ylabel=ylabel or metric, title=title or metric)
    ax.set_xticklabels(levels, rotation=0)
    _save(fig, out_path, cfg, footer_text)


# ---------------------------------------------------------------------------
# Barras de potencia por banda (color por grupo, etiqueta con Hz)
# ---------------------------------------------------------------------------
def plot_bandpower_bars(df, gcol, cfg, out_path, suffix="abs", footer_text=None):
    bands_cfg = cfg.get("bands", {})
    levels, cmap = _order(df[gcol].dropna().unique(), cfg)
    cols = [f"{b}_{suffix}" for b in bands_cfg if f"{b}_{suffix}" in df.columns]
    names = [c.rsplit("_", 1)[0] for c in cols]
    labels = [style.band_label(n, bands_cfg[n]) for n in names]
    x = np.arange(len(names)); w = 0.8 / max(len(levels), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(names) + 2), 4.8))
    for j, g in enumerate(levels):
        sub = df[df[gcol] == g]
        m = [sub[c].mean() for c in cols]
        sem = [sub[c].std(ddof=1) / np.sqrt(max(sub[c].notna().sum(), 1)) for c in cols]
        ax.bar(x + j * w, m, w, yerr=sem, capsize=3, color=cmap[g],
               alpha=0.85, label=f"{g} (n={len(sub)})")
    ylab = {"abs": "Potencia absoluta (amplitud²)", "rel": "Potencia relativa",
            "rms": "RMS (amplitud)"}[suffix]
    ax.set(xticks=x + w * (len(levels) - 1) / 2, ylabel=ylab,
           title=f"Potencia por banda ({suffix}) — media ± SEM")
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    _save(fig, out_path, cfg, footer_text)


# ---------------------------------------------------------------------------
# Comodulograma
# ---------------------------------------------------------------------------
def plot_comodulogram(phases, amps, mi_grid, out_path, cfg, title="Comodulograma (MI)"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(phases, amps, mi_grid, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Modulation Index")
    ax.set(xlabel="Frecuencia de FASE (Hz)", ylabel="Frecuencia de AMPLITUD (Hz)",
           title=title)
    _save(fig, out_path, cfg)
