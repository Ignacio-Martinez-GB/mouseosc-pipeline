"""
===============================================================================
FIGURAS
===============================================================================
Gráficas básicas con cabecera de procedencia incrustada. Matplotlib sin estado
global de estilo para que sea reproducible.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")           # backend sin ventana (corre en servidores)
import matplotlib.pyplot as plt
from .provenance import header_text


def _save(fig, path, cfg):
    fig.text(0.005, 0.005, header_text(cfg), fontsize=5, color="#999")
    fig.savefig(path, dpi=cfg.get("output", {}).get("dpi", 150), bbox_inches="tight")
    plt.close(fig)


def plot_group_psd(psd_by_group, out_path, cfg):
    """PSD medio ± SEM por grupo, en log-log. psd_by_group: {grupo:(freqs, [psd...])}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group, (freqs, psds) in psd_by_group.items():
        arr = np.array(psds)
        m = arr.mean(0)
        sem = arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(m)
        ax.plot(freqs, m, label=f"{group} (n={len(arr)})")
        ax.fill_between(freqs, m - sem, m + sem, alpha=0.25)
    ax.set(xscale="log", yscale="log", xlabel="Frecuencia (Hz)",
           ylabel="PSD (amplitud²/Hz)", title="PSD por grupo (media ± SEM)")
    ax.legend()
    _save(fig, out_path, cfg)


def plot_comodulogram(phases, amps, mi_grid, out_path, cfg, title="Comodulograma (MI)"):
    """Heatmap MI sobre la grilla fase×amplitud. Eje X = frecuencia de fase
    (banda lenta moduladora), eje Y = frecuencia de amplitud (banda rápida)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(phases, amps, mi_grid, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Modulation Index")
    ax.set(xlabel="Frecuencia de FASE (Hz)", ylabel="Frecuencia de AMPLITUD (Hz)",
           title=title)
    _save(fig, out_path, cfg)


def plot_band_box(df, metric, cfg, out_path):
    """Boxplot de una métrica por grupo."""
    gcol = cfg.get("statistics", {}).get("group_col", "group")
    groups = sorted(df[gcol].dropna().unique())
    data = [df.loc[df[gcol] == g, metric].dropna().values for g in groups]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(data, labels=groups, showmeans=True)
    ax.set(ylabel=metric, title=f"{metric} por grupo")
    _save(fig, out_path, cfg)
