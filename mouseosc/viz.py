"""
===============================================================================
FIGURAS — apariencia consistente, color por grupo, significancia y pie explicativo
===============================================================================

Todas las figuras:
  - usan la MISMA paleta por grupo (mouseosc.style.color_map),
  - llevan cabecera de procedencia y, cuando aplica, el método estadístico al pie,
  - autoescalan a la dispersión de los datos,
  - etiquetan las bandas con su rango en Hz (en el título o eje X),
  - marcan con barras/asteriscos las diferencias significativas.

PSD: eje X = frecuencia (Hz) en escala log con ticks legibles; sin sombras grises;
se excluye la franja de ruido eléctrico (config.plotting.notch_hz, p. ej. 57–63 Hz).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

from . import style
from .provenance import header_text

# Pie de figura estándar para cajas y bigotes.
BOX_FOOTER = ("Caja: mediana e IQR (Q1–Q3)  ·  bigotes: 1.5×IQR  ·  "
              "△ = media  ·  puntos = animales")


def _save(fig, path, cfg, footer_text=None):
    fig.text(0.005, 0.005, header_text(cfg), fontsize=5, color="#bbb")
    style.footer(fig, footer_text)
    fig.savefig(path, dpi=cfg.get("output", {}).get("dpi", 150), bbox_inches="tight")
    plt.close(fig)


def _order(levels, cfg):
    levels = sorted(levels)
    return levels, style.color_map(levels, cfg)


def _notch(cfg):
    return (cfg.get("plotting", {}) or {}).get("notch_hz", [57, 63])


def _mask_notch(freqs, *arrays, notch=None):
    """Pone NaN en la franja del notch (ruido eléctrico) para que NO se grafique
    ni cuente en la autoescala. Devuelve las arrays modificadas."""
    if not notch:
        return arrays
    lo, hi = notch
    bad = (freqs >= lo) & (freqs <= hi)
    out = []
    for a in arrays:
        a = np.array(a, dtype=float); a[bad] = np.nan; out.append(a)
    return out


def _freq_xaxis(ax):
    """Eje X de frecuencia en log con TICKS LEGIBLES (1, 2, 5, 10, 20, 50, 100…)."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))  # 0.5, 1, 2, 5, 10…
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Frecuencia (Hz)")


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


# ---------------------------------------------------------------------------
# Barras de significancia (corchetes) para boxplots
# ---------------------------------------------------------------------------
def _sig_brackets(ax, positions, pairs, data, ylog=False):
    """Dibuja corchetes de significancia entre posiciones x. `pairs` = lista
    (label_a, label_b, p). `positions` = {label: x}. `data` = {label: valores}."""
    if not pairs:
        return
    ymax = max([np.nanmax(v) for v in data.values() if len(v)] + [0])
    ymin = min([np.nanmin(v) for v in data.values() if len(v)] + [0])
    span = (ymax / max(ymin, 1e-9)) if ylog else (ymax - ymin)
    step = (span ** 0.06) if ylog else span * 0.08
    level = ymax * (1.05 if ylog else 1) + (0 if ylog else span * 0.05)
    for i, (a, b, p) in enumerate(sorted(pairs, key=lambda t: abs(positions[t[0]] - positions[t[1]]))):
        x1, x2 = positions[a], positions[b]
        y = level * (step ** i) if ylog else level + step * i
        h = y * 0.02 if ylog else span * 0.015
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, color="#333")
        ax.text((x1 + x2) / 2, y + h, _stars(p) or "ns", ha="center", va="bottom",
                fontsize=10, color="#333")


# ---------------------------------------------------------------------------
# PSD por grupo (media ± SEM, sin sombra gris, X log legible)
# ---------------------------------------------------------------------------
def plot_group_psd(psd_by_group, out_path, cfg, scale="loglog", title=None,
                   footer_text=None):
    levels, cmap = _order(list(psd_by_group), cfg)
    notch = _notch(cfg)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for g in levels:
        freqs, psds = psd_by_group[g]
        freqs = np.asarray(freqs); arr = np.asarray(psds)
        m = arr.mean(0); sem = arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(m)
        m, sem = _mask_notch(freqs, m, sem, notch=notch)
        ax.plot(freqs, m, color=cmap[g], lw=1.8, label=f"{g} (n={len(arr)})")
        ax.fill_between(freqs, m - sem, m + sem, color=cmap[g], alpha=0.20, linewidth=0)
    _freq_xaxis(ax)
    if scale in ("loglog", "semilog"):
        ax.set_yscale("log")
    ax.set(ylabel="PSD (amplitud²/Hz)", title=title or "PSD por grupo (media ± SEM)")
    ax.legend()
    _save(fig, out_path, cfg, footer_text)


def plot_psd_bands(psd_by_group, out_path, cfg, scale="loglog", title=None):
    """PSD por grupo con líneas verticales (no sombras) marcando las bandas."""
    levels, cmap = _order(list(psd_by_group), cfg)
    bands = cfg.get("bands", {}); notch = _notch(cfg)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    edges = sorted({v for rng in bands.values() for v in rng})
    for e in edges:
        ax.axvline(e, color="#ccc", lw=0.6, zorder=0)
    for name, (lo, hi) in bands.items():
        ax.text(np.sqrt(lo * hi), 0.99, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color="#888", rotation=90)
    for g in levels:
        freqs, psds = psd_by_group[g]
        freqs = np.asarray(freqs); arr = np.asarray(psds)
        m = arr.mean(0); sem = arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(m)
        m, sem = _mask_notch(freqs, m, sem, notch=notch)
        ax.plot(freqs, m, color=cmap[g], lw=1.8, label=f"{g} (n={len(arr)})")
        ax.fill_between(freqs, m - sem, m + sem, color=cmap[g], alpha=0.20, linewidth=0)
    _freq_xaxis(ax)
    if scale in ("loglog", "semilog"):
        ax.set_yscale("log")
    ax.set(ylabel="PSD (amplitud²/Hz)", title=title or "PSD con bandas")
    ax.legend()
    _save(fig, out_path, cfg)


def plot_band_psd_zoom(psd_by_group, band_name, rng, out_path, cfg):
    """PSD con ZOOM a una banda y escala propia. Excluye el notch (57–63 Hz) para
    que el ruido eléctrico no domine la autoescala."""
    lo, hi = rng
    levels, cmap = _order(list(psd_by_group), cfg)
    notch = _notch(cfg)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ymax = 0
    for g in levels:
        freqs, psds = psd_by_group[g]
        freqs = np.asarray(freqs); arr = np.asarray(psds)
        m = arr.mean(0); sem = arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(m)
        m, sem = _mask_notch(freqs, m, sem, notch=notch)
        sel = (freqs >= lo) & (freqs <= hi)
        ax.plot(freqs[sel], m[sel], color=cmap[g], lw=1.8, label=f"{g} (n={len(arr)})")
        ax.fill_between(freqs[sel], (m - sem)[sel], (m + sem)[sel], color=cmap[g], alpha=0.20, linewidth=0)
        if sel.any() and np.isfinite(m[sel]).any():
            ymax = max(ymax, np.nanmax((m + sem)[sel]))
    ax.set_xlim(lo, hi)
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.1)
    ax.set(xlabel="Frecuencia (Hz)", ylabel="PSD (amplitud²/Hz)",
           title=style.band_label(band_name, rng).replace("\n", " "))
    ax.legend()
    _save(fig, out_path, cfg)


# ---------------------------------------------------------------------------
# Boxplot por grupo (color + puntos + significancia + pie explicativo)
# ---------------------------------------------------------------------------
def plot_box(df, metric, gcol, cfg, out_path, ylabel=None, title=None,
             stat_text=None, sig_pairs=None, ylog=False):
    levels, cmap = _order(df[gcol].dropna().unique(), cfg)
    data = {g: df.loc[df[gcol] == g, metric].dropna().values for g in levels}
    fig, ax = plt.subplots(figsize=(1.7 * len(levels) + 2, 4.4))
    bp = ax.boxplot([data[g] for g in levels], labels=levels, showmeans=True,
                    patch_artist=True, widths=0.6, medianprops=dict(color="#222"),
                    meanprops=dict(marker="^", markerfacecolor="white",
                                   markeredgecolor="#222", markersize=8))
    for patch, g in zip(bp["boxes"], levels):
        patch.set_facecolor(cmap[g]); patch.set_alpha(0.45)
    rng = np.random.default_rng(0)
    for i, g in enumerate(levels, start=1):
        ax.plot(rng.normal(i, 0.06, len(data[g])), data[g], "o", ms=4,
                color=cmap[g], alpha=0.85, markeredgecolor="white", markeredgewidth=0.4)
    if ylog:
        ax.set_yscale("log")
    _sig_brackets(ax, {g: i for i, g in enumerate(levels, start=1)},
                  sig_pairs or [], data, ylog=ylog)
    ax.set(ylabel=ylabel or metric, title=title or metric)
    footer = BOX_FOOTER + (f"\n{stat_text}" if stat_text else "")
    _save(fig, out_path, cfg, footer)


# ---------------------------------------------------------------------------
# Potencia por banda — barras (descriptivo) o cajas (comparaciones)
# ---------------------------------------------------------------------------
def plot_bandpower(df, gcol, cfg, out_path, suffix="abs", kind="box",
                   sig_by_band=None):
    """Potencia por banda en CAJA Y BIGOTES con puntos (kind='box') o barras
    (kind='bar'), escala Y log para ver las bandas de alta frecuencia.
    Cuando hay 2 grupos, marca con un CORCHETE que une ambos grupos de cada banda
    significativa (sig_by_band: {banda: p})."""
    bands_cfg = cfg.get("bands", {})
    levels, cmap = _order(df[gcol].dropna().unique(), cfg)
    cols = [f"{b}_{suffix}" for b in bands_cfg if f"{b}_{suffix}" in df.columns]
    names = [c.rsplit("_", 1)[0] for c in cols]
    labels = [style.band_label(n, bands_cfg[n]) for n in names]
    x = np.arange(len(names)); w = 0.8 / max(len(levels), 1)
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(names) + 2), 5))
    ylab = {"abs": "Potencia absoluta (amplitud²)", "rel": "Potencia relativa",
            "rms": "RMS (amplitud)"}[suffix]

    band_max = np.full(len(names), 1e-12)   # máximo por banda (para colocar corchetes)
    for j, g in enumerate(levels):
        sub = df[df[gcol] == g]
        xpos = x + j * w
        for bi, (xi, c) in enumerate(zip(xpos, cols)):
            vals = sub[c].dropna().values
            if len(vals) == 0:
                continue
            band_max[bi] = max(band_max[bi], np.nanmax(vals))
            if kind == "bar":
                m = np.nanmean(vals); sem = np.nanstd(vals, ddof=1) / np.sqrt(len(vals))
                ax.bar(xi, m, w, yerr=sem, capsize=3, color=cmap[g], alpha=0.85)
            else:
                ax.boxplot([vals], positions=[xi], widths=w * 0.9, patch_artist=True,
                           showmeans=True, medianprops=dict(color="#222"),
                           meanprops=dict(marker="^", markerfacecolor="white",
                                          markeredgecolor="#222", markersize=6),
                           boxprops=dict(facecolor=cmap[g], alpha=0.45))
                rng = np.random.default_rng(0)
                ax.plot(rng.normal(xi, w * 0.08, len(vals)), vals, "o", ms=3,
                        color=cmap[g], alpha=0.8)
        ax.plot([], [], "s", color=cmap[g], alpha=0.7, label=f"{g} (n={len(sub)})")

    ax.set_yscale("log")
    ax.set(xticks=x + w * (len(levels) - 1) / 2, ylabel=ylab,
           title=f"Potencia por banda ({suffix}) — escala log")
    ax.set_xticklabels(labels, fontsize=8)

    # Corchete de significancia que UNE los dos grupos en cada banda (caso 2 grupos).
    if sig_by_band and len(levels) >= 2:
        x_left = x + 0 * w
        x_right = x + (len(levels) - 1) * w
        for bi, n in enumerate(names):
            p = sig_by_band.get(n)
            if p is not None and p < 0.05:
                y = band_max[bi] * 1.4
                ax.plot([x_left[bi], x_left[bi], x_right[bi], x_right[bi]],
                        [y, y * 1.15, y * 1.15, y], lw=1.1, color="#333")
                ax.text((x_left[bi] + x_right[bi]) / 2, y * 1.18, _stars(p),
                        ha="center", va="bottom", fontsize=10, color="#333")
    ax.legend()
    foot = BOX_FOOTER if kind == "box" else None
    _save(fig, out_path, cfg, foot)


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
