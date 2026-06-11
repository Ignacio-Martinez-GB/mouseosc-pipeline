"""
===============================================================================
ESTILO Y COLOR — apariencia consistente de todas las figuras
===============================================================================

Centraliza la apariencia para que TODAS las figuras se vean igual y los grupos
tengan SIEMPRE el mismo color en todo el proyecto.

  - apply_style()   : ajusta matplotlib (tipografía, rejilla, bordes) una vez.
  - color_map()     : asigna un color fijo a cada grupo (de config.plotting.palette
                      o de una paleta por defecto accesible).
  - band_label()    : etiqueta de banda con su rango en Hz, p. ej. "gamma_lo\\n(30–60 Hz)".
  - footer()        : escribe el método estadístico al pie de la figura.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paleta por defecto (Okabe-Ito: distinguible también en daltonismo).
_DEFAULT_CYCLE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                  "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def apply_style():
    """Ajustes globales de apariencia. Llamar una vez al inicio de la corrida."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,      # quita bordes superior/derecho (más limpio)
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "axes.axisbelow": True,
    })


def color_map(levels, cfg):
    """Devuelve {nivel: color}. Usa config.plotting.palette si define el nivel;
    si no, asigna colores de la paleta por defecto de forma estable (ordenada)."""
    palette = (cfg.get("plotting", {}) or {}).get("palette", {}) or {}
    out, k = {}, 0
    for lv in levels:
        if lv in palette:
            out[lv] = palette[lv]
        else:
            out[lv] = _DEFAULT_CYCLE[k % len(_DEFAULT_CYCLE)]
            k += 1
    return out


def band_label(name, rng=None):
    """Etiqueta de banda con su rango en Hz en una segunda línea."""
    if rng is None:
        return name
    lo, hi = rng
    return f"{name}\n({lo:g}–{hi:g} Hz)"


def footer(fig, text):
    """Escribe el método estadístico (u otra nota) al pie de la figura."""
    if text:
        fig.text(0.5, -0.02, text, ha="center", va="top",
                 fontsize=8, color="#555", style="italic")
