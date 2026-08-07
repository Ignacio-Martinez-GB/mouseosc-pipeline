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
    a los demás les asigna colores de la paleta por defecto SIN repetir los que
    ya usó la paleta (evita que dos grupos salgan del mismo color)."""
    palette = (cfg.get("plotting", {}) or {}).get("palette", {}) or {}
    out, used = {}, set()
    # 1) niveles con color definido en el config
    for lv in levels:
        if lv in palette:
            out[lv] = palette[lv]
            used.add(str(palette[lv]).lower())
    # 2) el resto: siguiente color del ciclo que NO esté ya usado
    for lv in levels:
        if lv not in out:
            col = next((c for c in _DEFAULT_CYCLE if c.lower() not in used), None)
            col = col or _DEFAULT_CYCLE[len(out) % len(_DEFAULT_CYCLE)]
            out[lv] = col
            used.add(col.lower())
    return out


# Familias de color: un TONO por cada nivel del factor externo (p. ej. sexo).
# Dentro de cada familia, variantes claras/oscuras para las subcondiciones, de
# modo que "hembras" y "machos" se distingan de un vistazo pero cada subgrupo
# tenga su matiz propio (como en las figuras de publicación).
_FAMILIES = {
    "hembra": ["#E8C547", "#F0DC82", "#C1467F", "#E58BB4"],   # amarillos → rosas
    "macho":  ["#6BAED6", "#A6CEE3", "#6A3D9A", "#B49BD8"],   # azules → morados
    "_alt1":  ["#0072B2", "#7FC4E8", "#00695C", "#66B2A8"],
    "_alt2":  ["#D55E00", "#F0A16A", "#8C3B00", "#C98A5E"],
}


def family_colors(outer_levels, n_inner, cfg=None):
    """
    Devuelve {nivel_externo: [colores...]} — una FAMILIA de color por cada nivel
    del factor externo (p. ej. hembra/macho), con `n_inner` variantes dentro para
    las subcondiciones (dieta × condición). Permite ver el sexo por el color
    general y la subcondición por el matiz.
    Se puede sobrescribir en config -> plotting.familias.
    """
    custom = ((cfg or {}).get("plotting", {}) or {}).get("familias", {}) or {}
    alt = ["_alt1", "_alt2"]
    out = {}
    for i, lv in enumerate(outer_levels):
        key = str(lv).lower()
        base = custom.get(lv) or custom.get(key) or _FAMILIES.get(key) \
            or _FAMILIES[alt[i % len(alt)]]
        # repetir/recortar la familia hasta cubrir n_inner subcondiciones
        out[lv] = [base[j % len(base)] for j in range(max(n_inner, 1))]
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
