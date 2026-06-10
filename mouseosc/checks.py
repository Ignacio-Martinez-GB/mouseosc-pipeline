"""
===============================================================================
CHECKS — capa de verificación (el "¿hace lo que debe?" del pipeline)
===============================================================================

Filosofía: cada etapa emite verificaciones con un SEMÁFORO.
  • verde  (ok)    : todo en orden.
  • ámbar  (warn)  : sospechoso, revisar (no detiene salvo que lo pidas).
  • rojo   (error) : algo está mal; detiene si checks.stop_on_error=true.

Un Check es un registro estructurado (no un print), para que el reporte pueda
listarlos por registro y dar un veredicto global. Esto convierte la validación
en un producto revisable, no en mensajes perdidos en la consola.

Las funciones check_* reciben los datos de una etapa y devuelven lista de Check.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Check:
    stage: str         # etapa: "carga", "preproceso", "espectro", "bandas"...
    name: str          # qué se verificó
    level: str         # "ok" | "warn" | "error"
    message: str       # explicación legible
    value: float | None = None

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# CHECK 1 — coherencia de la señal cargada
# ---------------------------------------------------------------------------
def check_signal(signal, fs, cfg):
    """Verifica que la señal y fs sean usables ANTES de gastar cómputo."""
    out = []
    n = len(signal)
    # ¿hay señal?
    if n == 0:
        return [Check("carga", "no_vacia", "error", "La señal está vacía.")]
    # NaN / inf
    n_bad = int(np.sum(~np.isfinite(signal)))
    out.append(Check("carga", "sin_nan_inf",
                     "ok" if n_bad == 0 else "error",
                     f"{n_bad} muestras no finitas (NaN/inf).", n_bad))
    # fs vs duración: si el archivo trae su propio fs, debe cuadrar con el config.
    fs_cfg = cfg["preprocessing"]["fs"]
    tol = cfg["preprocessing"].get("fs_tolerance", 0.02)
    if abs(fs - fs_cfg) / fs_cfg > tol:
        out.append(Check("carga", "fs_coincide", "warn",
                         f"fs del archivo ({fs:.1f} Hz) difiere del config "
                         f"({fs_cfg} Hz) más del {tol*100:.0f} %.", fs))
    else:
        out.append(Check("carga", "fs_coincide", "ok",
                         f"fs={fs:.1f} Hz coherente con el config.", fs))
    # Saturación del amplificador: muchas muestras pegadas al máximo => clipping.
    amax = np.max(np.abs(signal))
    frac_clip = float(np.mean(np.abs(signal) > 0.999 * amax))
    out.append(Check("carga", "sin_saturacion",
                     "ok" if frac_clip < 0.01 else "warn",
                     f"{frac_clip*100:.2f} % de muestras cerca del máximo (posible clipping).",
                     frac_clip))
    return out


# ---------------------------------------------------------------------------
# CHECK 2 — preprocesamiento (rechazo de épocas)
# ---------------------------------------------------------------------------
def check_preprocessing(pp_result, cfg):
    """Avisa si se rechazaron demasiadas épocas o quedan muy pocas para promediar."""
    out = []
    frac = pp_result["rejected_frac"]
    thr = cfg["checks"].get("max_rejected_epochs_frac", 0.30)
    out.append(Check("preproceso", "epocas_rechazadas",
                     "ok" if frac <= thr else "warn",
                     f"{frac*100:.1f} % de épocas rechazadas "
                     f"({pp_result['n_epochs_clean']}/{pp_result['n_epochs_total']} limpias).",
                     frac))
    if pp_result["n_epochs_clean"] < 5:
        out.append(Check("preproceso", "epocas_suficientes", "error",
                         f"Solo {pp_result['n_epochs_clean']} épocas limpias: "
                         f"el PSD será muy ruidoso.", pp_result["n_epochs_clean"]))
    return out


# ---------------------------------------------------------------------------
# CHECK 3 — espectro (calidad del PSD y del ajuste 1/f)
# ---------------------------------------------------------------------------
def check_spectral(spec, cfg):
    """PSD positivo y finito; R² de specparam por encima del mínimo aceptable."""
    out = []
    psd = spec["psd"]
    ok_psd = np.all(np.isfinite(psd)) and np.all(psd >= 0)
    out.append(Check("espectro", "psd_valido",
                     "ok" if ok_psd else "error",
                     "PSD finito y no negativo." if ok_psd else "PSD con valores inválidos."))
    if spec.get("specparam_ok"):
        r2 = spec["r_squared"]
        min_r2 = cfg["checks"].get("min_specparam_r2", 0.90)
        out.append(Check("espectro", "specparam_r2",
                         "ok" if r2 >= min_r2 else "warn",
                         f"R² del ajuste 1/f = {r2:.3f} (mínimo {min_r2}).", r2))
    elif "specparam_note" in spec:
        out.append(Check("espectro", "specparam", "warn", spec["specparam_note"]))
    return out


# ---------------------------------------------------------------------------
# CHECK 4 — conservación de energía (las bandas suman ≈ potencia total)
# ---------------------------------------------------------------------------
def check_energy_conservation(freqs, psd, metrics_row, cfg):
    """
    Verifica que la suma de potencias de las bandas que cubren el rango de
    referencia se aproxime a la integral del PSD en ese mismo rango. Si no
    cuadra, hay un error de integración o bandas mal definidas (huecos/solapes).
    """
    ref = cfg.get("relative_power", {}).get("reference_range", [0.5, 160.0])
    from .bands import band_power
    total_ref = band_power(freqs, psd, ref[0], ref[1])
    # Suma de las bandas contenidas en el rango de referencia.
    s = 0.0
    for name, (lo, hi) in cfg.get("bands", {}).items():
        if lo >= ref[0] and hi <= ref[1]:
            bp = band_power(freqs, psd, lo, hi)
            if np.isfinite(bp):
                s += bp
    tol = cfg["checks"].get("energy_conservation_tol", 0.05)
    rel_err = abs(s - total_ref) / total_ref if total_ref else np.inf
    return [Check("bandas", "conservacion_energia",
                  "ok" if rel_err <= tol else "warn",
                  f"Σ bandas vs integral del rango ref: error {rel_err*100:.1f} % "
                  f"(tolerancia {tol*100:.0f} %).", rel_err)]


def check_band_definitions(cfg):
    """
    Verifica (UNA vez por corrida, no por registro) que las bandas estén bien
    definidas: cada banda con lo<hi, sin solapes entre bandas contiguas y sin
    huecos dentro del rango de referencia. Atrapa errores de config temprano.
    """
    out = []
    bands = cfg.get("bands", {})
    items = sorted(bands.items(), key=lambda kv: kv[1][0])
    # lo < hi en cada banda
    for name, (lo, hi) in items:
        if not (lo < hi):
            out.append(Check("config", "banda_valida", "error",
                             f"Banda '{name}' tiene lo>=hi ([{lo}, {hi}])."))
    # solapes entre bandas contiguas
    for (n1, (l1, h1)), (n2, (l2, h2)) in zip(items, items[1:]):
        if l2 < h1:
            out.append(Check("config", "bandas_sin_solape", "warn",
                             f"'{n1}' y '{n2}' se solapan ({h1} > {l2}). "
                             f"La potencia compartida se contará dos veces."))
    if not out:
        out.append(Check("config", "bandas", "ok",
                         f"{len(items)} bandas bien definidas, sin solapes."))
    return out


def worst_level(checks):
    """Veredicto agregado: error > warn > ok."""
    levels = [c.level for c in checks]
    if "error" in levels:
        return "error"
    if "warn" in levels:
        return "warn"
    return "ok"
