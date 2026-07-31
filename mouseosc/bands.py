"""
===============================================================================
MÉTRICAS POR BANDA
===============================================================================

Por cada banda de config -> bands calcula:
  abs_power : integral del PSD en [lo,hi] (regla del trapecio). Unidades amplitud².
  rel_power : abs_power / potencia en el rango de referencia (relative_power).
  rms       : RMS de la señal filtrada a esa banda (dominio del tiempo, amplitud).

Métricas globales (una vez por PSD):
  total_power, median_freq, spectral_edge_95, spectral_entropy.
"""
from __future__ import annotations

import numpy as np
from .preprocessing import bandpass_filter

# Compatibilidad numpy: en 2.x la función es np.trapezoid; en 1.x es np.trapz.
# Tomamos la que exista para funcionar con cualquier versión del entorno.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def apply_analysis_band(cfg):
    """
    Recorta TODO el análisis al rango global cfg['analysis_band'] = [lo, hi].
    Se aplica de forma consistente en todo el proyecto:
      - las BANDAS se acotan SOLO en los extremos: una banda interior no cambia;
        una que cruza el límite se recorta a él; una totalmente fuera se elimina.
      - el PSD (Welch), specparam, el rango de potencia relativa, el tope de
        armónicos de ruido y las bandas de PAC se acotan al mismo rango.
    Si no hay analysis_band, no hace nada. Modifica y devuelve cfg.
    """
    ab = cfg.get("analysis_band")
    if not ab:
        return cfg
    lo, hi = float(ab[0]), float(ab[1])

    # 1) bandas: recortar solo los extremos; eliminar las que quedan vacías (fuera)
    nb = {}
    for name, (blo, bhi) in cfg.get("bands", {}).items():
        clo, chi = max(blo, lo), min(bhi, hi)
        if clo < chi:
            nb[name] = [clo, chi]
    cfg["bands"] = nb

    # 2) PSD de Welch: el rango de análisis define los límites (0.4 se conserva
    #    aunque el primer bin real dependa de la resolución de la ventana).
    w = cfg.setdefault("spectral", {}).setdefault("welch", {})
    w["freq_min"] = lo
    w["freq_max"] = hi

    # 3) specparam
    sp = cfg["spectral"].setdefault("specparam", {})
    fr = sp.get("freq_range", [1.0, 300.0])
    sp["freq_range"] = [max(fr[0], lo), min(fr[1], hi)]

    # 4) rango de referencia de potencia relativa
    rp = cfg.setdefault("relative_power", {})
    rr = rp.get("reference_range", [0.5, 160.0])
    rp["reference_range"] = [max(rr[0], lo), min(rr[1], hi)]

    # 5) tope de armónicos de ruido
    if isinstance(cfg.get("noise"), dict):
        cfg["noise"]["freq_max"] = min(cfg["noise"].get("freq_max", 200.0), hi)

    # 6) bandas de PAC (fase/amplitud)
    for pair in cfg.get("pac", {}).get("pairs", []) or []:
        for k in ("phase", "amp"):
            b = pair.get(k)
            if b:
                pair[k] = [max(b[0], lo), min(b[1], hi)]
    return cfg


def band_power(freqs, psd, lo, hi):
    """Integral del PSD en [lo,hi] Hz (área bajo la curva = potencia absoluta)."""
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        return np.nan
    return float(_trapz(psd[mask], freqs[mask]))


def band_rms(signal, fs, lo, hi):
    """RMS de la señal filtrada a la banda (amplitud, no al cuadrado)."""
    filt = bandpass_filter(signal, fs, lo, hi)
    return float(np.sqrt(np.mean(filt ** 2)))


def median_frequency(freqs, psd):
    """Frecuencia bajo la cual cae el 50 % de la potencia total."""
    cum = np.cumsum(psd * np.gradient(freqs))
    total = cum[-1]
    if total == 0:
        return np.nan
    return float(freqs[np.searchsorted(cum, total / 2)])


def spectral_edge(freqs, psd, frac=0.95):
    """Frecuencia bajo la cual cae `frac` de la potencia total (edge 95 %)."""
    cum = np.cumsum(psd * np.gradient(freqs))
    total = cum[-1]
    if total == 0:
        return np.nan
    return float(freqs[np.searchsorted(cum, total * frac)])


def spectral_entropy(psd):
    """Entropía de Shannon del PSD normalizado (0=picudo, 1=plano)."""
    p = psd / np.sum(psd)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def compute_all_metrics(freqs, psd, signal, fs, cfg):
    """
    Devuelve un dict plano (una fila) con todas las métricas por banda y globales.
    Pensado para acumularse en un DataFrame (una fila por registro).
    """
    bands = cfg.get("bands", {})
    ref = cfg.get("relative_power", {}).get("reference_range", [0.5, 160.0])
    total_ref = band_power(freqs, psd, ref[0], ref[1])

    row = {}
    sum_abs = 0.0
    for name, (lo, hi) in bands.items():
        ap = band_power(freqs, psd, lo, hi)
        row[f"{name}_abs"] = ap
        row[f"{name}_rel"] = ap / total_ref if total_ref and total_ref > 0 else np.nan
        row[f"{name}_rms"] = band_rms(signal, fs, lo, hi)
        if not np.isnan(ap):
            sum_abs += ap

    row["total_power_ref"] = total_ref
    row["sum_bands_abs"] = sum_abs              # usado por el check de conservación
    row["median_freq"] = median_frequency(freqs, psd)
    row["spectral_edge_95"] = spectral_edge(freqs, psd, 0.95)
    row["spectral_entropy"] = spectral_entropy(psd)
    return row
