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
