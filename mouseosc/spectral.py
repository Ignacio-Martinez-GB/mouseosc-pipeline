"""
===============================================================================
ESPECTRO — PSD de Welch + separación 1/f (specparam / FOOOF)
===============================================================================

WELCH: divide cada época (ventana Hann), calcula el periodograma |FFT|² y
promedia ENTRE épocas → PSD con varianza reducida. Resolución Δf = 1 / window_s.

SPECPARAM (opcional): descompone el PSD en
  • componente APERIÓDICO  L(f) = b − log10(f^χ)   (b=offset, χ=exponente ~ E/I)
  • PICOS oscilatorios (gaussianas): CF (centro), PW (altura), BW (ancho).

specparam es una dependencia OPCIONAL. Si no está instalada o se desactiva en
config (spectral.specparam.enabled=false), se calcula solo el PSD de Welch.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.signal import welch


def compute_psd_welch(epochs, fs, window_s=2.0, overlap=0.5,
                      freq_min=0.5, freq_max=500.0):
    """
    PSD de Welch promediado entre épocas.

    Con epoch_length_s == window_s cada época es UN segmento de Welch y el
    promedio ocurre entre épocas. Devuelve (freqs, psd) recortado a [freq_min, freq_max].
    """
    nperseg = int(round(window_s * fs))
    noverlap = int(nperseg * overlap)
    psd_list = []
    for epoch in epochs:
        f, p = welch(epoch, fs=fs, window="hann", nperseg=nperseg,
                     noverlap=noverlap, scaling="density")
        psd_list.append(p)
    psd_mean = np.mean(psd_list, axis=0)
    mask = (f >= freq_min) & (f <= freq_max)
    return f[mask], psd_mean[mask]


def fit_specparam(freqs, psd, cfg_sp):
    """
    Ajusta specparam para separar 1/f de los picos. Devuelve dict con
    aperiodic_params [offset, exponente], peaks [[CF,PW,BW]...], r_squared.
    Lanza ImportError si specparam no está instalado (el caller lo captura).
    """
    from specparam import SpectralModel
    fm = SpectralModel(
        peak_width_limits=cfg_sp.get("peak_width_limits", [0.5, 8.0]),
        max_n_peaks=cfg_sp.get("max_n_peaks", 6),
        min_peak_height=cfg_sp.get("min_peak_height", 0.05),
        aperiodic_mode=cfg_sp.get("aperiodic_mode", "fixed"),
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm.fit(freqs, psd, freq_range=cfg_sp.get("freq_range", [1.0, 300.0]))
    res = fm.results.get_results()
    peaks_raw = res.peak_converted
    peaks = peaks_raw.tolist() if peaks_raw is not None and len(peaks_raw) > 0 else []
    return {
        "aperiodic_params": res.aperiodic_fit.tolist(),
        "peaks": peaks,
        "r_squared": float(res.metrics.get("gof_rsquared", np.nan)),
    }


def analyze_recording(epochs, fs, cfg, window_s=None):
    """PSD + specparam de UN registro. specparam se omite si no está habilitado
    o no instalado (se anota en el resultado para que checks lo reporte)."""
    sp = cfg.get("spectral", {})
    w = sp.get("welch", {})
    if window_s is None:
        window_s = sp.get("primary_window_s", 2.0)
    freqs, psd = compute_psd_welch(
        epochs, fs, window_s=window_s,
        overlap=w.get("overlap", 0.5),
        freq_min=w.get("freq_min", 0.5),
        freq_max=w.get("freq_max", 500.0))
    out = {"freqs": freqs, "psd": psd, "window_s": window_s,
           "specparam_ok": False, "aperiodic_params": None,
           "peaks": [], "r_squared": np.nan}
    sp_cfg = sp.get("specparam", {})
    if sp_cfg.get("enabled", True):
        try:
            out.update(fit_specparam(freqs, psd, sp_cfg))
            out["specparam_ok"] = True
        except ImportError:
            out["specparam_note"] = "specparam no instalado (pip install mouseosc[specparam])"
        except Exception as e:
            out["specparam_note"] = f"specparam falló: {e}"
    return out
