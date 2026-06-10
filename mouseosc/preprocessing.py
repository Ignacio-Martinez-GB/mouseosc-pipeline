"""
===============================================================================
PREPROCESAMIENTO — limpieza de la señal antes del análisis
===============================================================================

Cadena: detrend lineal → filtro paso-alto (fase cero) → épocas → rechazo de
artefactos. Todos los parámetros vienen de config -> preprocessing.

Por qué FASE CERO (sosfiltfilt): aplica el filtro hacia adelante y hacia atrás,
de modo que NO desplaza la señal en el tiempo. Esto es crítico para no
distorsionar la FASE que luego usa el análisis PAC.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend as scipy_detrend


def _sos_filter(signal, fs, cutoff, btype, order=4):
    """Construye y aplica un Butterworth fase-cero. 'sos' es numéricamente
    más estable que (b,a) en órdenes altos."""
    nyq = fs / 2.0
    if btype == "high":
        wn = cutoff / nyq
    elif btype == "low":
        if cutoff >= nyq:        # nada que filtrar por encima de Nyquist
            return signal
        wn = cutoff / nyq
    else:
        raise ValueError(btype)
    sos = butter(order, wn, btype=btype, output="sos")
    return sosfiltfilt(sos, signal)


def highpass_filter(signal, fs, cutoff_hz):
    """Paso-alto: quita derivas lentas residuales tras el detrend."""
    return _sos_filter(signal, fs, cutoff_hz, "high")


def lowpass_filter(signal, fs, cutoff_hz):
    """Paso-bajo: anti-aliasing de software."""
    return _sos_filter(signal, fs, cutoff_hz, "low")


def bandpass_filter(signal, fs, lo, hi, order=4):
    """Paso-banda fase-cero para aislar una banda (RMS o envolvente de Hilbert)."""
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 1 - 1e-4)
    sos = butter(order, [lo_n, hi_n], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def preprocess(signal, fs, highpass_hz=0.5, lowpass_hz=None, do_detrend=True):
    """Detrend lineal + paso-alto (+ paso-bajo opcional). No muta la entrada."""
    sig = np.asarray(signal, dtype=float).copy()
    if do_detrend:
        sig = scipy_detrend(sig, type="linear")   # quita pendiente + DC
    sig = highpass_filter(sig, fs, highpass_hz)
    if lowpass_hz is not None:
        sig = lowpass_filter(sig, fs, lowpass_hz)
    return sig


def make_epochs(signal, fs, length_s=2.0, overlap=0.5):
    """Segmenta en épocas solapadas. Devuelve (n_épocas, n_muestras)."""
    n_per = int(round(length_s * fs))
    if n_per > len(signal):
        raise ValueError(
            f"epoch_length_s={length_s}s ({n_per} muestras) es mayor que la señal "
            f"({len(signal)} muestras). Baja epoch_length_s o revisa fs.")
    step = max(int(n_per * (1 - overlap)), 1)
    starts = range(0, len(signal) - n_per + 1, step)
    return np.array([signal[s:s + n_per] for s in starts])


def reject_artifacts(epochs, threshold_sd=8.0):
    """Descarta épocas con amplitud pico > threshold_sd × SD global.
    Devuelve (épocas_limpias, máscara_booleana)."""
    global_sd = np.std(epochs)
    peak_amp = np.max(np.abs(epochs), axis=1)
    mask = peak_amp < threshold_sd * global_sd
    return epochs[mask], mask


def full_pipeline(signal, fs, cfg_pp, epoch_length_s=None):
    """
    Preprocesa de punta a punta y devuelve un dict con todo lo necesario para
    los checks (no solo las épocas limpias): permite saber cuántas se rechazaron.
    """
    pp = preprocess(signal, fs,
                    highpass_hz=cfg_pp.get("highpass_hz", 0.5),
                    lowpass_hz=cfg_pp.get("lowpass_hz", None),
                    do_detrend=cfg_pp.get("detrend", True))
    if epoch_length_s is None:
        epoch_length_s = cfg_pp.get("epoch_length_s", 2.0)
    epochs = make_epochs(pp, fs, length_s=epoch_length_s,
                         overlap=cfg_pp.get("epoch_overlap", 0.5))
    clean, mask = reject_artifacts(epochs, threshold_sd=cfg_pp.get("artifact_threshold_sd", 8.0))
    return {
        "signal_clean": pp,        # señal continua filtrada (para PAC)
        "epochs": clean,           # épocas limpias (para Welch)
        "n_epochs_total": int(len(mask)),
        "n_epochs_clean": int(mask.sum()),
        "rejected_frac": float(1 - mask.mean()) if len(mask) else 1.0,
    }
