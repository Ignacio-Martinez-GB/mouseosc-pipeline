"""
===============================================================================
DETECCIÓN DE RÁFAGAS (BURSTS)
===============================================================================

Detecta ráfagas oscilatorias (eventos transitorios de amplitud alta) por
umbralización de la envolvente de Hilbert. Por cada banda de detección:
  1. filtrar la señal a la banda,
  2. envolvente analítica (amplitud instantánea) vía Hilbert,
  3. umbral = media + threshold_sd × SD de la envolvente,
  4. fusionar gaps cortos y descartar ráfagas demasiado breves.

PARÁMETROS (config -> bursts)
-----------------------------
  detection_bands : bandas donde buscar ráfagas (dict nombre→[lo,hi]).
  threshold_sd    : ↑ más estricto (menos ráfagas, más intensas).
  min_duration_ms : duración mínima para contar una ráfaga.
  merge_gap_ms    : fusiona ráfagas separadas por menos que este gap.

Devuelve, por banda: n_bursts, burst_rate_hz, mean_duration_ms, mean_amplitude,
burst_fraction (proporción del tiempo en ráfaga).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from .preprocessing import bandpass_filter


def detect_bursts(signal, fs, lo, hi, threshold_sd=2.0,
                  min_duration_ms=50.0, merge_gap_ms=25.0):
    """Detecta ráfagas en una banda. Devuelve lista de dicts por ráfaga."""
    envelope = np.abs(hilbert(bandpass_filter(signal, fs, lo, hi)))
    threshold = np.mean(envelope) + threshold_sd * np.std(envelope)
    above = envelope > threshold

    min_samples = int(min_duration_ms * fs / 1000)
    merge_samples = int(merge_gap_ms * fs / 1000)

    diff = np.diff(above.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [len(above)]])
    if len(starts) == 0:
        return []

    # Fusionar ráfagas separadas por menos de merge_samples.
    m_starts, m_ends = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - m_ends[-1] < merge_samples:
            m_ends[-1] = e
        else:
            m_starts.append(s); m_ends.append(e)

    bursts = []
    for s, e in zip(m_starts, m_ends):
        if (e - s) < min_samples:
            continue
        bursts.append({"start_s": s / fs, "end_s": e / fs,
                       "duration_ms": (e - s) / fs * 1000,
                       "peak_amplitude": float(np.max(envelope[s:e]))})
    return bursts


def burst_summary(bursts, duration_s):
    """Resume las métricas de ráfaga de una banda/registro."""
    if not bursts:
        return {"n_bursts": 0, "burst_rate_hz": 0.0, "mean_duration_ms": np.nan,
                "mean_amplitude": np.nan, "burst_fraction": 0.0}
    dur = np.array([b["duration_ms"] for b in bursts])
    amp = np.array([b["peak_amplitude"] for b in bursts])
    total = sum(b["end_s"] - b["start_s"] for b in bursts)
    return {"n_bursts": len(bursts), "burst_rate_hz": len(bursts) / duration_s,
            "mean_duration_ms": float(dur.mean()), "mean_amplitude": float(amp.mean()),
            "burst_fraction": total / duration_s}


def run_burst_analysis(signal, fs, cfg, duration_s=None):
    """Detección de ráfagas en todas las bandas de config -> bursts. Devuelve
    un dict plano {banda_metrica: valor} para acumular en la fila del registro."""
    if duration_s is None:
        duration_s = len(signal) / fs
    bcfg = cfg.get("bursts", {})
    det = bcfg.get("detection_bands", {"gamma_lo": [30.0, 60.0], "gamma_hi": [60.0, 160.0]})
    row = {}
    for name, (lo, hi) in det.items():
        bursts = detect_bursts(signal, fs, lo, hi,
                               threshold_sd=bcfg.get("threshold_sd", 2.0),
                               min_duration_ms=bcfg.get("min_duration_ms", 50.0),
                               merge_gap_ms=bcfg.get("merge_gap_ms", 25.0))
        for k, v in burst_summary(bursts, duration_s).items():
            row[f"burst_{name}_{k}"] = v
    return row
