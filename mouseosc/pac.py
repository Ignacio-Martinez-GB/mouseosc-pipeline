"""
===============================================================================
PHASE-AMPLITUDE COUPLING (PAC)
===============================================================================

Mide si la AMPLITUD de un ritmo rápido está modulada por la FASE de uno lento.

  MI  = Modulation Index de Tort et al. 2010. Divergencia KL del histograma de
        amplitud-por-bin-de-fase respecto al uniforme. Normalizado 0–1.
        Robusto a outliers. Típico ~0.001–0.01; >0.05 fuerte.
  MVL = Mean Vector Length de Canolty et al. 2006. |⟨A·e^{iφ}⟩|. Da gratis la
        FASE PREFERIDA. Sensible a outliers de amplitud.

SIGNIFICANCIA: el MI siempre es >0 por ruido. Se compara contra SUBROGADOS
(se desfasa la amplitud en un corte temporal aleatorio y se recalcula). El
p-valor = fracción de subrogados con MI ≥ al observado. Usa la semilla del config.

⚠ La banda de FASE debe elegirse con cuidado: en ratón in vivo 5–12 Hz solapa
con el latido cardiaco. Un PAC con fase cardiaca es real matemáticamente pero
no es acoplamiento neural. Ver contamination_zones en el config.

Refs: Tort 2010 J Neurophysiol 104:1195; Canolty 2006 Science 313:1626.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from .preprocessing import bandpass_filter


def _phase_amp(signal, fs, phase_band, amp_band):
    """Devuelve (fase del ritmo lento, envolvente del rápido) vía Hilbert."""
    ph = np.angle(hilbert(bandpass_filter(signal, fs, *phase_band)))
    amp = np.abs(hilbert(bandpass_filter(signal, fs, *amp_band)))
    return ph, amp


def modulation_index(phase, amp, n_bins=18):
    """MI de Tort: KL del histograma amplitud-por-fase vs uniforme, normalizado."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.digitize(phase, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mean_amp = np.array([amp[idx == b].mean() if np.any(idx == b) else 0.0
                         for b in range(n_bins)])
    p = mean_amp / mean_amp.sum()
    p = np.where(p == 0, 1e-12, p)               # evita log(0)
    kl = np.log(n_bins) + np.sum(p * np.log(p))   # = log N - H(p)
    return float(kl / np.log(n_bins))             # normalizado 0–1


def mean_vector_length(phase, amp):
    """MVL de Canolty: magnitud y fase preferida del vector complejo medio."""
    z = np.mean(amp * np.exp(1j * phase))
    return float(np.abs(z)), float(np.angle(z))


def pac_significance(phase, amp, mi_obs, n_surrogates=200, n_bins=18, rng=None):
    """p-valor del MI por subrogados (desfase temporal de la amplitud)."""
    if n_surrogates <= 0:
        return np.nan
    rng = rng or np.random.default_rng()
    n = len(amp)
    null = np.empty(n_surrogates)
    for i in range(n_surrogates):
        shift = rng.integers(1, n)
        null[i] = modulation_index(phase, np.roll(amp, shift), n_bins)
    return float((np.sum(null >= mi_obs) + 1) / (n_surrogates + 1))


def run_pac_analysis(signal, fs, cfg):
    """Calcula MI, MVL, fase preferida y p-valor para cada par del config."""
    pac_cfg = cfg.get("pac", {})
    n_bins = pac_cfg.get("n_phase_bins", 18)
    n_surr = pac_cfg.get("n_surrogates", 200)
    rng = np.random.default_rng(cfg.get("project", {}).get("seed", None))
    rows = []
    for pair in pac_cfg.get("pairs", []):
        ph, amp = _phase_amp(signal, fs, pair["phase"], pair["amp"])
        mi = modulation_index(ph, amp, n_bins)
        mvl, pref = mean_vector_length(ph, amp)
        pval = pac_significance(ph, amp, mi, n_surr, n_bins, rng)
        rows.append({"pair": pair["name"], "mi": mi, "mvl": mvl,
                     "preferred_phase_rad": pref, "p_value": pval,
                     "phase_band": tuple(pair["phase"]), "amp_band": tuple(pair["amp"])})
    return rows


def compute_comodulogram(signal, fs, cfg):
    """Mapa MI sobre una grilla fase×amplitud (caro). Devuelve (phases, amps, MI)."""
    c = cfg.get("pac", {}).get("comodulogram", {})
    n_bins = cfg.get("pac", {}).get("n_phase_bins", 18)
    phases = np.arange(*c.get("phase_range", [1, 14]), c.get("phase_step", 1.0))
    amps = np.arange(*c.get("amp_range", [20, 120]), c.get("amp_step", 10.0))
    pw = c.get("phase_step", 1.0)
    aw = c.get("amp_step", 10.0)
    mi_grid = np.zeros((len(amps), len(phases)))
    for j, pf in enumerate(phases):
        ph = np.angle(hilbert(bandpass_filter(signal, fs, pf, pf + pw)))
        for i, af in enumerate(amps):
            amp = np.abs(hilbert(bandpass_filter(signal, fs, af, af + aw)))
            mi_grid[i, j] = modulation_index(ph, amp, n_bins)
    return phases, amps, mi_grid
