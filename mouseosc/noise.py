"""
===============================================================================
RUIDO ELÉCTRICO — detección, referencia y corrección
===============================================================================

Maneja el ruido de línea (p. ej. 10 Hz + armónicos) para tres análisis:
  1. NORMAL         : todos los archivos tal cual.
  2. SIN RUIDO      : se excluyen los archivos contaminados.
  3. CORREGIDO      : a los contaminados se les RESTA el espectro promedio del
                      ruido (de la carpeta de ruido) en los armónicos; para PAC
                      y bursts se aplica un NOTCH en esos armónicos.

Además, el ruido de 60 Hz (línea eléctrica) se suprime SIEMPRE con notch en el
preprocesado (ver preprocessing.notch_filter), en los tres análisis.

DETECCIÓN (método reportado)
----------------------------
Por cada armónico se mide la relación señal-ruido local (prominencia del pico
respecto al fondo vecino) sobre el PSD, y la PERSISTENCIA temporal (fracción de
épocas donde el pico está presente). El ruido de línea es un pico angosto,
equiespaciado (peine) y estacionario en el tiempo — eso lo distingue de una
oscilación fisiológica. Se marca "contaminado" si varios armónicos superan el
umbral de SNR y de persistencia. (Referencias: ZapLine de Cheveigné 2020;
ZapLine-plus Klug 2022; test F de Thomson para la versión estadística.)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def harmonic_freqs(fundamental, n, freq_max):
    """Frecuencias fundamental·1..n que caen por debajo de freq_max."""
    return [fundamental * k for k in range(1, n + 1) if fundamental * k < freq_max]


def _snr_at(freqs, psd, f0, half_bw):
    """Prominencia del pico en [f0±half_bw] respecto al fondo vecino.
    SNR = pico en la banda / mediana del fondo (bins a ambos lados, sin el pico)."""
    band = (freqs >= f0 - half_bw) & (freqs <= f0 + half_bw)
    neigh = (((freqs >= f0 - 4 * half_bw) & (freqs < f0 - 2 * half_bw)) |
             ((freqs > f0 + 2 * half_bw) & (freqs <= f0 + 4 * half_bw)))
    if band.sum() == 0 or neigh.sum() == 0:
        return np.nan
    peak = np.nanmax(psd[band])
    bg = np.nanmedian(psd[neigh])
    return float(peak / bg) if bg > 0 else np.nan


def detect_contamination(freqs, psd, epochs, fs, cfg):
    """
    Decide si UN registro está contaminado por ruido de línea.

    Devuelve dict: {flag, harmonics: [{f, snr, persistencia}], n_hits}.
    """
    ncfg = cfg.get("noise", {})
    f0 = ncfg.get("fundamental_hz", 10.0)
    n = ncfg.get("n_armonicos", 6)
    fmax = ncfg.get("freq_max", 200.0)
    half_bw = ncfg.get("ancho_hz", 0.5)
    det = ncfg.get("deteccion", {})
    snr_thr = det.get("snr_umbral", 3.0)
    pers_thr = det.get("persistencia_temporal", 0.5)
    min_h = det.get("min_armonicos", 2)

    harms = harmonic_freqs(f0, n, min(fmax, fs / 2))
    # SNR por época (para persistencia): reutilizamos las épocas ya limpias.
    per_epoch = []
    win = int(min(len(epochs[0]) if len(epochs) else 0, 4 * fs))
    for ep in epochs[: min(len(epochs), 60)]:      # tope de 60 épocas por velocidad
        fe, pe = welch(ep, fs=fs, window="hann",
                       nperseg=min(len(ep), int(2 * fs)), scaling="density")
        per_epoch.append((fe, pe))

    results, hits = [], 0
    for h in harms:
        snr = _snr_at(freqs, psd, h, half_bw)
        # persistencia: fracción de épocas con SNR local por encima del umbral
        pres = 0
        for fe, pe in per_epoch:
            s = _snr_at(fe, pe, h, half_bw)
            if s == s and s >= snr_thr:
                pres += 1
        persistencia = pres / len(per_epoch) if per_epoch else 0.0
        contaminado_h = (snr == snr and snr >= snr_thr and persistencia >= pers_thr)
        if contaminado_h:
            hits += 1
        results.append({"f": h, "snr": snr, "persistencia": persistencia,
                        "contaminado": contaminado_h})
    return {"flag": hits >= min_h, "n_hits": hits, "harmonics": results}


def build_noise_reference(recs, cfg, load_signal_fn, preprocess_fn):
    """
    Referencia de ruido = PSD PROMEDIO de todos los archivos de la carpeta de
    ruido. Se preprocesa igual que los datos (detrend + paso-alto + notch 60 Hz),
    pero SIN tocar el fundamental de 10 Hz (que es justo lo que queremos medir).

    Devuelve (freqs, psd_medio) o (None, None) si no hay archivos.
    """
    if not recs:
        return None, None
    sp = cfg.get("spectral", {}); w = sp.get("welch", {})
    fs = cfg["preprocessing"]["fs"]
    window_s = sp.get("primary_window_s", 2.0)
    from .spectral import compute_psd_welch
    from .preprocessing import full_pipeline
    psds, freqs = [], None
    for rec in recs:
        try:
            sig, fs_file = load_signal_fn(rec, cfg)
            pp = full_pipeline(sig, fs, cfg["preprocessing"])
            f, p = compute_psd_welch(pp["epochs"], fs, window_s=window_s,
                                     overlap=w.get("overlap", 0.5),
                                     freq_min=w.get("freq_min", 0.5),
                                     freq_max=w.get("freq_max", 500.0))
            psds.append(p); freqs = f
        except Exception:
            continue
    if not psds:
        return None, None
    return freqs, np.mean(psds, axis=0)


def _harmonic_bands(freqs, cfg):
    """Genera, por cada armónico, las máscaras (banda del pico, fondo vecino)."""
    ncfg = cfg.get("noise", {})
    f0 = ncfg.get("fundamental_hz", 10.0); n = ncfg.get("n_armonicos", 6)
    fmax = ncfg.get("freq_max", 200.0); hb = ncfg.get("ancho_hz", 0.5)
    for h in harmonic_freqs(f0, n, fmax):
        band = (freqs >= h - hb) & (freqs <= h + hb)
        neigh = (((freqs >= h - 4 * hb) & (freqs < h - 2 * hb)) |
                 ((freqs > h + 2 * hb) & (freqs <= h + 4 * hb)))
        yield h, hb, band, neigh


def subtract_noise_psd(freqs, psd, ref_freqs, ref_psd, cfg):
    """(a) LITERAL: resta el PSD de ruido tal cual en las bandas de los armónicos,
    con piso en el fondo local."""
    ref_i = np.interp(freqs, ref_freqs, ref_psd)
    out = psd.copy()
    for _h, _hb, band, neigh in _harmonic_bands(freqs, cfg):
        floor = np.nanmedian(psd[neigh]) if neigh.sum() else 0.0
        out[band] = np.maximum(psd[band] - ref_i[band], floor)
    return out


def scaled_subtract_psd(freqs, psd, ref_freqs, ref_psd, cfg):
    """(b) ESCALADA: escala la FORMA del ruido para igualar el exceso del pico del
    registro sobre su fondo local, y lo resta (baja el pico hasta la base)."""
    ref_i = np.interp(freqs, ref_freqs, ref_psd)
    out = psd.copy()
    for _h, _hb, band, neigh in _harmonic_bands(freqs, cfg):
        if band.sum() == 0 or neigh.sum() == 0:
            continue
        bg = np.nanmedian(psd[neigh])
        exc_file = np.nanmax(psd[band]) - bg
        ref_bg = np.nanmedian(ref_i[neigh])
        exc_ref = np.nanmax(ref_i[band]) - ref_bg
        scale = (exc_file / exc_ref) if exc_ref > 0 and exc_file > 0 else 0.0
        out[band] = np.maximum(psd[band] - scale * (ref_i[band] - ref_bg), bg)
    return out


def interpolate_psd(freqs, psd, cfg):
    """(c) INTERPOLACIÓN espectral (Leske & Dalal 2019): reemplaza los bins de cada
    armónico por una recta entre los bordes del fondo vecino. Elimina el pico por
    completo y NO necesita los archivos de ruido."""
    out = psd.copy()
    for _h, _hb, band, neigh in _harmonic_bands(freqs, cfg):
        if band.sum() == 0:
            continue
        idx = np.where(band)[0]
        lo_i, hi_i = idx[0] - 1, idx[-1] + 1
        if lo_i < 0 or hi_i >= len(freqs):
            continue
        out[band] = np.interp(freqs[band], [freqs[lo_i], freqs[hi_i]],
                              [psd[lo_i], psd[hi_i]])
    return out


def correct_psd(freqs, psd, ref_freqs, ref_psd, cfg):
    """Aplica el método de corrección elegido en config.noise.metodo_correccion:
    'literal' | 'escalada' | 'interpolacion' (por defecto)."""
    metodo = cfg.get("noise", {}).get("metodo_correccion", "interpolacion")
    if metodo == "literal":
        return subtract_noise_psd(freqs, psd, ref_freqs, ref_psd, cfg)
    if metodo == "escalada":
        return scaled_subtract_psd(freqs, psd, ref_freqs, ref_psd, cfg)
    return interpolate_psd(freqs, psd, cfg)
