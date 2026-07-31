"""
===============================================================================
TESTS DE VERDAD CONOCIDA
===============================================================================

Cada test construye una señal donde SABEMOS la respuesta y verifica que el
cálculo la recupera. Si un cálculo se rompe, el test falla — no te enteras
tres papers después.

Correr:  pytest -v        (desde la raíz del proyecto)
"""
import numpy as np
import pytest

from mouseosc import preprocessing as pp, spectral, bands, pac, bursts, stats, checks, noise


FS = 2000
DUR = 20
T = np.arange(int(FS * DUR)) / FS


# ---------------------------------------------------------------------------
def test_welch_pico_en_frecuencia_conocida():
    """Una sinusoide de 40 Hz debe dar el pico del PSD EXACTAMENTE en 40 Hz."""
    sig = np.sin(2 * np.pi * 40 * T)
    epochs = pp.make_epochs(sig, FS, length_s=2.0, overlap=0.5)
    freqs, psd = spectral.compute_psd_welch(epochs, FS, window_s=2.0)
    f_peak = freqs[np.argmax(psd)]
    assert abs(f_peak - 40.0) < 0.6, f"pico en {f_peak} Hz, esperado 40 Hz"


def test_band_power_localiza_la_banda_correcta():
    """Tono en 45 Hz: la banda gamma_lo (30-60) debe llevarse casi toda la potencia."""
    sig = np.sin(2 * np.pi * 45 * T)
    epochs = pp.make_epochs(sig, FS, length_s=2.0)
    freqs, psd = spectral.compute_psd_welch(epochs, FS, window_s=2.0)
    p_gamma = bands.band_power(freqs, psd, 30, 60)
    p_beta = bands.band_power(freqs, psd, 12, 30)
    assert p_gamma > 50 * p_beta, "la potencia no se concentró en la banda correcta"


def test_pac_detecta_acoplamiento_y_lo_niega_en_ruido():
    """Señal con PAC theta→gamma sintético da MI alto; ruido blanco da MI ~0."""
    theta = np.sin(2 * np.pi * 8 * T)
    gamma = (1 + 0.8 * np.sin(2 * np.pi * 8 * T)) * np.sin(2 * np.pi * 45 * T)
    coupled = theta + gamma
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(T))

    ph_c, amp_c = pac._phase_amp(coupled, FS, [6, 10], [35, 55])
    ph_n, amp_n = pac._phase_amp(noise, FS, [6, 10], [35, 55])
    mi_coupled = pac.modulation_index(ph_c, amp_c)
    mi_noise = pac.modulation_index(ph_n, amp_n)
    assert mi_coupled > 10 * mi_noise, f"MI acoplado {mi_coupled} vs ruido {mi_noise}"


def test_mvl_recupera_fase_preferida():
    """La fase preferida del MVL debe coincidir con el máximo de amplitud impuesto.

    theta = sin(2π·8t); la fase analítica (Hilbert) de un seno es x − π/2.
    La amplitud de gamma es máxima cuando cos(2π·8t)=1 → 2π·8t=0 → fase = −π/2.
    Así que la fase preferida TEÓRICA es −π/2 ≈ −1.5708 rad.
    """
    theta = np.sin(2 * np.pi * 8 * T)
    gamma = (1 + 0.9 * np.cos(2 * np.pi * 8 * T)) * np.sin(2 * np.pi * 45 * T)
    sig = theta + gamma
    ph, amp = pac._phase_amp(sig, FS, [6, 10], [35, 55])
    _, pref = pac.mean_vector_length(ph, amp)
    assert abs(pref - (-np.pi / 2)) < 0.6, f"fase preferida {pref} rad, esperada −π/2"


def test_stats_detecta_diferencia_y_respeta_nulo():
    """Mann-Whitney: detecta una diferencia real; NO marca significancia sin ella."""
    rng = np.random.default_rng(1)
    df_diff = _toy_df(rng, mean_a=1.0, mean_b=3.0)
    df_same = _toy_df(rng, mean_a=1.0, mean_b=1.0)
    cfg = {"statistics": {"group_col": "group", "paired": False, "alpha": 0.05,
                          "correction": "holm", "min_n_per_group": 3}}
    r_diff = stats.compare_metric(df_diff, "x", cfg)
    r_same = stats.compare_metric(df_same, "x", cfg)
    assert bool(r_diff.iloc[0]["significant"]), "no detectó diferencia real"
    assert not bool(r_same.iloc[0]["significant"]), "falso positivo bajo el nulo"


def test_artifact_rejection_descarta_epoca_con_spike():
    """Una época con un spike gigante debe ser rechazada; las limpias se conservan."""
    rng = np.random.default_rng(2)
    sig = rng.standard_normal(FS * 10)
    sig[5000:5010] += 100      # spike enorme
    epochs = pp.make_epochs(sig, FS, length_s=1.0, overlap=0.0)
    clean, mask = pp.reject_artifacts(epochs, threshold_sd=8.0)
    assert mask.sum() < len(mask), "no rechazó la época con spike"
    assert mask.sum() >= len(mask) - 2, "rechazó demasiadas épocas limpias"


def test_bursts_cuenta_rafagas_insertadas():
    """Inserto 3 ráfagas de 45 Hz en una señal silenciosa; el detector debe
    encontrar ~3 (la fusión de gaps puede unir alguna, así que aceptamos 2–3)."""
    sig = 0.01 * np.random.default_rng(3).standard_normal(int(FS * 6))
    for c in (1.0, 3.0, 5.0):                      # 3 ráfagas a t=1,3,5 s
        i0 = int(c * FS); seg = np.arange(int(0.2 * FS)) / FS
        sig[i0:i0 + len(seg)] += np.sin(2 * np.pi * 45 * seg)
    found = bursts.detect_bursts(sig, FS, 35, 55, threshold_sd=2.0,
                                 min_duration_ms=50, merge_gap_ms=25)
    assert 2 <= len(found) <= 3, f"detectó {len(found)} ráfagas, esperaba ~3"


def test_analysis_band_recorta_extremos():
    """El rango global recorta solo los extremos: bandas interiores intactas,
    las que cruzan el límite se recortan, las de fuera se eliminan."""
    cfg = {"analysis_band": [0.4, 160.0],
           "bands": {"slow": [0.5, 2.0], "gamma_hi": [60.0, 200.0], "mua": [300.0, 500.0]},
           "spectral": {"welch": {"freq_min": 0.5, "freq_max": 500.0},
                        "specparam": {"freq_range": [1.0, 300.0]}}}
    bands.apply_analysis_band(cfg)
    assert cfg["bands"]["slow"] == [0.5, 2.0]          # interior: intacta
    assert cfg["bands"]["gamma_hi"] == [60.0, 160.0]   # cruza el límite: recortada
    assert "mua" not in cfg["bands"]                    # fuera: eliminada
    assert cfg["spectral"]["welch"]["freq_max"] == 160.0
    assert cfg["spectral"]["specparam"]["freq_range"][1] == 160.0


def test_check_bandas_detecta_solape():
    """El check de bandas debe marcar 'warn' cuando dos bandas se solapan."""
    cfg = {"bands": {"a": [1, 5], "b": [4, 10]}, "relative_power": {}}
    res = checks.check_band_definitions(cfg)
    assert any(c.name == "bandas_sin_solape" and c.level == "warn" for c in res)


def test_regresion_psd_numeros_de_referencia():
    """Test de REGRESIÓN: fija números de referencia para un caso determinista.
    Si una actualización de librería mueve estos valores, el test avisa."""
    rng = np.random.default_rng(42)
    sig = np.sin(2 * np.pi * 40 * T) + 0.1 * rng.standard_normal(len(T))
    epochs = pp.make_epochs(sig, FS, length_s=2.0, overlap=0.5)
    freqs, psd = spectral.compute_psd_welch(epochs, FS, window_s=2.0)
    f_peak = float(freqs[np.argmax(psd)])
    p_gamma = bands.band_power(freqs, psd, 30, 60)
    # Referencias capturadas en la implementación validada (valor medido ≈0.500).
    assert abs(f_peak - 40.0) < 0.6
    assert 0.45 < p_gamma < 0.55, f"potencia gamma fuera de referencia: {p_gamma}"


_NOISE_CFG = {"noise": {"fundamental_hz": 10.0, "n_armonicos": 4, "freq_max": 100,
                        "ancho_hz": 0.5,
                        "deteccion": {"snr_umbral": 3.0, "min_armonicos": 2,
                                      "persistencia_temporal": 0.4}}}


def test_deteccion_ruido_flag_y_nulo():
    """Una señal con línea de 10 Hz + armónicos se marca contaminada; ruido
    blanco no."""
    rng = np.random.default_rng(7)
    linea = sum(np.sin(2 * np.pi * (10 * k) * T) for k in (1, 2, 3))
    contaminada = 0.2 * rng.standard_normal(len(T)) + 3 * linea
    limpia = rng.standard_normal(len(T))
    for sig, esperado in ((contaminada, True), (limpia, False)):
        ep = pp.make_epochs(sig, FS, 2.0, 0.5)
        f, psd = spectral.compute_psd_welch(ep, FS, window_s=2.0)
        res = noise.detect_contamination(f, psd, ep, FS, _NOISE_CFG)
        assert res["flag"] is esperado, f"detección incorrecta (esperaba {esperado})"


def test_resta_ruido_baja_potencia_en_armonicos():
    """La resta espectral reduce la potencia en 10 Hz respecto al original."""
    rng = np.random.default_rng(8)
    sig = 0.2 * rng.standard_normal(len(T)) + 2 * np.sin(2 * np.pi * 10 * T)
    ep = pp.make_epochs(sig, FS, 2.0, 0.5)
    f, psd = spectral.compute_psd_welch(ep, FS, window_s=2.0)
    ref = psd.copy()                      # referencia = mismo espectro (caso extremo)
    corr = noise.subtract_noise_psd(f, psd, f, ref, _NOISE_CFG)
    i10 = np.argmin(np.abs(f - 10))
    assert corr[i10] < psd[i10], "la resta no redujo el pico de 10 Hz"


def test_metodos_correccion_reducen_pico():
    """Los 3 métodos bajan el pico de 10 Hz; la interpolación lo deja ~al fondo."""
    rng = np.random.default_rng(11)
    sig = 0.2 * rng.standard_normal(len(T)) + 2 * np.sin(2 * np.pi * 10 * T)
    ep = pp.make_epochs(sig, FS, 2.0, 0.5)
    f, psd = spectral.compute_psd_welch(ep, FS, window_s=2.0)
    i10 = np.argmin(np.abs(f - 10))
    ref = psd.copy()
    lit = noise.subtract_noise_psd(f, psd, f, ref, _NOISE_CFG)
    esc = noise.scaled_subtract_psd(f, psd, f, ref, _NOISE_CFG)
    itp = noise.interpolate_psd(f, psd, _NOISE_CFG)
    for name, corr in (("literal", lit), ("escalada", esc), ("interp", itp)):
        assert corr[i10] < psd[i10], f"{name} no redujo el pico"
    # la interpolación deja el bin cerca del fondo vecino (pico casi eliminado)
    bg = np.median(psd[(f > 12) & (f < 15)])
    assert itp[i10] < 5 * bg, "la interpolación no aplanó el pico"


def test_notch_elimina_linea_60hz():
    """El notch de 60 Hz reduce fuertemente una línea de 60 Hz."""
    sig = np.sin(2 * np.pi * 60 * T) + 0.01 * np.random.default_rng(9).standard_normal(len(T))
    filt = pp.notch_filter(sig, FS, [60.0], Q=30)
    assert np.std(filt) < 0.2 * np.std(sig), "el notch no atenuó los 60 Hz"


# ---------------------------------------------------------------------------
def _toy_df(rng, mean_a, mean_b, n=8):
    import pandas as pd
    a = rng.normal(mean_a, 0.3, n)
    b = rng.normal(mean_b, 0.3, n)
    return pd.DataFrame({"x": np.r_[a, b],
                         "group": ["control"] * n + ["tratamiento"] * n})
