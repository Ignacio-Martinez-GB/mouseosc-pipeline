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


def test_seleccion_adaptativa_de_prueba():
    """Con datos normales, 'auto' elige una prueba paramétrica (t);
    con datos claramente no normales, elige Mann-Whitney."""
    import pandas as pd
    rng = np.random.default_rng(21)
    cfg = {"statistics": {"group_col": "g", "metodo": "auto", "alpha": 0.05,
                          "correction": "holm", "paired": False,
                          "alpha_supuestos": 0.05, "min_n_per_group": 3}}
    # normal
    a = rng.normal(10, 1, 20); b = rng.normal(12, 1, 20)
    dfn = pd.DataFrame({"x": np.r_[a, b], "g": ["A"] * 20 + ["B"] * 20})
    r = stats.compare_metric(dfn, "x", cfg)
    assert "t de" in r.iloc[0]["test"], f"esperaba prueba t, salió {r.iloc[0]['test']}"
    # claramente no normal (exponencial muy sesgada)
    a2 = rng.exponential(1, 30); b2 = rng.exponential(3, 30)
    dfe = pd.DataFrame({"x": np.r_[a2, b2], "g": ["A"] * 30 + ["B"] * 30})
    r2 = stats.compare_metric(dfe, "x", cfg)
    assert "Mann-Whitney" in r2.iloc[0]["test"], f"esperaba MW, salió {r2.iloc[0]['test']}"
    # y reporta los supuestos
    assert "levene_p" in r.columns and "shapiro_p" in r.columns


def test_factorial_detecta_interaccion():
    """Diseño 2×2 con interacción real: el término de interacción sale significativo."""
    import pandas as pd
    rng = np.random.default_rng(22)
    filas = []
    for dieta in ("control", "obeso"):
        for sexo in ("hembra", "macho"):
            # interacción: el efecto de la dieta solo existe en machos
            base = 10 + (5 if (dieta == "obeso" and sexo == "macho") else 0)
            for v in rng.normal(base, 1.0, 12):
                filas.append({"y": v, "dieta": dieta, "sexo": sexo})
    df = pd.DataFrame(filas)
    cfg = {"statistics": {"metodo": "auto", "alpha": 0.05, "alpha_supuestos": 0.05}}
    res = stats.factorial_analysis(df, "y", ["dieta", "sexo"], cfg)
    assert len(res) >= 3, "faltan efectos (2 principales + interacción)"
    inter = res[res["efecto"].str.contains("×")]
    assert len(inter) == 1 and bool(inter.iloc[0]["significant"]), \
        "no detectó la interacción dieta × sexo"


def test_tukey_posthoc_tres_grupos():
    """Con 3 grupos y posthoc='tukey', se reportan las 3 parejas con p ajustado
    y detecta el grupo que difiere."""
    import pandas as pd
    rng = np.random.default_rng(23)
    df = pd.DataFrame({
        "x": np.r_[rng.normal(10, 1, 15), rng.normal(10.2, 1, 15), rng.normal(15, 1, 15)],
        "g": ["A"] * 15 + ["B"] * 15 + ["C"] * 15})
    cfg = {"statistics": {"group_col": "g", "metodo": "parametrico", "alpha": 0.05,
                          "correction": "holm", "paired": False, "posthoc": "tukey",
                          "alpha_supuestos": 0.05, "min_n_per_group": 3}}
    r = stats.compare_metric(df, "x", cfg)
    tuk = r[r["test"].str.contains("Tukey")]
    assert len(tuk) == 3, "esperaba 3 parejas de Tukey"
    sig = tuk[tuk["significant"]]["comparison"].tolist()
    assert any("C" in s for s in sig), "no detectó el grupo C distinto"
    assert not any(s == "A vs B" for s in sig), "falso positivo A vs B"


def test_posthoc_celdas_factorial():
    """El post-hoc de celdas compara las combinaciones del cruce (2×2 = 6 pares)."""
    import pandas as pd
    rng = np.random.default_rng(24)
    filas = []
    for dieta in ("control", "obeso"):
        for sexo in ("hembra", "macho"):
            base = 10 + (5 if (dieta == "obeso" and sexo == "macho") else 0)
            for v in rng.normal(base, 1.0, 10):
                filas.append({"y": v, "dieta": dieta, "sexo": sexo})
    df = pd.DataFrame(filas)
    cfg = {"statistics": {"metodo": "auto", "alpha": 0.05, "correction": "holm",
                          "alpha_supuestos": 0.05, "posthoc": "auto"}}
    ph = stats.factorial_posthoc(df, "y", ["dieta", "sexo"], cfg)
    assert len(ph) == 6, f"esperaba 6 parejas de celdas, salieron {len(ph)}"
    # la celda obeso·macho debe diferir de las demás
    sig = ph[ph["significant"]]["comparison"].tolist()
    assert any("obeso·macho" in s for s in sig), "no detectó la celda distinta"


def test_figura_celdas_factoriales(tmp_path):
    """La figura del cruce (sexo × dieta × condición) se genera con las 8 celdas
    y una familia de color distinta por sexo."""
    import pandas as pd
    from mouseosc import viz, style
    rng = np.random.default_rng(31)
    filas = [{"y": v, "sexo": s, "dieta": d, "condicion": c}
             for s in ("hembra", "macho") for d in ("control", "obeso")
             for c in ("foto", "meso") for v in rng.normal(10, 1, 6)]
    df = pd.DataFrame(filas)
    cfg = {"bands": {}, "output": {"dpi": 60}, "plotting": {}}
    out = tmp_path / "celdas.png"
    viz.plot_factorial_cells(df, "y", ["sexo", "dieta", "condicion"], cfg, out)
    assert out.exists() and out.stat().st_size > 5000, "no se generó la figura"
    fams = style.family_colors(["hembra", "macho"], 4)
    assert fams["hembra"][0] != fams["macho"][0], "las familias de color coinciden"


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
