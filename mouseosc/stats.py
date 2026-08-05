"""
===============================================================================
ESTADÍSTICA DE GRUPOS
===============================================================================

CÓMO SE DECIDE LA PRUEBA  (config -> statistics.metodo)
-------------------------------------------------------
  "auto" (por defecto)  Se comprueban los SUPUESTOS y se elige en consecuencia:
        1. Normalidad  : Shapiro-Wilk en cada grupo  (p > alpha_supuestos → normal)
        2. Homocedast. : Levene entre grupos         (p > alpha_supuestos → varianzas iguales)
        → si NORMAL y VARIANZAS IGUALES  : t de Student / ANOVA de una vía
        → si NORMAL y varianzas DESIGUALES: t de Welch  / ANOVA de Welch
        → si NO normal                    : Mann-Whitney / Kruskal-Wallis
  "parametrico"      Fuerza t/ANOVA (con Welch si las varianzas difieren).
  "no_parametrico"   Fuerza Mann-Whitney / Kruskal-Wallis (o Wilcoxon/Friedman
                     si el diseño es pareado). Es la opción conservadora.

En TODOS los casos el CSV de salida reporta: qué prueba se usó, el p de Shapiro
de cada grupo y el p de Levene, para que la decisión sea auditable.

NOTA HONESTA: elegir la prueba tras un pre-test de normalidad ("auto") es el flujo
convencional en biología (y el que sugiere GraphPad Prism), pero con n pequeño
Shapiro tiene poca potencia y el condicionamiento altera levemente la tasa de
error tipo I. Por eso se reportan siempre los supuestos y se puede fijar el
método a priori con "parametrico"/"no_parametrico".

QUÉ COMPARACIONES SE HACEN
--------------------------
  1. OMNIBUS por columna         : ¿difiere algún nivel? (2 grupos = la prueba directa)
  2. PARES (post-hoc)            : todas las parejas, con corrección múltiple
                                   (holm por defecto) → p_corrected.
  3. FACTORIAL (opcional)        : ANOVA de N vías con EFECTOS PRINCIPALES e
                                   INTERACCIONES (p. ej. dieta × sexo). Versión no
                                   paramétrica por Aligned Rank Transform (ART).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# pandas avisa al concatenar DataFrames vacíos; no afecta el resultado.
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
from itertools import combinations
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# SUPUESTOS
# ---------------------------------------------------------------------------
def check_assumptions(groups_data, alpha=0.05):
    """
    Comprueba normalidad (Shapiro-Wilk por grupo) y homocedasticidad (Levene).

    Devuelve dict con:
      normal            : True si TODOS los grupos pasan Shapiro (p > alpha)
      equal_var         : True si Levene no rechaza (p > alpha)
      shapiro_p         : lista de p por grupo (nan si n<3)
      levene_p          : p de Levene (nan si no calculable)
    """
    shapiro_p = []
    for g in groups_data:
        g = np.asarray(g, dtype=float)
        if len(g) >= 3 and np.ptp(g) > 0:
            try:
                shapiro_p.append(float(sps.shapiro(g).pvalue))
            except Exception:
                shapiro_p.append(np.nan)
        else:
            shapiro_p.append(np.nan)   # n insuficiente para probar normalidad
    valid = [p for p in shapiro_p if p == p]
    normal = bool(valid) and all(p > alpha for p in valid)
    try:
        levene_p = float(sps.levene(*[np.asarray(g, float) for g in groups_data
                                      if len(g) >= 2]).pvalue)
    except Exception:
        levene_p = np.nan
    equal_var = (levene_p != levene_p) or (levene_p > alpha)   # nan → asumir iguales
    return {"normal": normal, "equal_var": bool(equal_var),
            "shapiro_p": shapiro_p, "levene_p": levene_p}


# ---------------------------------------------------------------------------
# PRUEBA PRINCIPAL (omnibus) con selección adaptativa
# ---------------------------------------------------------------------------
def _omnibus(groups_data, paired, metodo="no_parametrico", alpha_sup=0.05):
    """
    Devuelve (nombre_prueba, estadístico, p, info_supuestos).
    `metodo`: auto | parametrico | no_parametrico.
    """
    k = len(groups_data)
    sup = check_assumptions(groups_data, alpha_sup)

    usar_parametrico = (metodo == "parametrico") or \
                       (metodo == "auto" and sup["normal"])

    if k == 2:
        a, b = [np.asarray(x, float) for x in groups_data]
        if paired:
            if usar_parametrico:
                s, p = sps.ttest_rel(a, b); return "t pareada", s, p, sup
            s, p = sps.wilcoxon(a, b); return "Wilcoxon", s, p, sup
        if usar_parametrico:
            if sup["equal_var"]:
                s, p = sps.ttest_ind(a, b, equal_var=True)
                return "t de Student", s, p, sup
            s, p = sps.ttest_ind(a, b, equal_var=False)
            return "t de Welch", s, p, sup
        s, p = sps.mannwhitneyu(a, b, alternative="two-sided")
        return "Mann-Whitney U", s, p, sup

    # ≥3 grupos
    if paired:
        if usar_parametrico:
            s, p = sps.friedmanchisquare(*groups_data)   # sin RM-ANOVA: se reporta Friedman
            return "Friedman", s, p, sup
        s, p = sps.friedmanchisquare(*groups_data); return "Friedman", s, p, sup
    if usar_parametrico:
        if sup["equal_var"]:
            s, p = sps.f_oneway(*groups_data); return "ANOVA 1 vía", s, p, sup
        try:
            s, p = sps.alexandergovern(*groups_data)[:2]
            return "ANOVA de Welch", s, p, sup
        except Exception:
            s, p = sps.f_oneway(*groups_data); return "ANOVA 1 vía", s, p, sup
    s, p = sps.kruskal(*groups_data); return "Kruskal-Wallis", s, p, sup


def _pair_test(a, b, paired, metodo, alpha_sup):
    """Prueba para UNA pareja, con la misma lógica de selección. Devuelve (nombre, p)."""
    sup = check_assumptions([a, b], alpha_sup)
    usar_par = (metodo == "parametrico") or (metodo == "auto" and sup["normal"])
    try:
        if paired and len(a) == len(b):
            if usar_par:
                return "t pareada", float(sps.ttest_rel(a, b).pvalue)
            return "Wilcoxon", float(sps.wilcoxon(a, b).pvalue)
        if usar_par:
            ev = sup["equal_var"]
            return ("t de Student" if ev else "t de Welch"), \
                   float(sps.ttest_ind(a, b, equal_var=ev).pvalue)
        return "Mann-Whitney U", float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return "n/a", np.nan


# ---------------------------------------------------------------------------
# COMPARACIÓN POR COLUMNA: omnibus + pares corregidos
# ---------------------------------------------------------------------------
def compare_metric(df, metric, cfg):
    st = cfg.get("statistics", {})
    gcol = st.get("group_col", "group")
    paired = st.get("paired", False)
    alpha = st.get("alpha", 0.05)
    correction = st.get("correction", "holm")
    min_n = st.get("min_n_per_group", 3)
    metodo = st.get("metodo", "auto")
    alpha_sup = st.get("alpha_supuestos", 0.05)

    levels = sorted(df[gcol].dropna().unique())
    data = {g: df.loc[df[gcol] == g, metric].dropna().values for g in levels}
    ns = {g: len(v) for g, v in data.items()}

    rows = []
    name, stat, p, sup = _omnibus(list(data.values()), paired, metodo, alpha_sup)
    rows.append({"metric": metric, "test": name, "comparison": "omnibus",
                 "statistic": float(stat), "p_value": float(p),
                 "p_corrected": float(p), "significant": bool(p < alpha),
                 "n": dict(ns),
                 "normal(Shapiro)": sup["normal"],
                 "shapiro_p": [round(x, 4) if x == x else None for x in sup["shapiro_p"]],
                 "varianzas_iguales(Levene)": sup["equal_var"],
                 "levene_p": round(sup["levene_p"], 4) if sup["levene_p"] == sup["levene_p"] else None,
                 "warning": "" if ns and min(ns.values()) >= min_n
                 else f"n<{min_n} en algún grupo"})

    # PARES (post-hoc) — siempre que haya ≥3 grupos (con 2, el omnibus ya es el par)
    if len(levels) >= 3:
        # Tukey HSD: post-hoc clásico de ANOVA (solo si la prueba fue paramétrica
        # y el usuario lo pidió). Controla el error familiar sin corrección extra.
        posthoc = st.get("posthoc", "auto")
        if posthoc in ("tukey", "auto") and name.startswith(("ANOVA", "t de")) \
           and not paired and posthoc == "tukey":
            rows += _tukey_rows(data, levels, metric, alpha, ns)
            return pd.DataFrame(rows)

        pairs = list(combinations(levels, 2))
        names, raw_p = [], []
        for a, b in pairs:
            nm, pv = _pair_test(data[a], data[b], paired, metodo, alpha_sup)
            names.append(nm); raw_p.append(pv)
        ok = [i for i, v in enumerate(raw_p) if v == v]
        p_corr = list(raw_p)
        if correction != "none" and len(ok) > 1:
            _, pc, _, _ = multipletests([raw_p[i] for i in ok], alpha=alpha, method=correction)
            for i, v in zip(ok, pc):
                p_corr[i] = v
        for (a, b), nm, pv, pc in zip(pairs, names, raw_p, p_corr):
            rows.append({"metric": metric, "test": f"post-hoc: {nm}",
                         "comparison": f"{a} vs {b}",
                         "statistic": np.nan, "p_value": pv, "p_corrected": pc,
                         "significant": bool(pc == pc and pc < alpha),
                         "n": {a: ns[a], b: ns[b]}, "warning": ""})
    return pd.DataFrame(rows)


def _tukey_rows(data, levels, metric, alpha, ns):
    """Post-hoc de Tukey HSD (todas las parejas, error familiar ya controlado).
    Es el post-hoc clásico tras un ANOVA paramétrico."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    vals = np.concatenate([data[g] for g in levels])
    grp = np.concatenate([[g] * len(data[g]) for g in levels])
    res = pairwise_tukeyhsd(vals, grp, alpha=alpha)
    rows = []
    for r in res.summary().data[1:]:
        a, b, _md, p_adj = r[0], r[1], r[2], float(r[3])
        rows.append({"metric": metric, "test": "post-hoc: Tukey HSD",
                     "comparison": f"{a} vs {b}", "statistic": np.nan,
                     "p_value": p_adj, "p_corrected": p_adj,
                     "significant": bool(p_adj < alpha),
                     "n": {a: ns.get(a), b: ns.get(b)}, "warning": ""})
    return rows


def compare_all(df, metrics, cfg):
    out = [compare_metric(df, m, cfg) for m in metrics if m in df.columns]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def significant_pairs(df, metric, gcol, cfg):
    """Parejas significativas (p corregido) para dibujar barras en las figuras."""
    st = cfg.get("statistics", {})
    paired = st.get("paired", False); alpha = st.get("alpha", 0.05)
    correction = st.get("correction", "holm")
    metodo = st.get("metodo", "auto"); alpha_sup = st.get("alpha_supuestos", 0.05)
    levels = sorted(df[gcol].dropna().unique())
    data = {g: df.loc[df[gcol] == g, metric].dropna().values for g in levels}
    pairs = list(combinations(levels, 2))
    raw = []
    for a, b in pairs:
        if len(data[a]) < 2 or len(data[b]) < 2:
            raw.append(np.nan); continue
        raw.append(_pair_test(data[a], data[b], paired, metodo, alpha_sup)[1])
    ok = [i for i, v in enumerate(raw) if v == v]
    if not ok:
        return []
    corr = list(raw)
    if correction != "none" and len(ok) > 1:
        _, pc, _, _ = multipletests([raw[i] for i in ok], alpha=alpha, method=correction)
        for i, v in zip(ok, pc):
            corr[i] = v
    return [(a, b, float(pc)) for (a, b), pc in zip(pairs, corr)
            if pc == pc and pc < alpha]


# ---------------------------------------------------------------------------
# FACTORIAL: efectos principales + INTERACCIONES
# ---------------------------------------------------------------------------
def _art_transform(df, metric, factors):
    """
    Aligned Rank Transform (ART): permite un factorial NO paramétrico con
    interacciones. Para cada efecto se "alinea" la respuesta quitando los demás
    efectos (medias de celda) y luego se rankea; el ANOVA sobre esos rangos
    prueba ese efecto. Devuelve {efecto: serie_alineada_rankeada}.
    Ref.: Wobbrock et al. 2011 (ARTool).
    """
    out = {}
    y = df[metric].astype(float)
    grand = y.mean()
    cell = df.groupby(factors, observed=True)[metric].transform("mean")
    # efectos principales
    for f in factors:
        main = df.groupby(f, observed=True)[metric].transform("mean")
        others = cell - main            # parte explicada por el resto (aprox.)
        aligned = y - (others - grand + grand) if len(factors) > 1 else y
        out[f] = sps.rankdata(aligned)
    # interacción de todos los factores (2 vías o más)
    if len(factors) >= 2:
        mains = sum(df.groupby(f, observed=True)[metric].transform("mean") for f in factors)
        aligned_int = y - mains + (len(factors) - 1) * grand
        out[":".join(factors)] = sps.rankdata(aligned_int)
    return out


def factorial_analysis(df, metric, factors, cfg):
    """
    ANOVA de N vías con efectos principales + interacciones.

    metodo (config.statistics.metodo):
      - "parametrico" o "auto" con residuos normales → ANOVA clásico (statsmodels OLS)
      - si no → ART (Aligned Rank Transform): ANOVA sobre rangos alineados.

    Devuelve DataFrame con: efecto, F, p, tipo_prueba, y aviso si n es bajo.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    st = cfg.get("statistics", {})
    metodo = st.get("metodo", "auto"); alpha = st.get("alpha", 0.05)
    alpha_sup = st.get("alpha_supuestos", 0.05)

    d = df[[metric] + factors].dropna().copy()
    for f in factors:
        d[f] = d[f].astype(str)
    if d.empty or any(d[f].nunique() < 2 for f in factors):
        return pd.DataFrame()

    formula = f"Q('{metric}') ~ " + " * ".join([f"C(Q('{f}'))" for f in factors])
    rows = []
    try:
        model = ols(formula, data=d).fit()
        resid_p = float(sps.shapiro(model.resid).pvalue) if len(model.resid) >= 3 else np.nan
        normal_resid = (resid_p != resid_p) or (resid_p > alpha_sup)
        usar_par = (metodo == "parametrico") or (metodo == "auto" and normal_resid)
        if usar_par:
            aov = sm.stats.anova_lm(model, typ=2)
            for efecto, r in aov.iterrows():
                if efecto == "Residual":
                    continue
                nombre = (efecto.replace("C(Q('", "").replace("'))", "")
                          .replace(")", "").replace(":", " × "))
                rows.append({"metric": metric, "efecto": nombre,
                             "prueba": "ANOVA (paramétrico)",
                             "F": float(r["F"]), "p_value": float(r["PR(>F)"]),
                             "significant": bool(r["PR(>F)"] < alpha),
                             "shapiro_residuos_p": round(resid_p, 4) if resid_p == resid_p else None,
                             "n": int(len(d))})
            return pd.DataFrame(rows)
    except Exception:
        pass

    # --- ART (no paramétrico) ---
    try:
        arts = _art_transform(d, metric, factors)
        for efecto, ranks in arts.items():
            dd = d.copy(); dd["_r"] = ranks
            f_eff = efecto.split(":")
            fml = "_r ~ " + " * ".join([f"C(Q('{f}'))" for f in f_eff])
            m2 = ols(fml, data=dd).fit()
            aov2 = sm.stats.anova_lm(m2, typ=2)
            key = [i for i in aov2.index if i != "Residual"][-1]
            rows.append({"metric": metric, "efecto": efecto.replace(":", " × "),
                         "prueba": "ART (no paramétrico)",
                         "F": float(aov2.loc[key, "F"]),
                         "p_value": float(aov2.loc[key, "PR(>F)"]),
                         "significant": bool(aov2.loc[key, "PR(>F)"] < alpha),
                         "shapiro_residuos_p": None, "n": int(len(d))})
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def factorial_posthoc(df, metric, factors, cfg):
    """
    POST-HOC del factorial: compara todas las CELDAS del cruce (p. ej.
    control·hembra vs obeso·macho) con corrección múltiple.

    Se usa sobre todo cuando la INTERACCIÓN sale significativa: el factorial dice
    que el efecto de un factor depende del otro, y este post-hoc dice entre qué
    combinaciones concretas está la diferencia.

    posthoc = "tukey" → Tukey HSD sobre las celdas; si no, pruebas por pares con
    la misma lógica adaptativa + corrección (holm por defecto).
    """
    st = cfg.get("statistics", {})
    alpha = st.get("alpha", 0.05); correction = st.get("correction", "holm")
    metodo = st.get("metodo", "auto"); alpha_sup = st.get("alpha_supuestos", 0.05)
    posthoc = st.get("posthoc", "auto")

    d = df[[metric] + factors].dropna().copy()
    if d.empty:
        return pd.DataFrame()
    d["_celda"] = d[factors].astype(str).agg("·".join, axis=1)
    levels = sorted(d["_celda"].unique())
    if len(levels) < 2:
        return pd.DataFrame()
    data = {g: d.loc[d["_celda"] == g, metric].values for g in levels}
    ns = {g: len(v) for g, v in data.items()}

    if posthoc == "tukey":
        rows = _tukey_rows(data, levels, metric, alpha, ns)
        for r in rows:
            r["test"] = "post-hoc celdas: Tukey HSD"
        return pd.DataFrame(rows)

    pairs = list(combinations(levels, 2)); names, raw = [], []
    for a, b in pairs:
        if len(data[a]) < 2 or len(data[b]) < 2:
            names.append("n/a"); raw.append(np.nan); continue
        nm, pv = _pair_test(data[a], data[b], False, metodo, alpha_sup)
        names.append(nm); raw.append(pv)
    ok = [i for i, v in enumerate(raw) if v == v]
    corr = list(raw)
    if correction != "none" and len(ok) > 1:
        _, pc, _, _ = multipletests([raw[i] for i in ok], alpha=alpha, method=correction)
        for i, v in zip(ok, pc):
            corr[i] = v
    return pd.DataFrame([
        {"metric": metric, "test": f"post-hoc celdas: {nm}", "comparison": f"{a} vs {b}",
         "p_value": pv, "p_corrected": pc,
         "significant": bool(pc == pc and pc < alpha), "n": {a: ns[a], b: ns[b]}}
        for (a, b), nm, pv, pc in zip(pairs, names, raw, corr)])


def factorial_all(df, metrics, factors, cfg):
    """Aplica factorial_analysis a varias métricas y concatena."""
    out = [factorial_analysis(df, m, factors, cfg) for m in metrics if m in df.columns]
    out = [o for o in out if len(o) and not o.isna().all().all()]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def factorial_posthoc_all(df, metrics, factors, cfg, solo_si_interaccion=True,
                          factorial_df=None):
    """
    Post-hoc de celdas para varias métricas. Si `solo_si_interaccion` es True
    (recomendado), solo lo hace en las métricas cuya INTERACCIÓN salió
    significativa en `factorial_df` — evita cientos de comparaciones inútiles.
    """
    metrics_use = metrics
    if solo_si_interaccion and factorial_df is not None and len(factorial_df):
        inter = factorial_df[factorial_df["efecto"].str.contains("×", na=False)
                             & factorial_df["significant"]]
        metrics_use = sorted(inter["metric"].unique())
    out = [factorial_posthoc(df, m, factors, cfg) for m in metrics_use if m in df.columns]
    out = [o for o in out if len(o) and not o.isna().all().all()]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
