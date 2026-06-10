"""
===============================================================================
ESTADÍSTICA DE GRUPOS
===============================================================================

Compara una métrica entre grupos. Elige automáticamente la prueba según el
diseño (config -> statistics):

  2 grupos, no pareado : Mann-Whitney U (no asume normalidad)
  2 grupos, pareado    : Wilcoxon signed-rank
  ≥3 grupos, no pareado: Kruskal-Wallis (omnibus)
  ≥3 grupos, pareado   : Friedman (omnibus)

Se usan pruebas NO PARAMÉTRICAS por defecto porque con n pequeño (típico en
ratón) no se puede asumir normalidad con confianza. Para ≥3 grupos, tras el
omnibus se hacen comparaciones por pares con corrección múltiple (holm por
defecto), de modo que el p reportado ya está corregido.

Devuelve un DataFrame ordenado, listo para revisar o exportar a GraphPad/Prisma.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests


def _omnibus(groups_data, paired):
    """Prueba global: ¿difiere ALGÚN grupo? Devuelve (nombre, stat, p)."""
    k = len(groups_data)
    if k == 2:
        a, b = groups_data
        if paired:
            stat, p = sps.wilcoxon(a, b)
            return "Wilcoxon", stat, p
        stat, p = sps.mannwhitneyu(a, b, alternative="two-sided")
        return "Mann-Whitney U", stat, p
    if paired:
        stat, p = sps.friedmanchisquare(*groups_data)
        return "Friedman", stat, p
    stat, p = sps.kruskal(*groups_data)
    return "Kruskal-Wallis", stat, p


def compare_metric(df, metric, cfg):
    """
    Compara `metric` entre los niveles de la columna de grupo.
    Devuelve un DataFrame con el omnibus y, si aplica, los pares corregidos.
    """
    st = cfg.get("statistics", {})
    gcol = st.get("group_col", "group")
    paired = st.get("paired", False)
    alpha = st.get("alpha", 0.05)
    correction = st.get("correction", "holm")
    min_n = st.get("min_n_per_group", 3)

    levels = sorted(df[gcol].dropna().unique())
    data = {g: df.loc[df[gcol] == g, metric].dropna().values for g in levels}
    ns = {g: len(v) for g, v in data.items()}

    rows = []
    name, stat, p = _omnibus(list(data.values()), paired)
    rows.append({"metric": metric, "test": name, "comparison": "omnibus",
                 "statistic": float(stat), "p_value": float(p),
                 "p_corrected": float(p), "significant": bool(p < alpha),
                 "n": dict(ns), "warning": "" if min(ns.values()) >= min_n
                 else f"n<{min_n} en algún grupo"})

    # Comparaciones por pares solo si hay ≥3 grupos (con 2, el omnibus ya es el par).
    if len(levels) >= 3:
        pairs = list(combinations(levels, 2))
        raw_p = []
        for a, b in pairs:
            if paired:
                s, pv = sps.wilcoxon(data[a], data[b])
            else:
                s, pv = sps.mannwhitneyu(data[a], data[b], alternative="two-sided")
            raw_p.append(pv)
        if correction != "none" and raw_p:
            _, p_corr, _, _ = multipletests(raw_p, alpha=alpha, method=correction)
        else:
            p_corr = raw_p
        for (a, b), pv, pc in zip(pairs, raw_p, p_corr):
            rows.append({"metric": metric, "test": "pairwise", "comparison": f"{a} vs {b}",
                         "statistic": np.nan, "p_value": float(pv), "p_corrected": float(pc),
                         "significant": bool(pc < alpha), "n": {a: ns[a], b: ns[b]},
                         "warning": ""})
    return pd.DataFrame(rows)


def compare_all(df, metrics, cfg):
    """Aplica compare_metric a una lista de métricas y concatena."""
    out = [compare_metric(df, m, cfg) for m in metrics if m in df.columns]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
