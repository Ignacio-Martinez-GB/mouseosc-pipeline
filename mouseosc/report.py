"""
===============================================================================
REPORTE DE SALUD
===============================================================================

Genera un reporte HTML por corrida: cuántos registros entraron/salieron y por
qué, los checks de cada uno con su semáforo, y un veredicto global. Antes de
confiar en una figura, miras este reporte.

También guarda los checks como CSV (para filtrar/auditar) y la procedencia.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from .provenance import provenance

_COLOR = {"ok": "#2e7d32", "warn": "#f9a825", "error": "#c62828"}
_ICON = {"ok": "●", "warn": "▲", "error": "✖"}


def write_reports(checks_records, cfg, out_dir):
    """
    checks_records: lista de dicts {rec_id, group, verdict, checks:[Check...]}
    Escribe report.html y checks.csv en out_dir. Devuelve el path del HTML.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance(cfg)

    # --- CSV plano de todos los checks ---
    rows = []
    for r in checks_records:
        for c in r["checks"]:
            d = c.as_dict()
            d.update({"rec_id": r["rec_id"], "group": r["group"]})
            rows.append(d)
    pd.DataFrame(rows).to_csv(out_dir / "checks.csv", index=False)

    # --- conteos para el resumen ---
    n_total = len(checks_records)
    n_err = sum(r["verdict"] == "error" for r in checks_records)
    n_warn = sum(r["verdict"] == "warn" for r in checks_records)
    n_ok = n_total - n_err - n_warn

    html = [f"""<!doctype html><meta charset="utf-8">
<title>Reporte de salud — {prov['config_name']}</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:2rem;color:#222}}
 h1{{margin-bottom:.2rem}} .prov{{color:#666;font-size:.85rem;margin-bottom:1.5rem}}
 .summary span{{display:inline-block;padding:.4rem .8rem;border-radius:.4rem;
   margin-right:.5rem;color:#fff;font-weight:600}}
 table{{border-collapse:collapse;width:100%;margin-top:1rem;font-size:.9rem}}
 th,td{{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}}
 th{{background:#f5f5f5}} .rec{{font-weight:600;background:#fafafa}}
</style>
<h1>Reporte de salud — {prov['config_name']}</h1>
<div class="prov">mouseosc {prov['mouseosc_version']} · {prov['generado']} ·
 config_hash <code>{prov['config_hash']}</code> · semilla {prov['seed']}</div>
<div class="summary">
 <span style="background:{_COLOR['ok']}">{n_ok} OK</span>
 <span style="background:{_COLOR['warn']}">{n_warn} con avisos</span>
 <span style="background:{_COLOR['error']}">{n_err} con error</span>
 <span style="background:#555">{n_total} registros</span>
</div>
<table><tr><th>Registro</th><th>Grupo</th><th>Etapa</th><th>Check</th>
<th>Nivel</th><th>Detalle</th></tr>"""]

    for r in checks_records:
        v = r["verdict"]
        html.append(f'<tr class="rec"><td colspan="4">{r["rec_id"]}</td>'
                    f'<td style="color:{_COLOR[v]}">{_ICON[v]} {v.upper()}</td>'
                    f'<td>{r["group"]}</td></tr>')
        for c in r["checks"]:
            html.append(
                f'<tr><td></td><td></td><td>{c.stage}</td><td>{c.name}</td>'
                f'<td style="color:{_COLOR[c.level]}">{_ICON[c.level]} {c.level}</td>'
                f'<td>{c.message}</td></tr>')
    html.append("</table>")

    html_path = out_dir / "report.html"
    html_path.write_text("\n".join(html), encoding="utf-8")
    return html_path
