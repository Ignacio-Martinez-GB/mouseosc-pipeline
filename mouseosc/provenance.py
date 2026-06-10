"""
===============================================================================
PROCEDENCIA (provenance) — reproducibilidad
===============================================================================

PROPÓSITO
---------
Cada salida del pipeline debe poder responder: "¿con qué se generó esto?".
Este módulo construye una CABECERA DE PROCEDENCIA con:
  - fecha/hora de la corrida,
  - hash SHA-256 del config.yaml (si cambias un parámetro, cambia el hash),
  - versión del paquete mouseosc,
  - versiones de las librerías científicas clave.

Si dentro de un año regeneras una figura y el hash no coincide, sabes que algún
parámetro cambió. Es la diferencia entre "creo que usé ventana de 2 s" y saberlo.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone


def config_hash(cfg: dict) -> str:
    """SHA-256 corto del config (serializado de forma estable: claves ordenadas)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _versions() -> dict:
    """Versiones de las librerías que afectan los números del análisis."""
    vers = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "pandas", "statsmodels", "specparam"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = "no-instalado"
    return vers


def provenance(cfg: dict) -> dict:
    """Diccionario de procedencia completo para incrustar en salidas."""
    from . import __version__
    return {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mouseosc_version": __version__,
        "config_hash": config_hash(cfg),
        "config_name": cfg.get("project", {}).get("name", "?"),
        "seed": cfg.get("project", {}).get("seed"),
        "librerias": _versions(),
    }


def header_text(cfg: dict) -> str:
    """Cabecera de una línea para poner arriba de CSVs/figuras."""
    p = provenance(cfg)
    return (f"# mouseosc {p['mouseosc_version']} | {p['generado']} | "
            f"config_hash={p['config_hash']} | proyecto={p['config_name']}")
