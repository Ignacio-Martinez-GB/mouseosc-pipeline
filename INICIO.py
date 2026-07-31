"""
===============================================================================
👉 EMPIEZA AQUÍ — cómo correr el pipeline sin usar la terminal
===============================================================================

Si abriste este proyecto en PyCharm y no sabes por dónde empezar, este es el
archivo correcto. Funciona así:

  1. Edita SOLO la sección "CONFIGURACIÓN" de abajo (entre las líneas =====).
  2. Dale al botón ▶ (Run) arriba a la derecha, con este archivo abierto.
  3. Mira la consola: te dirá qué hizo y dónde quedaron los resultados.

No necesitas escribir comandos. Todo lo que un usuario normal cambia está aquí.
Los detalles finos (bandas, filtros, etc.) viven en config.yaml, también
comentados uno por uno.
"""

# =============================================================================
# CONFIGURACIÓN  —  edita estas líneas y nada más
# =============================================================================

# ¿Qué quieres hacer? Pon UNA de estas opciones entre comillas:
#   "demo"     → genera datos de prueba y corre todo (para ver que funciona).
#   "scan"     → crea la plantilla de manifiesto a partir de tu carpeta de datos.
#   "validate" → revisa la SALUD de tus datos sin producir resultados.
#   "run"      → análisis COMPLETO (métricas + estadística + figuras).
MODO = "demo"

# Archivo de configuración a usar (con bandas, fs, grupos, etc.).
CONFIG = "config.yaml"

# Solo para MODO = "scan": carpeta donde están tus archivos de datos.
CARPETA_DE_DATOS = "Datos"

# =============================================================================
# A partir de aquí NO necesitas tocar nada.
# =============================================================================

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
import run as _run


class _Args:
    """Imita los argumentos de la terminal para reutilizar la lógica de run.py."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    print("=" * 70)
    print(f"  Modo: {MODO.upper()}   |   Config: {CONFIG}")
    print("=" * 70)

    if MODO == "demo":
        return _run.cmd_demo(_Args())

    if MODO == "scan":
        print(f"Escaneando '{CARPETA_DE_DATOS}' → manifest.csv ...")
        _run.cmd_scan_folder(_Args(folder=CARPETA_DE_DATOS, out="manifest.csv",
                                   config=str(PROJ / CONFIG)))
        print("\nSiguiente paso: revisa manifest.csv (las columnas de factores ya")
        print("vienen rellenas si definiste dataset.scan.factores en el config).")
        print("Luego cambia arriba MODO = \"validate\" y vuelve a darle a ▶.")
        return 0

    if MODO in ("validate", "run"):
        if not (PROJ / CONFIG).exists():
            print(f"⚠ No encuentro {CONFIG}. Revisa el nombre en la sección CONFIGURACIÓN.")
            return 1
        return _run.cmd_run(_Args(config=str(PROJ / CONFIG)),
                            validate_only=(MODO == "validate"))

    print(f"⚠ MODO = \"{MODO}\" no es válido. Usa: demo, scan, validate o run.")
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
