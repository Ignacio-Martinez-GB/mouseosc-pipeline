#!/usr/bin/env bash
# =============================================================================
# setup.sh — arranque de UN comando (Mac / Linux)
# =============================================================================
# Crea un entorno virtual, instala el paquete y corre los tests para confirmar
# que todo funciona. Uso:   bash setup.sh
# =============================================================================
set -e
cd "$(dirname "$0")"

echo "1/4  Creando entorno virtual (.venv)..."
python3 -m venv .venv

echo "2/4  Activando e instalando dependencias..."
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip

if [ -f requirements.lock ]; then
  echo "     Usando requirements.lock (versiones exactas)."
  pip install --quiet -r requirements.lock
fi
pip install --quiet -e .

echo "3/4  Corriendo tests de verdad conocida..."
pip install --quiet pytest
pytest -q

echo "4/4  Listo. Activa el entorno con:  source .venv/bin/activate"
echo "     Prueba el pipeline con:        python run.py demo"
