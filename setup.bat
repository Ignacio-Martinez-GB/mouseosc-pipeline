@echo off
REM ===========================================================================
REM setup.bat - arranque de UN comando (Windows)
REM ===========================================================================
REM Crea un entorno virtual, instala el paquete y corre los tests.
REM Uso (doble clic, o en cmd):   setup.bat
REM ===========================================================================
cd /d "%~dp0"

echo 1/4  Creando entorno virtual (.venv)...
python -m venv .venv

echo 2/4  Activando e instalando dependencias...
call .venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
if exist requirements.lock (
  echo      Usando requirements.lock ^(versiones exactas^).
  pip install --quiet -r requirements.lock
)
pip install --quiet -e .

echo 3/4  Corriendo tests de verdad conocida...
pip install --quiet pytest
pytest -q

echo 4/4  Listo. Activa el entorno con:  .venv\Scripts\activate
echo      Prueba el pipeline con:        python run.py demo
pause
