# Atajos del pipeline. Uso:  make <objetivo>
.PHONY: setup test demo lock clean

setup:        ## crea entorno e instala todo (Mac/Linux)
	bash setup.sh

test:         ## corre los tests de verdad conocida
	pytest -q

demo:         ## genera datos sintéticos, corre el pipeline y deja el reporte
	python run.py demo

lock:         ## regenera requirements.lock desde un entorno limpio
	python -m venv /tmp/_lockenv && /tmp/_lockenv/bin/pip install -e . && \
	/tmp/_lockenv/bin/pip freeze --exclude-editable > requirements.lock

clean:        ## borra salidas y cachés (no toca datos ni código)
	rm -rf resultados resultados_demo resultados_test _config_demo.yaml \
	       examples/datos_sinteticos .pytest_cache **/__pycache__
