# PRUEBA COMPLETA 1 — instrucciones para Claude Code

Correr el pipeline **mouseosc** completo sobre CONTROL vs Obesos (proyecto
"Analisis Ale"): 3 análisis de ruido, banda 0.4–160 Hz, specparam + PAC + bursts,
estadística adaptativa + factorial, y comparaciones por dieta y sexo.

> **Todo se configura en `config.yaml`** — es el único archivo de parámetros del
> proyecto. Ya viene con los valores de esta prueba puestos.

## 0. Dónde está todo

- **Proyecto (código):** `~/Code/mouseosc-pipeline`
- **Configuración:** `~/Code/mouseosc-pipeline/config.yaml`
- **Datos:**
  `~/Library/Mobile Documents/com~apple~CloudDocs/Antigravity/AnalisisAleSteph/Analisis Ale/Datos`
  (contiene `CONTROL/`, `Obesos/` y `Ruido/`; la carpeta `Ruido/` se **excluye
  automáticamente** del análisis y solo se usa como referencia de ruido.)
- **Referencia de ruido:** `ruido_referencia.csv` (ya incluida en el repo).

> Requisito: **Python 3.10–3.12** (no 3.13/3.14).

## 1. Preparar el entorno (una vez)

```bash
cd ~/Code/mouseosc-pipeline
bash setup.sh                 # crea .venv, instala y corre los tests (deben pasar)
source .venv/bin/activate
pip install -e ".[specparam]" # necesario para el ajuste 1/f
```

## 2. Ajustar las 2 rutas de datos en `config.yaml`

Abre `config.yaml` y pon las rutas absolutas de TU máquina:

```yaml
dataset:
  root: "/Users/<TU_USUARIO>/Library/Mobile Documents/com~apple~CloudDocs/Antigravity/AnalisisAleSteph/Analisis Ale/Datos"
noise:
  ruido:
    dir: "/Users/<TU_USUARIO>/Library/Mobile Documents/com~apple~CloudDocs/Antigravity/AnalisisAleSteph/Analisis Ale/Datos/Ruido"
```

El resto ya está configurado (banda 0.4–160, ruido con los 3 análisis, estadística
adaptativa, factorial dieta × sexo, comparaciones dieta/sexo).

## 3. Generar el manifiesto

```bash
python run.py scan-folder "$(python -c "import yaml;print(yaml.safe_load(open('config.yaml'))['dataset']['root'])")" \
  --out manifest.csv --config config.yaml
```

Debe reportar ~167 archivos con columnas `dieta, condicion, sexo` (grupos
`control` y `obeso`) y **excluir** la carpeta `Ruido`.

## 4. Validar

```bash
python run.py validate --config config.yaml
```

Abre `resultados/report.html`. Si la mayoría está en verde, continúa.

## 5. Correr el análisis completo

```bash
python run.py run --config config.yaml
```

Tarda **varios minutos** (specparam + PAC con 200 subrogados × 167 archivos × 3
análisis). Es normal — déjalo terminar.

## 6. Qué debe generarse (verificación)

En `resultados/`:

- `report.html` — salud por registro.
- `deteccion_ruido/contaminacion.csv` — qué archivos se marcaron contaminados a 10 Hz.
- `analisis_1_normal/`, `analisis_2_sin_ruido/`, `analisis_3_corregido/`, cada uno con:
  - `descriptivo/` y `comparaciones/` (dieta, dieta_por_sexo, sexo, sexo_por_dieta)
  - dentro: `espectro/`, `bandas/` (+ `prism/`), `specparam/`, `pac/`, `bursts/`,
    `estadistica/stats_comparisons.csv`
  - `factorial/efectos_e_interacciones.csv` (+ `posthoc_celdas.csv` si hay interacción)

Comprobaciones clave:
- El PSD llega hasta **160 Hz** y **no existe la banda `mua`**.
- En `analisis_3_corregido/` los picos de 10 Hz están **aplanados** vs `analisis_1_normal/`.
- En `stats_comparisons.csv` aparece la prueba usada y los p de Shapiro/Levene.
- Colores: control azul, obeso naranja; bandas ordenadas por frecuencia.

## Notas

- Para una corrida más rápida: en `config.yaml` pon `pac.enabled: false` (lo más
  lento) o `noise.analyses: [1, 3]`.
- Método de corrección del Análisis 3: `noise.metodo_correccion` =
  `interpolacion` (default) | `escalada` | `literal`.
- `manifest.csv` y `resultados/` no se suben a git (están ignorados).
