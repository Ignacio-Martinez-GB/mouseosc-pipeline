# mouseosc — pipeline genérico de actividad oscilatoria en ratón

Pipeline **reproducible, configurable y verificable** para analizar oscilaciones
(LFP/ERG) de ratón: PSD de Welch, separación 1/f (specparam), métricas por banda,
acoplamiento fase-amplitud (PAC) y estadística de grupos. Pensado para
**compartirse y reutilizarse con distintos sets de datos** sin tocar el código:
todo se controla desde `config.yaml`.

## 👉 ¿Primera vez? Empieza por `INICIO.py`

Si no quieres usar la terminal: abre **`INICIO.py`** en PyCharm, edita las dos
variables de arriba (`MODO` y `CONFIG`) y dale al botón ▶ Run. Empieza con
`MODO = "demo"` para ver que todo funciona. Ese archivo explica el resto.
(Si abres `run.py` y lo corres sin más, te mostrará esta misma guía.)

## Principios de diseño

1. **Config-driven, cero números mágicos.** Cada parámetro vive en `config.yaml`,
   comentado con qué es, qué pasa si lo subes/bajas y un valor típico para ratón.
2. **Genérico por manifiesto.** El pipeline no adivina tu estructura de carpetas:
   lee un `manifest.csv` (una fila por registro). Sirve para cualquier diseño.
   Formatos: `.mat`, `.csv/.tsv`, `.abf` (Axon), `.nwb`.
3. **Verificación de primera clase.** Cada etapa emite *checks* con semáforo
   (verde/ámbar/rojo). Se genera un **reporte de salud** HTML por corrida y, si
   quieres, el pipeline se detiene ante un check rojo.
4. **Reproducibilidad.** Versiones fijadas (`pyproject.toml`), semilla fija, y
   cada salida lleva una cabecera con fecha + hash del config + versiones.

## Requisito de Python

Usa **Python 3.10, 3.11 o 3.12**. Con 3.13/3.14 la instalación de numpy/scipy
puede fallar (no hay paquetes precompilados de las versiones fijadas e intenta
compilar desde fuente, lo que requiere un compilador Fortran). Comprueba tu
versión con `python3 --version`; si es muy nueva, crea el entorno con 3.12.

## Instalación

Arranque de **un comando** (crea el entorno, instala y corre los tests):

```bash
bash setup.sh        # Mac / Linux
setup.bat            # Windows (doble clic o en cmd)
```

O manual:

```bash
cd PipelineRatonGenerico
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock   # versiones EXACTAS (reproducibilidad)
pip install -e .                    # núcleo
pip install -e ".[specparam]"       # + separación 1/f (opcional)
pip install -e ".[abf,nwb,test]"    # + lectores extra y pytest (opcional)
```

Atajos con `make`: `make setup`, `make test`, `make demo`, `make lock`, `make clean`.

## ¿Es fácil de correr en otra computadora?

Sí. Es Python puro y multiplataforma. Para reproducir números **idénticos**,
`requirements.lock` fija todas las versiones (directas y transitivas). Quien
reciba el proyecto corre `bash setup.sh` y en un minuto tiene el entorno y los
tests en verde. Prueba la instalación sin tus datos con `python run.py demo`.

## Uso en 4 pasos

```bash
# 1. Generar una plantilla de manifiesto a partir de tu carpeta de datos
python run.py scan-folder Datos/        # crea manifest.csv (rellena group/animal_id)

# 2. (si es .mat) ver qué variable tiene la señal
python run.py inspect Datos/raton01.mat

# 3. Validar la salud de los datos SIN producir resultados
python run.py validate                  # → resultados/report.html

# 4. Corrida completa
python run.py run                        # métricas + stats + figuras + reporte
```

## ¿Qué parámetro toco para...?

| Quiero... | En `config.yaml` |
|---|---|
| cambiar la frecuencia de muestreo | `preprocessing.fs` |
| ser más/menos estricto con artefactos | `preprocessing.artifact_threshold_sd` |
| más resolución en frecuencias bajas | activar ventana de 10 s en `spectral.windows` |
| redefinir las bandas | `bands` |
| elegir qué comparar entre grupos | `statistics.group_col`, `groups.expected` |
| que NO se detenga ante un error | `checks.stop_on_error: false` |

## Verificación (checkpoints)

- **Tests de verdad conocida** (`tests/`): un tono de 40 Hz debe dar el pico del
  PSD en 40 Hz; una señal con PAC sintético debe dar MI alto y ~0 en ruido; la
  estadística debe detectar una diferencia real y no inventarla bajo el nulo.
  Corre `pytest -v`.
- **Checks por registro** (`mouseosc/checks.py`): fs coherente con los datos,
  sin NaN/saturación, % de épocas rechazadas, R² del ajuste 1/f, y
  **conservación de energía** (las bandas suman ≈ la potencia total).
- **Reporte de salud** (`resultados/report.html`): semáforo por registro y
  veredicto global antes de confiar en ninguna figura.

## Estructura

```
PipelineRatonGenerico/
├── config.yaml            ← único punto de control (todo comentado)
├── pyproject.toml         ← entorno con versiones fijadas
├── run.py                 ← CLI: scan-folder | inspect | validate | run
├── mouseosc/              ← paquete
│   ├── io.py              carga genérica + manifiesto
│   ├── preprocessing.py   detrend, filtros fase-cero, épocas, artefactos
│   ├── spectral.py        Welch + specparam
│   ├── bands.py           métricas por banda + globales
│   ├── pac.py             MI (Tort) + MVL (Canolty) + comodulograma
│   ├── stats.py           comparaciones de grupos + corrección múltiple
│   ├── checks.py          capa de verificación
│   ├── report.py          reporte de salud HTML
│   ├── provenance.py      hash de config + versiones
│   └── viz.py             figuras
├── tests/test_synthetic.py
└── examples/make_synthetic_data.py   ← dataset de demostración
```

## Demo reproducible

```bash
python examples/make_synthetic_data.py   # 2 grupos con diferencia conocida en gamma
# apunta dataset.root/manifest al dataset sintético y:
python run.py run
# → la estadística detecta la diferencia inyectada en gamma_lo (p≈0.002)
```

## Estructura de salidas (una carpeta por análisis)

Tras `run`, en `resultados/` encuentras, por cada tipo de análisis, su figura,
los **datos detrás** de la figura, y CSVs en **formato GraphPad Prism**
(columnas = grupos, filas = valor por animal, pegables directo):

```
resultados/
  report.html                 reporte de salud
  metrics_all.csv             maestro: una fila por registro, todas las métricas
  espectro/
    psd_por_grupo.png
    psd_grupo_media_sem.csv    datos detrás de la figura (freq, media, sem por grupo)
    psd_por_sujeto.csv         PSD completo: una columna por registro
  bandas/
    bandpower_abs.png, bandpower_rel.png, box_<banda>_<abs|rel>.png
    bandas_largo.csv           datos completos (formato largo)
    prism/<banda>_<abs|rel|rms>.csv   ← pegables en Prism
  specparam/   (si specparam activo)  box_*.png + prism/*.csv
  pac/         (si PAC activo)        box_pac_*.png + prism/*.csv [+ comodulograma]
  bursts/      (si bursts activo)     box_burst_*.png + prism/*.csv
  estadistica/
    stats_comparisons.csv      omnibus + pares con corrección múltiple
```

Los CSV de `prism/` no llevan cabecera: su primera fila son los nombres de grupo,
así que se copian y pegan tal cual en una tabla "Column" de GraphPad Prism.

## Recetas

**Correr con archivos `.mat`** (señal en una variable):
```yaml
# config.yaml
dataset: {format: mat, root: "MisDatos", manifest: "manifest.csv"}
dataset: {mat: {channel: "someData2"}}   # python run.py inspect archivo.mat para ver variables
preprocessing: {fs: 2000}
```

**Correr con archivos `.abf` (Axon):**
```bash
pip install -e ".[abf]"
# config: dataset.format=abf, dataset.abf.channel=0; fs se lee del propio archivo
python run.py validate && python run.py run
```

**Añadir un grupo nuevo:** edita `manifest.csv` (columna `group`) y declara el
grupo en `config.yaml -> groups.expected`. Con 3+ grupos, la estadística pasa de
una sola prueba a omnibus + comparaciones por pares corregidas, automáticamente.

**Activar bursts y comodulograma** (opt-in, por coste):
```yaml
bursts: {enabled: true}
pac: {comodulogram: {enabled: true}}
```

**Modo exploratorio (no detenerse ante errores):**
```yaml
checks: {stop_on_error: false}   # registra los problemas en el reporte y sigue
```

**Cambiar el muestreo:** solo `preprocessing.fs`. Un check verifica que sea
coherente con los datos antes de analizar.

## Referencias

- Tort AB et al. (2010) *J Neurophysiol* 104:1195. (Modulation Index)
- Canolty RT et al. (2006) *Science* 313:1626. (Mean Vector Length)
- Donoghue T et al. (2020) *Nat Neurosci* 23:1655. (specparam / FOOOF)
