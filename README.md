# mouseosc — pipeline de actividad oscilatoria en ratón

Pipeline **reproducible, configurable y verificable** para analizar oscilaciones
(LFP/ERG) de ratón: PSD de Welch, separación 1/f (specparam), métricas por banda,
acoplamiento fase-amplitud (PAC) y estadística de grupos. Todo se controla desde
un único archivo `config.yaml` y un `manifest.csv`, sin tocar el código.

---

## 🚀 Inicio rápido (5 pasos)

Si es tu primera vez, sigue esto en orden. No necesitas saber programar.

**1. Instala Python 3.12** (no uses 3.13/3.14, ver más abajo). Compruébalo:
   `python3 --version`.

**2. Instala el proyecto** — un solo comando en la terminal, dentro de la carpeta:

```bash
bash setup.sh        # Mac / Linux
setup.bat            # Windows (doble clic)
```

Esto crea el entorno, instala todo y corre las pruebas. Si ves `9 passed`, quedó bien.

**3. Prueba que funciona sin tus datos.** Abre `INICIO.py`, deja `MODO = "demo"`
   y dale al botón ▶ Run (en PyCharm) o corre `python run.py demo` (terminal).
   Mira la carpeta `resultados_demo/`: si hay figuras y un `report.html`, todo OK.

**4. Prepara tus datos.** Pon tus archivos en una carpeta y crea el manifiesto:
   en `INICIO.py` pon `MODO = "scan"` y `CARPETA_DE_DATOS = "ruta/a/tus/datos"`, ▶ Run.
   Se crea `manifest.csv`: ábrelo y rellena la columna `group` (y `sex`, `condition`… si quieres compararlas).
   Mira **`manifest_ejemplo.csv`** para ver el formato esperado (incluye `sex` y `condition`).

**5. Ajusta y corre.** En `config.yaml` revisa `preprocessing.fs` (tu muestreo) y,
   si son `.mat`, `dataset.mat.channel`. Luego en `INICIO.py` pon `MODO = "validate"`
   (revisa la salud) y después `MODO = "run"` (análisis completo). Los resultados
   quedan en `resultados/`.

### Las tres formas de correrlo

| Forma | Cómo | Para quién |
|---|---|---|
| **Sin terminal** | Abrir `INICIO.py`, editar `MODO`, botón ▶ Run | Lo más fácil (PyCharm/VS Code) |
| **Terminal** | `python run.py demo` / `validate` / `run` | Si te sientes cómodo en consola |
| **PyCharm Run Config** | Script `run.py`, parámetros `run --config config.yaml` | Para repetir corridas con un clic |

> Si abres `run.py` y le das a ▶ sin más, no falla: te muestra una guía de qué hacer.

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

## Segmentos de comparación (config)

Defines en `config.yaml` qué comparaciones generar. Cada bloque produce su propia
carpeta con estadística de **dos grupos** para **todas las parejas** de los
niveles de `by`:

```yaml
descriptivo: {enabled: true, by: "group"}      # todos los grupos juntos
comparisons:
  - {name: "grupo",          by: "group", within: null,  enabled: true}
  - {name: "grupo_por_sexo", by: "group", within: "sex", enabled: false}  # estratifica por sexo
  - {name: "sexo",           by: "sex",   within: null,  enabled: false}
```

`by` y `within` son columnas del **manifiesto** (añade `sex`, `condition`, etc.
para poder compararlas). `within` repite la comparación dentro de cada nivel
(p. ej. control vs pups por separado en hembras y en machos).

## Estructura de salidas

```
resultados/
  report.html                 reporte de salud
  metrics_all.csv             maestro global (una fila por registro)
  descriptivo/                ← todos los grupos juntos
    espectro/   psd_por_grupo.png, psd_con_bandas.png, por_banda/psd_<banda>.png (zoom),
                psd_media_sem.csv (datos detrás), psd_por_sujeto.csv
    bandas/     bandpower_abs/rel.png, box_<banda>_<abs|rel>.png (con pie estadístico),
                prism/<metrica>.csv, bandas_largo.csv
    specparam/  pac/  bursts/   (si activos) box_*.png + prism/*.csv
    estadistica/ stats_comparisons.csv ; metrics.csv
  comparaciones/
    <nombre>/<A>_vs_<B>/...           misma estructura, 2 grupos, stats de 2 grupos
    <nombre>/<sex=…>/<A>_vs_<B>/...   si usaste `within`
```

Cada figura usa la **paleta fija por grupo** (`config.plotting.palette`), muestra
el método estadístico al pie y las bandas llevan su rango en Hz. Los CSV de
`prism/` no llevan cabecera (1ª fila = grupos) → pegables en GraphPad Prism.

## Escaneo automático del árbol de datos

El modo **scan** puede rellenar el manifiesto solo, leyendo las **palabras** de
las carpetas (sin importar el nivel ni el orden). Defines una vez el diccionario
en `config.yaml → dataset.scan.factores`:

```yaml
dataset:
  scan:
    group_from: dieta
    factores:
      dieta:     {control: [CONTROL, Control], obeso: [PUPS, OBESO, Obesos]}
      condicion: {foto: [FOTO, Foto], meso: [MESO, Meso]}
      sexo:      {hembra: [HEMBRAS, Hembras], macho: [MACHOS, Machos]}
```

Al escanear, cada archivo se clasifica por las palabras de su ruta y el
`manifest.csv` sale con una columna por factor ya rellena. El escáner además
reporta el **conteo por combinación** y los **segmentos no reconocidos**
(posibles typos o factores que faltan en el diccionario). Sirve para árboles con
profundidades y órdenes distintos: solo cambias el diccionario, no el código.

## Rango global de análisis

`analysis_band: [0.4, 160]` en el config **recorta todo el proyecto** a esa
ventana de frecuencia (PSD, bandas, specparam, potencia relativa, PAC y detección
de ruido) — en los 3 tipos de análisis y todas las comparaciones. Las bandas se
acotan **solo en los extremos**: una banda interior no cambia, una que cruza el
límite se recorta, y una totalmente fuera (p. ej. `mua` 300–500 Hz) se elimina.
Pon `null` para no limitar. El valor recomendado sale del análisis de
separabilidad señal/ruido (la señal se separa del ruido hasta ~160–170 Hz).

## Ruido eléctrico (esquema de 3 análisis)

El ruido de **60 Hz** (línea eléctrica) se suprime **siempre** con un notch
(`preprocessing.notch_default`). Para el ruido de **10 Hz + armónicos** hay un
esquema aparte que activas con `noise.enabled: true` y eliges qué análisis correr
en `noise.analyses`:

- **[1] normal** — todos los archivos tal cual.
- **[2] sin ruido** — se excluyen los archivos detectados como contaminados.
- **[3] corregido** — a los contaminados se les **resta el espectro promedio del
  ruido** (de `noise.ruido.dir`) en los armónicos, y se les aplica un **notch** en
  10 Hz para PAC/bursts; los no contaminados quedan intactos.

Cada análisis sale en su carpeta: `resultados/analisis_1_normal/`,
`analisis_2_sin_ruido/`, `analisis_3_corregido/`, más `deteccion_ruido/` con la
tabla de qué archivo se marcó contaminado y por qué (SNR y persistencia por
armónico). La detección usa prominencia del pico + persistencia temporal (el
ruido de línea es un pico angosto, en peine y estacionario).

La referencia de ruido se toma del CSV `noise.ruido.reference_csv` si existe
(pre-calculado, versionable), o se computa promediando los archivos de
`noise.ruido.dir`.

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
