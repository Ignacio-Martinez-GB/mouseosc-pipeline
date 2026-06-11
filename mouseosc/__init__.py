"""
mouseosc — pipeline reproducible de actividad oscilatoria en ratón.

Submódulos
----------
io            carga genérica (mat/csv/abf/nwb) + manifiesto
preprocessing detrend, filtros fase-cero, épocas, rechazo de artefactos
spectral      PSD de Welch + separación 1/f (specparam)
bands         métricas por banda (potencia abs/rel, RMS, frecuencia mediana...)
pac           acoplamiento fase-amplitud (Tort MI, Canolty MVL) + comodulograma
bursts        detección de ráfagas oscilatorias (envolvente de Hilbert)
stats         comparaciones de grupos con corrección múltiple
checks        capa de verificación: asserts por etapa + diagnósticos
report        reporte de salud por registro (semáforos verde/ámbar/rojo)
provenance    hash de config + versiones (cabecera de procedencia en cada salida)
viz           figuras
"""

__version__ = "0.1.0"
