"""
===============================================================================
IO — carga genérica de señales + manifiesto
===============================================================================

PROPÓSITO
---------
Desacoplar el pipeline de CUALQUIER organización de carpetas. En vez de adivinar
metadatos del path (frágil, lleno de typos), el usuario provee un MANIFIESTO:
un CSV donde cada fila es un registro, con al menos:

    file_path,group,animal_id
    Datos/ctrl_01.mat,control,m01
    Datos/trat_01.mat,tratamiento,m11

Columnas extra (sexo, condición, día...) se conservan y viajan con las métricas.

LECTORES (loaders)
------------------
Un "registro de loaders" mapea formato → función que devuelve (señal_1D, fs).
Soporta .mat, .csv, .abf y .nwb. Los formatos pesados (.abf vía pyabf, .nwb vía
pynwb) se importan SOLO si se usan, así el núcleo no depende de ellos.

Para añadir un formato nuevo: escribe load_xxx(path, opts) -> (np.ndarray, float)
y regístralo en LOADERS. Nada más cambia en el pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Recording:
    """Un registro listo para procesar: metadatos + ruta. La señal se carga
    bajo demanda con load_signal() para no tener todo en memoria a la vez."""
    file_path: Path
    group: str
    animal_id: str
    meta: dict = field(default_factory=dict)   # columnas extra del manifiesto

    @property
    def rec_id(self) -> str:
        return f"{self.group}__{self.animal_id}__{self.file_path.stem}"


# ---------------------------------------------------------------------------
# LECTORES POR FORMATO  (cada uno devuelve: señal 1-D float64, fs en Hz)
# ---------------------------------------------------------------------------

def load_mat(path: Path, opts: dict) -> tuple[np.ndarray, float | None]:
    """Lee la señal de un archivo MATLAB .mat tomando la variable `channel`."""
    import scipy.io
    channel = opts.get("channel", "data")
    mat = scipy.io.loadmat(path)
    if channel not in mat:
        # Ayuda al usuario: lista las variables reales del archivo.
        keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(
            f"Canal '{channel}' no está en {path.name}. "
            f"Variables disponibles: {keys}. Ajusta dataset.mat.channel.")
    sig = np.asarray(mat[channel], dtype=float).squeeze()
    return sig, None  # fs se toma de config (los .mat aquí no lo guardan fiable)


def load_csv(path: Path, opts: dict) -> tuple[np.ndarray, float | None]:
    """Lee señal (y opcionalmente fs) de un CSV/TSV con columnas tiempo,voltaje."""
    df = pd.read_csv(path, delimiter=opts.get("delimiter", ","),
                     skiprows=opts.get("skiprows", 0))
    sig_col = opts.get("signal_col", 1)
    sig = np.asarray(df.iloc[:, sig_col] if isinstance(sig_col, int)
                     else df[sig_col], dtype=float)
    # Si hay columna de tiempo, derivamos fs de ella (más fiable que el config).
    fs = None
    tcol = opts.get("time_col", None)
    if tcol is not None:
        t = np.asarray(df.iloc[:, tcol] if isinstance(tcol, int) else df[tcol],
                       dtype=float)
        dt = np.median(np.diff(t))
        if dt > 0:
            fs = 1.0 / dt
    return sig, fs


def load_abf(path: Path, opts: dict) -> tuple[np.ndarray, float | None]:
    """Lee un archivo Axon .abf (requiere `pip install mouseosc[abf]`)."""
    try:
        import pyabf
    except ImportError as e:
        raise ImportError("Para leer .abf instala el extra: pip install mouseosc[abf]") from e
    abf = pyabf.ABF(str(path))
    abf.setSweep(0, channel=opts.get("channel", 0))
    return np.asarray(abf.sweepY, dtype=float), float(abf.dataRate)


def load_nwb(path: Path, opts: dict) -> tuple[np.ndarray, float | None]:
    """Lee un TimeSeries de un archivo NWB (requiere `pip install mouseosc[nwb]`)."""
    try:
        from pynwb import NWBHDF5IO
    except ImportError as e:
        raise ImportError("Para leer .nwb instala el extra: pip install mouseosc[nwb]") from e
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        ts = nwb.acquisition[opts.get("series", "ElectricalSeries")]
        sig = np.asarray(ts.data[:], dtype=float).squeeze()
        fs = float(ts.rate) if ts.rate else None
    return sig, fs


LOADERS = {"mat": load_mat, "csv": load_csv, "abf": load_abf, "nwb": load_nwb}
_EXT2FMT = {".mat": "mat", ".csv": "csv", ".tsv": "csv", ".abf": "abf", ".nwb": "nwb"}


def _resolve_format(path: Path, fmt: str) -> str:
    """Decide el formato: explícito o 'auto' por extensión."""
    if fmt and fmt != "auto":
        return fmt
    ext = path.suffix.lower()
    if ext not in _EXT2FMT:
        raise ValueError(f"No sé leer la extensión '{ext}' de {path.name}. "
                         f"Fija dataset.format en el config.")
    return _EXT2FMT[ext]


def load_signal(rec: Recording, cfg: dict) -> tuple[np.ndarray, float]:
    """
    Carga la señal de un Recording y devuelve (señal_1D, fs).

    Si el archivo trae su propio fs (CSV con tiempo, ABF, NWB) se usa ése y se
    VERIFICA contra el del config (checks lo hará). Si no, se usa el del config.
    """
    ds = cfg.get("dataset", {})
    fmt = _resolve_format(rec.file_path, ds.get("format", "auto"))
    opts = ds.get(fmt, {})
    sig, fs_file = LOADERS[fmt](rec.file_path, opts)
    fs = fs_file if fs_file is not None else cfg["preprocessing"]["fs"]
    return np.asarray(sig, dtype=float).ravel(), float(fs)


def read_manifest(cfg: dict) -> list[Recording]:
    """Lee el manifiesto CSV y devuelve la lista de Recording."""
    ds = cfg.get("dataset", {})
    root = Path(ds.get("root", "."))
    man_path = Path(ds.get("manifest", "manifest.csv"))
    if not man_path.is_absolute():
        man_path = root.parent / man_path if (root / man_path).exists() is False else root / man_path
    df = pd.read_csv(man_path)
    required = {"file_path", "group", "animal_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Al manifiesto {man_path} le faltan columnas: {missing}")
    recs = []
    for _, row in df.iterrows():
        fp = Path(row["file_path"])
        if not fp.is_absolute():
            fp = root / fp
        extra = {k: row[k] for k in df.columns if k not in required}
        recs.append(Recording(file_path=fp, group=str(row["group"]),
                              animal_id=str(row["animal_id"]), meta=extra))
    return recs


def scan_folder(folder: Path, out_csv: Path) -> pd.DataFrame:
    """
    Utilidad de arranque: recorre una carpeta, lista los archivos de señal
    reconocidos y escribe una PLANTILLA de manifiesto para que el usuario
    rellene las columnas group/animal_id. No adivina metadatos.
    """
    rows = []
    for fp in sorted(Path(folder).rglob("*")):
        if fp.suffix.lower() in _EXT2FMT:
            rows.append({"file_path": str(fp), "group": "", "animal_id": fp.stem})
    df = pd.DataFrame(rows, columns=["file_path", "group", "animal_id"])
    df.to_csv(out_csv, index=False)
    return df
