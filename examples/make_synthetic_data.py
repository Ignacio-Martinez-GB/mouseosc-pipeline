#!/usr/bin/env python3
"""
Genera un dataset SINTÉTICO de ratón para probar el pipeline de punta a punta.

Crea dos grupos con una diferencia CONOCIDA: el grupo "tratamiento" tiene más
potencia en gamma bajo (30-60 Hz). Así puedes verificar que la estadística
detecta lo que debe. Escribe CSVs + un manifest.csv listo para `python run.py run`.
"""
from pathlib import Path
import numpy as np
import pandas as pd

FS = 2000
DUR = 60          # s
N_PER_GROUP = 6
OUT = Path(__file__).resolve().parent / "datos_sinteticos"


def synth(seed, gamma_amp):
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * DUR)) / FS
    # 1/f de fondo (ruido rosa aproximado) + theta 8 Hz + gamma 45 Hz modulado por theta
    pink = np.cumsum(rng.standard_normal(len(t))); pink /= np.std(pink)
    theta = np.sin(2 * np.pi * 8 * t)
    gamma = gamma_amp * (1 + 0.6 * theta) * np.sin(2 * np.pi * 45 * t)  # PAC theta→gamma
    sig = 50 * pink + 30 * theta + gamma + 5 * rng.standard_normal(len(t))
    return t, sig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(N_PER_GROUP):
        for group, gamma_amp in (("control", 10.0), ("tratamiento", 25.0)):
            t, sig = synth(seed=1000 + i + (0 if group == "control" else 500),
                           gamma_amp=gamma_amp)
            fn = OUT / f"{group}_{i:02d}.csv"
            pd.DataFrame({"t": t, "uV": sig}).to_csv(fn, index=False)
            rows.append({"file_path": fn.name, "group": group, "animal_id": f"m{i:02d}"})
    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)
    print(f"Dataset sintético en {OUT} ({len(rows)} registros).")


if __name__ == "__main__":
    main()
