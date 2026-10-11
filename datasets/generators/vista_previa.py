#!/usr/bin/env python3
"""Dibuja un ejemplo de cada clase en cada factor de velocidad (del test)."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FACTORES = [0.5, 0.8, 1.0, 1.2, 2.0]
NOMBRES = {
    0: "grave->agudo",
    1: "agudo->grave",
    2: "sube",
    3: "baja",
}


def primer_ejemplo(ruta, clase):
    with open(ruta) as fh:
        r = csv.reader(fh)
        next(r)  # cabecera
        for row in r:
            if int(row[0]) == clase:
                n = int(row[1])
                return np.array([float(v) for v in row[2:2 + n]])
    raise ValueError(f"no hay clase {clase} en {ruta}")


import os

script_dir = os.path.dirname(os.path.abspath(__file__))
synthetic_dir = os.path.join(script_dir, "..", "synthetic")

fig, axes = plt.subplots(4, 5, figsize=(14, 8), sharey=True)
for ci in range(4):
    for fi, f in enumerate(FACTORES):
        ax = axes[ci][fi]
        csv_path = os.path.join(synthetic_dir, f"warpstress_test_{f}x.csv")
        if not os.path.exists(csv_path):
            csv_path = f"warpstress_test_{f}x.csv"
        y = primer_ejemplo(csv_path, ci)
        ax.plot(y, linewidth=0.8)
        ax.set_title(f"clase {ci} ({NOMBRES[ci]}) @ {f}x  n={len(y)}", fontsize=8)
        ax.tick_params(labelsize=7)
fig.suptitle("WarpStress v1: un ejemplo por clase y velocidad (test, con jitter)")
fig.tight_layout()
out_png = os.path.join(script_dir, "vista_previa.png")
fig.savefig(out_png, dpi=90)
print(f"{out_png} ok")
