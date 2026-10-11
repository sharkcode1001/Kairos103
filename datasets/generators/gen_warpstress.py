#!/usr/bin/env python3
"""
gen_warpstress.py -- Generador del dataset sintetico WarpStress v1.

4 clases de senales continuas 1D. La clase vive en la FORMA de la onda
(orden de los eventos / direccion del barrido), no en su duracion absoluta,
asi que un cambio de velocidad temporal no deberia cambiar la etiqueta.

Uso:
    python3 gen_warpstress.py [carpeta_salida]

Todo es determinista con SEMILLA = 20261010: correrlo de nuevo
produce los mismos CSV bit a bit (salvo el orden de las filas, que
tambien es determinista por la misma semilla).
"""

import csv
import os
import sys

import numpy as np

SEMILLA = 20261010
BASE_LEN = 128            # muestras por senal a velocidad 1.0x
FACTORES = [0.5, 0.8, 1.0, 1.2, 2.0]
MAX_LEN = int(round(BASE_LEN / min(FACTORES)))   # 256 columnas m0..m255
N_CLASES = 4
TRAIN_POR_CLASE = 400
TEST_POR_CLASE = 200
RUIDO_TRAIN = 0.03        # desviacion del ruido de amplitud en train
RUIDO_TEST = 0.05         # desviacion del ruido de amplitud en test
JITTER_TEST = 0.004       # desviacion del jitter en la grilla de tiempo (t en [0,1))

NOMBRES = {
    0: "dos pulsos: grave primero, agudo despues",
    1: "dos pulsos: agudo primero, grave despues",
    2: "barrido de grave a agudo (sube)",
    3: "barrido de agudo a grave (baja)",
}


def gauss(t, centro, ancho):
    return np.exp(-(((t - centro) / ancho) ** 2))


def pulso(t, frec, centro, ancho):
    return np.sin(2.0 * np.pi * frec * (t - centro)) * gauss(t, centro, ancho)


def barrido(t, f0, f1, t0=0.15, t1=0.85, centro=0.5, ancho=0.30):
    tau = np.clip(t, t0, t1) - t0
    fase = f0 * tau + (f1 - f0) * tau ** 2 / (2.0 * (t1 - t0))
    return np.sin(2.0 * np.pi * fase) * gauss(t, centro, ancho)


def plantilla(clase):
    def f(t):
        t = np.asarray(t, dtype=float)
        if clase == 0:
            y = pulso(t, 4.0, 0.28, 0.09) + pulso(t, 13.0, 0.72, 0.09)
        elif clase == 1:
            y = pulso(t, 13.0, 0.28, 0.09) + pulso(t, 4.0, 0.72, 0.09)
        elif clase == 2:
            y = barrido(t, 3.0, 15.0)
        else:
            y = barrido(t, 15.0, 3.0)
        return y

    # Normalizar a pico 1 con una grilla fina (misma escala para todas las clases)
    pico = np.max(np.abs(f(np.linspace(0, 1, 4096))))

    def fn(t):
        return f(t) / pico

    return fn


PLANTILLAS = [plantilla(c) for c in range(N_CLASES)]


def muestra(clase, factor, rng, ruido_amp, jitter):
    """Una senal: plantilla remuestreada al factor de velocidad + ruido."""
    largo = int(round(BASE_LEN / factor))
    i = np.arange(largo)
    t = (i * factor) / BASE_LEN          # posicion en el tiempo del template
    if jitter > 0:
        t = t + rng.normal(0.0, jitter, largo)
        t = np.maximum.accumulate(t)     # el warp temporal tiene que ser monotono
        t = np.clip(t, 0.0, 1.0 - 1e-9)
    y = PLANTILLAS[clase](t)
    y = y + rng.normal(0.0, ruido_amp, largo)
    return y


def guardar_csv(ruta, filas):
    cabecera = ["etiqueta", "longitud"] + [f"m{k}" for k in range(MAX_LEN)]
    with open(ruta, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cabecera)
        for etiqueta, y in filas:
            largo = len(y)
            w.writerow([etiqueta, largo]
                       + [f"{v:.6f}" for v in y]
                       + ["0.0"] * (MAX_LEN - largo))


def main():
    default_salida = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "synthetic"))
    salida = sys.argv[1] if len(sys.argv) > 1 else default_salida
    os.makedirs(salida, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)

    # Train: 1.0x, sin jitter, ruido leve
    filas = []
    for clase in range(N_CLASES):
        for _ in range(TRAIN_POR_CLASE):
            filas.append((clase, muestra(clase, 1.0, rng, RUIDO_TRAIN, 0.0)))
    rng.shuffle(filas)
    guardar_csv(os.path.join(salida, "warpstress_train.csv"), filas)
    print(f"train: {len(filas)} filas")

    # Test: cada factor con jitter + ruido (1.0x incluido, como control)
    for factor in FACTORES:
        filas = []
        for clase in range(N_CLASES):
            for _ in range(TEST_POR_CLASE):
                filas.append((clase, muestra(clase, factor, rng, RUIDO_TEST, JITTER_TEST)))
        rng.shuffle(filas)
        nombre = f"warpstress_test_{factor}x.csv"
        guardar_csv(os.path.join(salida, nombre), filas)
        largos = sorted({len(y) for _, y in filas})
        print(f"{nombre}: {len(filas)} filas, largos {largos}")


if __name__ == "__main__":
    main()
