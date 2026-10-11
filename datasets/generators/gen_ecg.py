#!/usr/bin/env python3
"""
gen_ecg.py -- Dataset ECG real (MIT-BIH Arrhythmia Database) con stress de time-warping.

2 clases: 0 = latido normal (N), 1 = latido ventricular (V).
Segmentos de 0.5 s (180 muestras a 360 Hz) centrados en el pico R,
cada latido normalizado por su propio pico.
Train a velocidad 1.0x (sin jitter). Test a 0.5/0.8/1.0/1.2/2.0x con
jitter temporal + ruido de amplitud (el test a 1.0x lleva el mismo
jitter y ruido que los demas: es el control).

Registros MIT-BIH usados (PhysioNet, https://physionet.org/content/mitdb/1.0.0/):
  N: 100, 208, 213, 221
  V: 207, 208, 213, 221
Los ficheros .hea/.dat/.atr estan en ecg_raw/ (descargados de PhysioNet).
Se usa el canal 0 de cada registro.

Determinista con SEMILLA = 20261010.

Requiere: pip install wfdb   (probado con wfdb 4.3.1)
Uso:
    cd <carpeta con ecg_raw/> && python3 gen_ecg.py [carpeta_salida]
"""

import csv
import os
import sys

import numpy as np
import wfdb

SEMILLA = 20261010
VENTANA = 90                      # muestras a cada lado del pico R -> base 180
BASE_LEN = 2 * VENTANA
FACTORES = [0.5, 0.8, 1.0, 1.2, 2.0]
MAX_LEN = int(round(BASE_LEN / min(FACTORES)))   # 360 columnas m0..m359
TRAIN_POR_CLASE = 400
TEST_POR_CLASE = 200
RUIDO_TRAIN = 0.03
RUIDO_TEST = 0.05
JITTER_TEST = 0.6                 # muestras (desv. del jitter en la grilla)

REGISTROS = {"N": ["100", "208", "213", "221"],
             "V": ["207", "208", "213", "221"]}
CLASE_DE = {"N": 0, "V": 1}


def extraer_latidos(simbolo, tope):
    """Latidos del simbolo dado, mezclando registros, hasta 'tope'."""
    latidos = []
    for rec in REGISTROS[simbolo]:
        senal = wfdb.rdrecord(rec).p_signal[:, 0]
        ann = wfdb.rdann(rec, "atr")
        for s, sym in zip(ann.sample, ann.symbol):
            if sym != simbolo:
                continue
            if s <= VENTANA or s + VENTANA >= len(senal):
                continue
            lat = senal[s - VENTANA:s + VENTANA].astype(float).copy()
            pico = np.max(np.abs(lat))
            if pico < 1e-6:
                continue
            latidos.append(lat / pico)
            if len(latidos) >= tope:
                return latidos
    return latidos


def warp(latido, factor, rng, ruido_amp, jitter):
    largo = int(round(BASE_LEN / factor))
    i = np.arange(largo)
    pos = i * factor                       # posicion en el latido base (0..179)
    if jitter > 0:
        pos = pos + rng.normal(0.0, jitter, largo)
        pos = np.maximum.accumulate(pos)   # el warp temporal es monotono
    pos = np.clip(pos, 0.0, BASE_LEN - 1)
    y = np.interp(pos, np.arange(BASE_LEN), latido)
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
    default_salida = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ecg"))
    salida = sys.argv[1] if len(sys.argv) > 1 else default_salida
    os.makedirs(salida, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    total = TRAIN_POR_CLASE + TEST_POR_CLASE * len(FACTORES)

    tren, pruebas = [], {f: [] for f in FACTORES}
    for simbolo in ["N", "V"]:
        latidos = extraer_latidos(simbolo, total)
        print(f"{simbolo}: {len(latidos)} latidos (se usan {total})")
        assert len(latidos) >= total, f"faltan latidos {simbolo}"
        idx = np.arange(total)
        rng.shuffle(idx)
        latidos = [latidos[k] for k in idx[:total]]
        etiqueta = CLASE_DE[simbolo]
        tren += [(etiqueta, warp(b, 1.0, rng, RUIDO_TRAIN, 0.0))
                 for b in latidos[:TRAIN_POR_CLASE]]
        k = TRAIN_POR_CLASE
        for factor in FACTORES:
            pruebas[factor] += [(etiqueta, warp(b, factor, rng, RUIDO_TEST, JITTER_TEST))
                                for b in latidos[k:k + TEST_POR_CLASE]]
            k += TEST_POR_CLASE

    rng.shuffle(tren)
    guardar_csv(os.path.join(salida, "ecg_train.csv"), tren)
    print(f"ecg_train.csv: {len(tren)} filas")
    for factor in FACTORES:
        filas = pruebas[factor]
        rng.shuffle(filas)
        nombre = f"ecg_test_{factor}x.csv"
        guardar_csv(os.path.join(salida, nombre), filas)
        largos = sorted({len(y) for _, y in filas})
        print(f"{nombre}: {len(filas)} filas, largos {largos}")


if __name__ == "__main__":
    main()
