# Datasets de Benchmark y Estrés Temporal (WarpStress v1)

> **Idioma / Language:** [English (Inglés)](README.md) | **Español**

Este directorio contiene los conjuntos de datos experimentales utilizados para validar la **invarianza estricta al escalamiento temporal (*time-warping*)** de la arquitectura **Kairos103** frente a baselines continuos (Transformers y LSTMs).

---

## 1. Estructura de Directorios

```text
datasets/
├── README.md               # Documentación en inglés
├── README_ES.md            # Esta documentación técnica en español
├── synthetic/              # Dataset 1D Sintético (WarpStress 1D)
│   ├── warpstress_train.csv
│   ├── warpstress_test_0.5x.csv
│   ├── warpstress_test_0.8x.csv
│   ├── warpstress_test_1.0x.csv
│   ├── warpstress_test_1.2x.csv
│   └── warpstress_test_2.0x.csv
├── ecg/                    # Dataset Biomédico Real (MIT-BIH Arrhythmia Database)
│   ├── ecg_train.csv
│   ├── ecg_test_0.5x.csv
│   ├── ecg_test_0.8x.csv
│   ├── ecg_test_1.0x.csv
│   ├── ecg_test_1.2x.csv
│   └── ecg_test_2.0x.csv
└── generators/             # Scripts deterministas en Python para regeneración de datos
    ├── gen_warpstress.py   # Generador del dataset sintético (semilla 20261010)
    ├── gen_ecg.py          # Extractor y remuestreador MIT-BIH (semilla 20261010)
    ├── preview.py          # Script generador de la inspección visual (antes vista_previa.py)
    └── preview.png         # Gráfica de formas de onda por clase y velocidad
```

---

## 2. Descripción de los Datasets

### A. Dataset Sintético (`datasets/synthetic/`)
* **Naturaleza:** 4 clases de señales continuas 1D sintéticas donde la información reside exclusivamente en la **morfología y secuencia relativa de los eventos**, no en su duración métrica absoluta:
  * **Clase 0:** Dos pulsos gaussianos: frecuencia grave primero ($4\,\text{Hz}$), agudo después ($13\,\text{Hz}$).
  * **Clase 1:** Dos pulsos gaussianos: agudo primero ($13\,\text{Hz}$), grave después ($4\,\text{Hz}$).
  * **Clase 2:** Barrido chirp ascendente ($3\,\text{Hz} \to 15\,\text{Hz}$).
  * **Clase 3:** Barrido chirp descendente ($15\,\text{Hz} \to 3\,\text{Hz}$).
* **Longitud base ($1.0\times$):** 128 muestras.
* **Longitud máxima con padding:** 256 columnas ($m_0 \dots m_{255}$).
* **Partición:**
  * **Train:** 1,600 ejemplos (400 por clase).
  * **Test por factor:** 800 ejemplos (200 por clase) para cada uno de los factores ($0.5\times, 0.8\times, 1.0\times, 1.2\times, 2.0\times$).

### B. Dataset Real ECG (`datasets/ecg/`)
* **Naturaleza:** Latidos reales extraídos de la base de datos **MIT-BIH Arrhythmia Database** (PhysioNet).
  * **Clase 0:** Latido normal (**N**) proveniente de registros 100, 208, 213, 221.
  * **Clase 1:** Latido ventricular prematuro (**V**) proveniente de registros 207, 208, 213, 221.
* **Segmentación:** Ventanas de $0.5\,\text{s}$ (180 muestras a $360\,\text{Hz}$) centradas en el pico R, canal 0, con normalización por su propio valor pico.
* **Longitud base ($1.0\times$):** 180 muestras.
* **Longitud máxima con padding:** 360 columnas ($m_0 \dots m_{359}$).
* **Partición:**
  * **Train:** 800 ejemplos (400 por clase).
  * **Test por factor:** 400 ejemplos (200 por clase) para cada uno de los factores ($0.5\times, 0.8\times, 1.0\times, 1.2\times, 2.0\times$).

---

## 3. Formato de Archivos (CSV)

Todos los archivos CSV comparten una estructura idéntica normalizada:

```csv
label,length,m0,m1,m2,...,mN
```

* `label` ($int$): Clase de la señal ($0$ a $3$ en sintético; $0$ a $1$ en ECG).
* `length` ($int$): Cantidad de muestras efectivas antes de la terminación de la señal (varía según el factor de velocidad: $64$ a $256$ en sintético, $90$ a $360$ en ECG).
* `m0, m1, ...`: Valores continuos normalizados de la señal. Todo valor posterior al índice `length` contiene `0.0` (cero relleno / padding).

---

## 4. Protocolo Experimental y Factores de Time-Warping

1. **Entrenamiento ($1.0\times$ Nominal):**
   * Se entrena estrictamente con señales a velocidad nominal ($1.0\times$), con ruido leve de amplitud ($\sigma = 0.03$) y **sin jitter temporal**.
   * No se aplica aumento de datos con warp (*data augmentation*) para evaluar la invariancia intrínseca de la arquitectura.
2. **Evaluación de Estrés ($0.5\times$ a $2.0\times$):**
   * Factores evaluados: $0.5\times$ (doble de duración/lenta), $0.8\times$, $1.0\times$ (control), $1.2\times$, $2.0\times$ (doble de velocidad/rápida).
   * Todos los sets de test incorporan adición de jitter temporal monótono ($\Delta t$) y ruido gaussiano de amplitud ($\sigma = 0.05$).
   * El test $1.0\times$ actúa como grupo de control para aislar el impacto del jitter frente al impacto del escalamiento temporal.

---

## 5. Criterios de Éxito y Parada

* **Éxito:**
  1. Kairos103 mantiene su precisión nominal ($1.0\times \pm 3\%$) a lo largo de los 5 factores en ambos datasets.
  2. Los modelos baseline continuos (Transformer / LSTM) experimentan una degradación $\ge 10\%$ en $0.5\times$ o $2.0\times$.
  3. Kairos103 consume $\le 1/10$ del cómputo por inferencia respecto al baseline.
* **Parada / Rechazo:**
  * Si Kairos103 no mantiene estabilidad plana en su precisión, la hipótesis se rechaza sin alterar artificialmente los baselines.
  * Si el baseline no degrada ante dilatación temporal, no se prosigue la comparativa sin redefinición formal.
