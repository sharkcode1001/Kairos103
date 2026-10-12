# Benchmark and Time-Warp Stress Datasets (WarpStress v1)

> **Language / Idioma:** **English** | [Español (Spanish)](README_ES.md)

This directory contains the experimental datasets used to validate the **strict time-warp invariance** of the **Kairos103** architecture against continuous deep learning baselines (Transformers and LSTMs).

---

## 1. Directory Structure

```text
datasets/
├── README.md               # Technical documentation in English
├── README_ES.md            # Technical documentation in Spanish
├── synthetic/              # Synthetic 1D Dataset (WarpStress 1D)
│   ├── warpstress_train.csv
│   ├── warpstress_test_0.5x.csv
│   ├── warpstress_test_0.8x.csv
│   ├── warpstress_test_1.0x.csv
│   ├── warpstress_test_1.2x.csv
│   └── warpstress_test_2.0x.csv
├── ecg/                    # Real Biomedical Dataset (MIT-BIH Arrhythmia Database)
│   ├── ecg_train.csv
│   ├── ecg_test_0.5x.csv
│   ├── ecg_test_0.8x.csv
│   ├── ecg_test_1.0x.csv
│   ├── ecg_test_1.2x.csv
│   └── ecg_test_2.0x.csv
└── generators/             # Deterministic Python scripts for dataset generation
    ├── gen_warpstress.py   # Synthetic dataset generator (seed 20261010)
    ├── gen_ecg.py          # MIT-BIH extractor and resampler (seed 20261010)
    ├── preview.py          # Visual inspection plotting script
    └── preview.png         # Waveform chart across classes and scale factors
```

---

## 2. Dataset Descriptions

### A. Synthetic Dataset (`datasets/synthetic/`)
* **Nature:** 4 classes of synthetic continuous 1D signals where information resides exclusively in the **morphology and relative sequence of events**, rather than absolute metric duration:
  * **Class 0:** Two Gaussian pulses: low frequency first ($4\,\text{Hz}$), high frequency second ($13\,\text{Hz}$).
  * **Class 1:** Two Gaussian pulses: high frequency first ($13\,\text{Hz}$), low frequency second ($4\,\text{Hz}$).
  * **Class 2:** Upward chirp sweep ($3\,\text{Hz} \to 15\,\text{Hz}$).
  * **Class 3:** Downward chirp sweep ($15\,\text{Hz} \to 3\,\text{Hz}$).
* **Base Length ($1.0\times$):** 128 samples.
* **Maximum Padded Length:** 256 columns ($m_0 \dots m_{255}$).
* **Partitions:**
  * **Train:** 1,600 samples (400 per class).
  * **Test per Factor:** 800 samples (200 per class) for each factor ($0.5\times, 0.8\times, 1.0\times, 1.2\times, 2.0\times$).

### B. Real ECG Dataset (`datasets/ecg/`)
* **Nature:** Real heartbeats extracted from the **MIT-BIH Arrhythmia Database** (PhysioNet).
  * **Class 0:** Normal beat (**N**) from records 100, 208, 213, 221.
  * **Class 1:** Premature ventricular contraction (**V**) from records 207, 208, 213, 221.
* **Segmentation:** $0.5\,\text{s}$ windows (180 samples at $360\,\text{Hz}$) centered on the R-peak, channel 0, normalized by each beat's peak value.
* **Base Length ($1.0\times$):** 180 samples.
* **Maximum Padded Length:** 360 columns ($m_0 \dots m_{359}$).
* **Partitions:**
  * **Train:** 800 samples (400 per class).
  * **Test per Factor:** 400 samples (200 per class) for each factor ($0.5\times, 0.8\times, 1.0\times, 1.2\times, 2.0\times$).
* **Methodological Scope & Future Clinical Validation:**
  * The current benchmark uses deterministic beat-level extraction to assess **morphological invariant encoding under simulated heart-rate scaling (bradycardia at $0.5\times$, tachycardia at $2.0\times$)**.
  * While beats in test sets are strictly distinct instances from train beats, full clinical deployment benchmarking will benefit from inter-patient (*leave-one-patient-out*) cross-validation to isolate patient-specific morphologies.

---

## 3. File Format (CSV)

All CSV files follow an identical standardized structure:

```csv
label,length,m0,m1,m2,...,mN
```

* `label` ($int$): Signal class ($0$ to $3$ in synthetic; $0$ to $1$ in ECG).
* `length` ($int$): Number of effective samples before signal termination (varies by speed factor: $64$ to $256$ in synthetic, $90$ to $360$ in ECG).
* `m0, m1, ...`: Continuous normalized signal amplitude values. Indices beyond `length` are zero-padded ($0.0$).

---

## 4. Experimental Protocol and Time-Warping Factors

1. **Training ($1.0\times$ Nominal):**
   * Trained strictly with nominal speed signals ($1.0\times$), slight amplitude noise ($\sigma = 0.03$), and **zero temporal jitter**.
   * No time-warp data augmentation is applied to evaluate intrinsic architectural invariance.
2. **Stress Evaluation ($0.5\times$ to $2.0\times$):**
   * Evaluated factors: $0.5\times$ (twice the duration/slow), $0.8\times$, $1.0\times$ (control), $1.2\times$, $2.0\times$ (twice the speed/fast).
   * All test sets include monotonic temporal jitter ($\Delta t$) and Gaussian amplitude noise ($\sigma = 0.05$).
   * The $1.0\times$ test condition serves as a control group to isolate jitter impact from temporal scaling impact.

---

## 5. Success and Stopping Criteria

* **Success:**
  1. Kairos103 maintains nominal accuracy within $\pm 2\%$ in moderate scaling ($0.8\times - 1.2\times$) and $\ge 90\%$ (drop $\le 10\%$) in extreme scaling ($0.5\times, 2.0\times$) across both datasets.
  2. Continuous baseline models (Transformer / LSTM) suffer an accuracy drop $\ge 15\%$ at $0.5\times$ or $2.0\times$.
  3. Kairos103 consumes $\le 1/10$ of the compute per decision compared to dense self-attention (Transformer).
* **Stopping / Rejection:**
  * If Kairos103 fails to maintain flat accuracy stability, the hypothesis is rejected without artificially altering baselines.
  * If baselines do not degrade under temporal dilation, the comparison is not pursued without formal redefinition.
