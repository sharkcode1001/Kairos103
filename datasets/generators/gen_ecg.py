#!/usr/bin/env python3
"""
gen_ecg.py -- Real ECG dataset generator (MIT-BIH Arrhythmia Database) with time-warping stress.

2 classes: 0 = normal beat (N), 1 = premature ventricular contraction (V).
Segments of 0.5 s (180 samples at 360 Hz) centered on the R-peak,
each beat normalized by its own peak value.
Train at nominal 1.0x speed (no jitter). Test at 0.5/0.8/1.0/1.2/2.0x with
temporal jitter + amplitude noise (1.0x test acts as the control group).

MIT-BIH records used (PhysioNet, https://physionet.org/content/mitdb/1.0.0/):
  N: 100, 208, 213, 221
  V: 207, 208, 213, 221
Files (.hea/.dat/.atr) are located in ecg_raw/ (downloaded from PhysioNet).
Channel 0 is used for each record.

Deterministic with SEED = 20261010.

Requires: pip install wfdb (tested with wfdb 4.3.1)
Usage:
    cd <folder_with_ecg_raw> && python3 gen_ecg.py [output_dir]
"""

import csv
import os
import sys

import numpy as np
import wfdb

SEED = 20261010
WINDOW = 90                       # Samples on each side of R-peak -> base length 180
BASE_LEN = 2 * WINDOW
FACTORS = [0.5, 0.8, 1.0, 1.2, 2.0]
MAX_LEN = int(round(BASE_LEN / min(FACTORS)))   # 360 columns m0..m359
TRAIN_PER_CLASS = 400
TEST_PER_CLASS = 200
TRAIN_NOISE = 0.03
TEST_NOISE = 0.05
TEST_JITTER = 0.6                 # Samples (jitter std dev on time grid)

RECORDS = {"N": ["100", "208", "213", "221"],
           "V": ["207", "208", "213", "221"]}
CLASS_OF = {"N": 0, "V": 1}


def extract_beats(symbol, limit):
    """Extract beats for the given symbol across records up to 'limit'."""
    beats = []
    for rec in RECORDS[symbol]:
        signal = wfdb.rdrecord(rec).p_signal[:, 0]
        ann = wfdb.rdann(rec, "atr")
        for s, sym in zip(ann.sample, ann.symbol):
            if sym != symbol:
                continue
            if s <= WINDOW or s + WINDOW >= len(signal):
                continue
            beat = signal[s - WINDOW:s + WINDOW].astype(float).copy()
            peak = np.max(np.abs(beat))
            if peak < 1e-6:
                continue
            beats.append(beat / peak)
            if len(beats) >= limit:
                return beats
    return beats


def warp(beat, factor, rng, amp_noise, jitter):
    length = int(round(BASE_LEN / factor))
    i = np.arange(length)
    pos = i * factor                       # Position in base beat (0..179)
    if jitter > 0:
        pos = pos + rng.normal(0.0, jitter, length)
        pos = np.maximum.accumulate(pos)   # Monotonic time warp
    pos = np.clip(pos, 0.0, BASE_LEN - 1)
    y = np.interp(pos, np.arange(BASE_LEN), beat)
    y = y + rng.normal(0.0, amp_noise, length)
    return y


def save_csv(file_path, rows):
    header = ["label", "length"] + [f"m{k}" for k in range(MAX_LEN)]
    with open(file_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for label, y in rows:
            length = len(y)
            writer.writerow([label, length]
                            + [f"{v:.6f}" for v in y]
                            + ["0.0"] * (MAX_LEN - length))


def main():
    default_output = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ecg"))
    output_dir = sys.argv[1] if len(sys.argv) > 1 else default_output
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    total = TRAIN_PER_CLASS + TEST_PER_CLASS * len(FACTORS)

    train_rows, test_rows = [], {f: [] for f in FACTORS}
    for symbol in ["N", "V"]:
        beats = extract_beats(symbol, total)
        print(f"{symbol}: {len(beats)} beats (using {total})")
        assert len(beats) >= total, f"insufficient beats for {symbol}"
        idx = np.arange(total)
        rng.shuffle(idx)
        beats = [beats[k] for k in idx[:total]]
        label = CLASS_OF[symbol]
        train_rows += [(label, warp(b, 1.0, rng, TRAIN_NOISE, 0.0))
                       for b in beats[:TRAIN_PER_CLASS]]
        k = TRAIN_PER_CLASS
        for factor in FACTORS:
            test_rows[factor] += [(label, warp(b, factor, rng, TEST_NOISE, TEST_JITTER))
                                  for b in beats[k:k + TEST_PER_CLASS]]
            k += TEST_PER_CLASS

    rng.shuffle(train_rows)
    save_csv(os.path.join(output_dir, "ecg_train.csv"), train_rows)
    print(f"ecg_train.csv: {len(train_rows)} rows")
    for factor in FACTORS:
        rows = test_rows[factor]
        rng.shuffle(rows)
        filename = f"ecg_test_{factor}x.csv"
        save_csv(os.path.join(output_dir, filename), rows)
        lengths = sorted({len(y) for _, y in rows})
        print(f"{filename}: {len(rows)} rows, lengths {lengths}")


if __name__ == "__main__":
    main()
