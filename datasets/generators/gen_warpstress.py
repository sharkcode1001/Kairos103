#!/usr/bin/env python3
"""
gen_warpstress.py -- Synthetic WarpStress v1 dataset generator.

4 continuous 1D signal classes. Class identity resides in the waveform MORPHOLOGY
(event order / chirp sweep direction), rather than absolute metric duration.
Consequently, a temporal rate shift should preserve the class label.

Usage:
    python3 gen_warpstress.py [output_dir]

Fully deterministic with SEED = 20261010. Re-running produces identical CSV data bit-by-bit.
"""

import csv
import os
import sys

import numpy as np

SEED = 20261010
BASE_LEN = 128            # Samples per signal at nominal 1.0x speed
FACTORS = [0.5, 0.8, 1.0, 1.2, 2.0]
MAX_LEN = int(round(BASE_LEN / min(FACTORS)))   # 256 columns m0..m255
N_CLASSES = 4
TRAIN_PER_CLASS = 400
TEST_PER_CLASS = 200
TRAIN_NOISE = 0.03        # Amplitude noise standard deviation during training
TEST_NOISE = 0.05         # Amplitude noise standard deviation during testing
TEST_JITTER = 0.004       # Temporal grid jitter standard deviation (t in [0,1))

CLASS_NAMES = {
    0: "two pulses: low freq first, high freq second",
    1: "two pulses: high freq first, low freq second",
    2: "chirp sweep: low to high (upward)",
    3: "chirp sweep: high to low (downward)",
}


def gauss(t, center, width):
    return np.exp(-(((t - center) / width) ** 2))


def pulse(t, freq, center, width):
    return np.sin(2.0 * np.pi * freq * (t - center)) * gauss(t, center, width)


def chirp(t, f0, f1, t0=0.15, t1=0.85, center=0.5, width=0.30):
    tau = np.clip(t, t0, t1) - t0
    phase = f0 * tau + (f1 - f0) * tau ** 2 / (2.0 * (t1 - t0))
    return np.sin(2.0 * np.pi * phase) * gauss(t, center, width)


def template(class_id):
    def f(t):
        t = np.asarray(t, dtype=float)
        if class_id == 0:
            y = pulse(t, 4.0, 0.28, 0.09) + pulse(t, 13.0, 0.72, 0.09)
        elif class_id == 1:
            y = pulse(t, 13.0, 0.28, 0.09) + pulse(t, 4.0, 0.72, 0.09)
        elif class_id == 2:
            y = chirp(t, 3.0, 15.0)
        else:
            y = chirp(t, 15.0, 3.0)
        return y

    # Normalize to peak 1.0 on a fine grid (equal scale across all classes)
    peak = np.max(np.abs(f(np.linspace(0, 1, 4096))))

    def fn(t):
        return f(t) / peak

    return fn


TEMPLATES = [template(c) for c in range(N_CLASSES)]


def sample(class_id, factor, rng, amp_noise, jitter):
    """Generate one signal: template resampled to scale factor + noise."""
    length = int(round(BASE_LEN / factor))
    i = np.arange(length)
    t = (i * factor) / BASE_LEN          # Normalized time coordinate
    if jitter > 0:
        t = t + rng.normal(0.0, jitter, length)
        t = np.maximum.accumulate(t)     # Time warp must remain strictly monotonic
        t = np.clip(t, 0.0, 1.0 - 1e-9)
    y = TEMPLATES[class_id](t)
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
    default_output = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "synthetic"))
    output_dir = sys.argv[1] if len(sys.argv) > 1 else default_output
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Train: 1.0x nominal speed, zero jitter, low residual noise
    rows = []
    for c in range(N_CLASSES):
        for _ in range(TRAIN_PER_CLASS):
            rows.append((c, sample(c, 1.0, rng, TRAIN_NOISE, 0.0)))
    rng.shuffle(rows)
    save_csv(os.path.join(output_dir, "warpstress_train.csv"), rows)
    print(f"warpstress_train.csv: {len(rows)} rows")

    # Test: each factor with jitter + amplitude noise (1.0x included as control)
    for factor in FACTORS:
        rows = []
        for c in range(N_CLASSES):
            for _ in range(TEST_PER_CLASS):
                rows.append((c, sample(c, factor, rng, TEST_NOISE, TEST_JITTER)))
        rng.shuffle(rows)
        filename = f"warpstress_test_{factor}x.csv"
        save_csv(os.path.join(output_dir, filename), rows)
        lengths = sorted({len(y) for _, y in rows})
        print(f"{filename}: {len(rows)} rows, lengths {lengths}")


if __name__ == "__main__":
    main()
