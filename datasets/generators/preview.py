#!/usr/bin/env python3
"""Plot one representative sample for each class across all test speed factors."""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FACTORS = [0.5, 0.8, 1.0, 1.2, 2.0]
CLASS_NAMES = {
    0: "low->high",
    1: "high->low",
    2: "chirp up",
    3: "chirp down",
}


def first_sample(path, class_id):
    with open(path) as fh:
        reader = csv.reader(fh)
        next(reader)  # Header
        for row in reader:
            if int(row[0]) == class_id:
                n = int(row[1])
                return np.array([float(v) for v in row[2:2 + n]])
    raise ValueError(f"No class {class_id} found in {path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    synthetic_dir = os.path.join(script_dir, "..", "synthetic")

    fig, axes = plt.subplots(4, 5, figsize=(14, 8), sharey=True)
    for ci in range(4):
        for fi, f in enumerate(FACTORS):
            ax = axes[ci][fi]
            csv_path = os.path.join(synthetic_dir, f"warpstress_test_{f}x.csv")
            if not os.path.exists(csv_path):
                csv_path = f"warpstress_test_{f}x.csv"
            y = first_sample(csv_path, ci)
            ax.plot(y, linewidth=0.8)
            ax.set_title(f"Class {ci} ({CLASS_NAMES[ci]}) @ {f}x  n={len(y)}", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("WarpStress v1: One Sample per Class and Speed Factor (Test, with Jitter)")
    fig.tight_layout()
    out_png = os.path.join(script_dir, "preview.png")
    fig.savefig(out_png, dpi=90)
    print(f"{out_png} saved successfully")


if __name__ == "__main__":
    main()
