"""Convert MIT-BIH WFDB beat annotations to NPPE2-compatible CSV.

This script only prepares and validates data; it does not train a model.
Dependency: pip install wfdb numpy
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import wfdb


LABELS = {
    # Normal and normal-like beats.
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    # Supraventricular ectopic beats.
    "A": 1, "a": 1, "J": 1, "S": 1,
    # Ventricular ectopic beats.
    "V": 2, "E": 2,
    # Fusion of ventricular and normal beat.
    "F": 3,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Folder containing WFDB .hea/.dat/.atr files")
    p.add_argument("--output", required=True, help="Output NPPE2-compatible CSV")
    p.add_argument("--audit", required=True, help="Output JSON audit report")
    p.add_argument("--window-left", type=int, default=100)
    p.add_argument("--window-right", type=int, default=150)
    p.add_argument("--channel", default="MLII")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="Optional deterministic cap; 0 means no cap")
    return p.parse_args()


def choose_channel(record: wfdb.io.record.Record, preferred: str) -> int:
    names = [str(x).upper() for x in (record.sig_name or [])]
    preferred = preferred.upper()
    if preferred in names:
        return names.index(preferred)
    return 0


def finite_minmax(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Signal contains no finite values")
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float64)
    return (values - lo) / (hi - lo)


def main() -> None:
    args = parse_args()
    root = Path(args.input)
    output = Path(args.output)
    audit_path = Path(args.audit)
    width = args.window_left + args.window_right
    if width != 250:
        raise ValueError("NPPE2 conversion requires window-left + window-right = 250")

    records = sorted(p.stem for p in root.glob("*.hea"))
    if not records:
        raise FileNotFoundError(f"No .hea records found in {root}")

    fields = ["id"] + [f"sig_{i}" for i in range(250)] + ["pre_rr", "post_rr", "rr_ratio", "label"]
    rows: list[list[object]] = []
    stats = Counter()
    skipped = Counter()
    per_record = defaultdict(Counter)

    for record_id in records:
        try:
            record = wfdb.rdrecord(str(root / record_id), physical=True)
            ann = wfdb.rdann(str(root / record_id), "atr")
        except Exception as exc:  # keep a complete audit and continue
            skipped["read_error"] += 1
            print(f"[WARN] {record_id}: {exc}")
            continue

        channel = choose_channel(record, args.channel)
        signal = np.asarray(record.p_signal[:, channel], dtype=np.float64)
        positions = np.asarray(ann.sample, dtype=np.int64)
        symbols = list(ann.symbol)
        kept_positions = [int(pos) for pos, sym in zip(positions, symbols) if sym in LABELS]
        kept_symbols = [sym for sym in symbols if sym in LABELS]

        for idx, (pos, sym) in enumerate(zip(kept_positions, kept_symbols)):
            start = pos - args.window_left
            end = pos + args.window_right
            if start < 0 or end > len(signal):
                skipped["edge_window"] += 1
                continue
            if idx == 0 or idx + 1 >= len(kept_positions):
                skipped["missing_rr"] += 1
                continue

            beat = signal[start:end]
            if beat.shape[0] != 250 or not np.all(np.isfinite(beat)):
                skipped["invalid_signal"] += 1
                continue

            pre_rr = float(pos - kept_positions[idx - 1])
            post_rr = float(kept_positions[idx + 1] - pos)
            if pre_rr <= 0 or post_rr <= 0:
                skipped["invalid_rr"] += 1
                continue

            label = LABELS[sym]
            if args.max_per_class and per_record[record_id][label] >= args.max_per_class:
                skipped["class_cap"] += 1
                continue

            normalized = finite_minmax(beat)
            row = [f"mit_{record_id}_{pos:07d}"]
            row.extend(float(x) for x in normalized)
            row.extend([pre_rr, post_rr, pre_rr / post_rr, label])
            rows.append(row)
            stats[label] += 1
            per_record[record_id][label] += 1

    if not rows:
        raise RuntimeError("No valid beats were produced")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)

    report = {
        "input": str(root),
        "output": str(output),
        "records_found": len(records),
        "rows_written": len(rows),
        "signal_window": width,
        "channel_preference": args.channel,
        "labels": {str(k): int(v) for k, v in sorted(stats.items())},
        "skipped": dict(skipped),
        "annotation_mapping": LABELS,
        "validation": {
            "header_columns": len(fields),
            "signal_columns": 250,
            "labels_in_range": all(int(row[-1]) in range(4) for row in rows),
            "all_rows_have_expected_width": all(len(row) == len(fields) for row in rows),
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
