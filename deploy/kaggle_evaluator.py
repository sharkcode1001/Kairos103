#!/usr/bin/env python3
"""
Kairos103 Competition & Kaggle Submission Helper
================================================
Zero-dependency Python helper for competitive ML platforms (Kaggle, PhysioNet, CodaLab).
Reads competition test sets, formats them for Kairos103 CLI inference, and formats the output
into the standard submission format (submission.csv with 'Id' and 'Prediction' columns).

Uses ONLY Python standard library (no pandas/numpy required).

Usage:
    python kaggle_evaluator.py --test test.csv --output submission.csv
    python kaggle_evaluator.py --cli ./Kairos103.Cli --model model.kairos --test test.csv --output submission.csv
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys


def resolve_cli_command(custom_cli=None):
    """Resolves the executable or dotnet invocation command as a list of arguments."""
    if custom_cli:
        if os.path.exists(custom_cli):
            return [os.path.abspath(custom_cli)]
        return custom_cli.split()

    # Look for local compiled binary in current directory or deploy/
    candidates = [
        "./Kairos103.Cli",
        "./Kairos103.Cli.exe",
        "../Kairos103.Cli",
        "../Kairos103.Cli.exe",
        "./src/Kairos103.Cli/bin/Release/net10.0/Kairos103.Cli",
        "./src/Kairos103.Cli/bin/Release/net10.0/Kairos103.Cli.exe",
        "./src/Kairos103.Cli/bin/Debug/net10.0/Kairos103.Cli",
        "./src/Kairos103.Cli/bin/Debug/net10.0/Kairos103.Cli.exe"
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK if sys.platform != "win32" else os.R_OK):
            return [os.path.abspath(c)]

    # Check if dotnet is available in PATH
    dotnet_bin = shutil.which("dotnet")
    if dotnet_bin:
        # Search for project file
        proj_candidates = [
            "src/Kairos103.Cli/Kairos103.Cli.csproj",
            "../src/Kairos103.Cli/Kairos103.Cli.csproj"
        ]
        for p in proj_candidates:
            if os.path.isfile(p):
                return [dotnet_bin, "run", "--project", os.path.abspath(p), "--"]

    raise RuntimeError(
        "Could not find a compiled Kairos103.Cli binary or dotnet SDK. "
        "Please provide the CLI path explicitly using: --cli <path_to_binary>"
    )


def prepare_kaggle_input(input_csv, temp_csv):
    """
    Standardizes CSV data for Kairos103 ingestion.
    Supports standard Kaggle format (rows of time-series values with/without header and Id).
    Writes an explicit CSV header to prevent downstream parsing ambiguities.
    """
    ids = []
    rows_written = 0
    seq_len = 0

    with open(input_csv, "r", encoding="utf-8", errors="replace") as fin, \
         open(temp_csv, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"Input file is empty: {input_csv}")

        # Check if first row is a text header
        has_header = False
        try:
            float(first_row[0].strip())
        except ValueError:
            has_header = True

        id_col_idx = None
        data_start_idx = 0

        if has_header:
            header_lower = [h.strip().lower() for h in first_row]
            if "id" in header_lower:
                id_col_idx = header_lower.index("id")
            first_data_row = next(reader, None)
        else:
            first_data_row = first_row

        if first_data_row is None:
            raise ValueError("No data rows found in input CSV.")

        # Determine sequence length
        row_values = []
        for idx, val in enumerate(first_data_row):
            if idx == id_col_idx:
                continue
            row_values.append(val.strip())

        seq_len = len(row_values)
        print(f"[*] Detected sequence length: {seq_len} features per sample.")

        # Write explicit header row to avoid parser misinterpretation
        header_cols = ["label", "length"] + [f"v{i}" for i in range(seq_len)]
        writer.writerow(header_cols)

        # Process first row
        sample_id = first_data_row[id_col_idx] if id_col_idx is not None else str(rows_written)
        ids.append(sample_id)
        writer.writerow([0, seq_len] + row_values)
        rows_written += 1

        # Process remaining rows
        for row in reader:
            if not row or not any(row):
                continue
            vals = [val.strip() for idx, val in enumerate(row) if idx != id_col_idx]
            sample_id = row[id_col_idx] if id_col_idx is not None else str(rows_written)
            ids.append(sample_id)
            writer.writerow([0, seq_len] + vals)
            rows_written += 1

    print(f"[*] Prepared {rows_written} samples for inference.")
    return ids


def run_inference(cli_cmd_parts, model_path, test_csv, raw_output_csv):
    """Executes Kairos103 inference without shell=True to ensure security and cross-platform robustness."""
    cmd = cli_cmd_parts + [
        "predict",
        "--model", os.path.abspath(model_path),
        "--data", os.path.abspath(test_csv),
        "--output", os.path.abspath(raw_output_csv)
    ]
    print(f"[*] Executing inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Kairos103 process exited with code {result.returncode}")


def build_submission(ids, raw_output_csv, final_submission_csv):
    """Extracts predictions and formats standard competition submission CSV (Id,Prediction)."""
    predictions = []
    with open(raw_output_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(row["PredictedLabel"])

    if len(ids) != len(predictions):
        raise ValueError(f"ID count ({len(ids)}) does not match prediction count ({len(predictions)})")

    with open(final_submission_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"])
        for sample_id, pred in zip(ids, predictions):
            writer.writerow([sample_id, pred])

    print(f"[+] Successfully generated submission artifact: {final_submission_csv}")
    print(f"    Total entries: {len(predictions)}")


def main():
    parser = argparse.ArgumentParser(description="Kairos103 Competition Evaluator (Zero External Dependencies)")
    parser.add_argument("--cli", default=None, help="Path to Kairos103.Cli executable or dotnet run command")
    parser.add_argument("--model", default="model.kairos", help="Path to trained .kairos model")
    parser.add_argument("--test", required=True, help="Path to competition test CSV")
    parser.add_argument("--output", default="submission.csv", help="Output submission CSV path")
    args = parser.parse_args()

    cli_cmd_parts = resolve_cli_command(args.cli)
    temp_input = os.path.abspath("_temp_kairos_test_input.csv")
    temp_output = os.path.abspath("_temp_kairos_raw_preds.csv")

    try:
        ids = prepare_kaggle_input(args.test, temp_input)
        run_inference(cli_cmd_parts, args.model, temp_input, temp_output)
        build_submission(ids, temp_output, args.output)
    finally:
        for temp_file in [temp_input, temp_output]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
