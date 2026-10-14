#!/usr/bin/env python3
"""
Kairos103 Competition & Kaggle Submission Helper
================================================
This script integrates Kairos103 with standard competitive ML platforms (Kaggle, PhysioNet, CodaLab).
It reads competition test sets, formats them for Kairos103 CLI inference, and formats the output into
the standard submission format (e.g., submission.csv with 'Id' and 'Prediction' columns).

Usage:
    python kaggle_evaluator.py --model model.kairos --test test.csv --output submission.csv
"""

import argparse
import subprocess
import os
import sys
import pandas as pd
import numpy as np


def ensure_kairos_cli():
    """Locates or compiles the Kairos103 CLI executable."""
    # Check if dotnet is available
    res = subprocess.run(["dotnet", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0:
        return "dotnet run --project src/Kairos103.Cli --"
    
    # Check if standalone binary exists
    if os.path.exists("./Kairos103.Cli"):
        return "./Kairos103.Cli"
    elif os.path.exists("./Kairos103.Cli.exe"):
        return "./Kairos103.Cli.exe"
        
    raise RuntimeError("Neither 'dotnet' SDK nor compiled 'Kairos103.Cli' binary found in PATH.")


def prepare_kaggle_input(input_csv, temp_csv):
    """
    Standardizes CSV data for Kairos103 ingestion.
    Supports:
      - Standard Kaggle format: rows of time-series values (e.g., 187 values per row).
      - Header or no-header CSVs.
    """
    df = pd.read_csv(input_csv, header=None)
    
    # Check if first row is string header
    try:
        float(df.iloc[0, 0])
    except ValueError:
        df = pd.read_csv(input_csv)
    
    # If there is an 'Id' column, separate it
    id_col = None
    if 'Id' in df.columns or 'id' in df.columns:
        id_name = 'Id' if 'Id' in df.columns else 'id'
        id_col = df[id_name].values
        df = df.drop(columns=[id_name])
    else:
        id_col = np.arange(len(df))

    # Kairos expects: Label, Length, v0, v1, ... or Label, v0, v1, ...
    # For blind test sets, we set dummy label 0
    num_samples = len(df)
    seq_len = df.shape[1]
    
    print(f"[*] Processing {num_samples} samples of length {seq_len}...")
    
    formatted_data = []
    for i in range(num_samples):
        vals = df.iloc[i].values.astype(float)
        row_str = f"0,{seq_len}," + ",".join(f"{v:.6f}" for v in vals)
        formatted_data.append(row_str)
        
    with open(temp_csv, "w") as f:
        f.write("\n".join(formatted_data))
        
    return id_col


def run_inference(cli_cmd, model_path, test_csv, raw_output_csv):
    """Invokes Kairos103 predict command."""
    cmd = f"{cli_cmd} predict --model {model_path} --data {test_csv} --output {raw_output_csv}"
    print(f"[*] Executing: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError("Kairos103 inference failed.")


def build_submission(id_col, raw_output_csv, final_submission_csv):
    """Formats raw predictions into official Kaggle submission format."""
    pred_df = pd.read_csv(raw_output_csv)
    # raw_output_csv contains: Index, TrueLabel, PredictedLabel, Spikes, FrontEndMacs
    sub = pd.DataFrame({
        "Id": id_col,
        "Prediction": pred_df["PredictedLabel"].values
    })
    sub.to_csv(final_submission_csv, index=False)
    print(f"[+] Successfully generated official submission file: {final_submission_csv}")
    print(f"    Total predictions: {len(sub)}")
    print(f"    Class distribution:\n{sub['Prediction'].value_counts().to_string()}")


def main():
    parser = argparse.ArgumentParser(description="Kairos103 Competition Evaluator")
    parser.add_argument("--model", default="model.kairos", help="Path to trained .kairos model")
    parser.add_argument("--test", required=True, help="Path to competition test CSV")
    parser.add_argument("--output", default="submission.csv", help="Output submission CSV")
    args = parser.parse_args()

    cli_cmd = ensure_kairos_cli()
    temp_input = "_temp_kairos_input.csv"
    temp_output = "_temp_kairos_preds.csv"

    try:
        id_col = prepare_kaggle_input(args.test, temp_input)
        run_inference(cli_cmd, args.model, temp_input, temp_output)
        build_submission(id_col, temp_output, args.output)
    finally:
        for f in [temp_input, temp_output]:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    main()
