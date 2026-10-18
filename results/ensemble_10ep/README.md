# Kaggle ECG ensemble checkpoints

These three `.kairos` files are the OvR ensemble checkpoints trained for the
NPPE2-T2-26 ECG competition. They use 16 subregions, 64 hidden units, 250
waveform samples, 22 derived auxiliary features, and the serialized Adam
optimizer state.

To reuse them for Kaggle inference, download the three checkpoints and invoke
the `predict-kaggle` command with their paths separated by commas. The
submission generator preserves the order and IDs from `sample_submission.csv`.
