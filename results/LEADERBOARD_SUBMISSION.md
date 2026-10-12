# Kairos103 - Benchmark Submission Record

- **Model:** Kairos103 (Clockless Neuromorphic Architecture)
- **Date:** 2026-10-12
- **Primary Metric:** Strict Time-Warp Invariance ($0.5\times$ to $2.0\times$) and Zero-Float Multiplication Classification

## Empirical Results Summary

### 1. Synthetic WarpStress 1D
| Scale Factor | Kairos103 Accuracy | Baseline Transformer | Baseline LSTM |
| :---: | :---: | :---: | :---: |
| 0.5x | **99.00%** | 31.00% | 30.00% |
| 0.8x | **98.88%** | 49.50% | 38.50% |
| 1.0x | **98.75%** | 80.50% | 98.50% |
| 1.2x | **98.38%** | 39.00% | 0.00% |
| 2.0x | **91.62%** | 24.50% | 0.50% |

### 2. MIT-BIH Arrhythmia ECG (Real Biomedical Dataset)
| Scale Factor | Kairos103 Accuracy | Baseline Transformer | Baseline LSTM |
| :---: | :---: | :---: | :---: |
| 0.5x | **100.00%** | 90.00% | 56.50% |
| 0.8x | **99.75%** | 94.50% | 74.00% |
| 1.0x | **100.00%** | 97.00% | 73.50% |
| 1.2x | **100.00%** | 88.50% | 82.50% |
| 2.0x | **100.00%** | 88.50% | 85.00% |

> *Note on ECG LSTM behavior:* The accuracy increase in LSTM under $2.0\times$ compression (73.50% at nominal vs. 85.00% at $2.0\times$) is an artifact of sequence length halving ($T \to T/2$), which mitigates recurrent forget-gate drift over shorter time horizons.

## Verification
Reproduce locally with single commands:
```bash
# Regenerate standardized leaderboard submission artifacts:
dotnet run --project src/Kairos103.Cli -- online-bench

# Run full comparative benchmark suite (Kairos vs Transformer vs LSTM):
dotnet run --project src/Kairos103.Cli -- benchmark
```
