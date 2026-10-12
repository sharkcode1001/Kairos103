# Kairos103 - Benchmark Submission Record

- **Model:** Kairos103 (Clockless Neuromorphic Architecture)
- **Date:** 2026-10-12
- **Primary Metric:** Strict Time-Warp Invariance ($0.5\times$ to $2.0\times$) and Zero-Float Multiplication Classification

## Empirical Results Summary

### 1. Synthetic WarpStress 1D
| Scale Factor | Speed | Kairos103 Accuracy | Baseline Transformer | Baseline LSTM |
| :---: | :---: | :---: | :---: | :---: |
| 0.5x | - | **99.00%** | ~30-49% | ~0-38% |
| 0.8x | - | **98.88%** | ~30-49% | ~0-38% |
| 1.0x | - | **98.75%** | ~30-49% | ~0-38% |
| 1.2x | - | **98.38%** | ~30-49% | ~0-38% |
| 2.0x | - | **91.62%** | ~30-49% | ~0-38% |

### 2. MIT-BIH Arrhythmia ECG (Real Biomedical Dataset)
| Scale Factor | Speed | Kairos103 Accuracy | Baseline Transformer | Baseline LSTM |
| :---: | :---: | :---: | :---: | :---: |
| 0.5x | - | **100.00%** | ~88.5% | ~56.5% |
| 0.8x | - | **99.75%** | ~88.5% | ~56.5% |
| 1.0x | - | **100.00%** | ~88.5% | ~56.5% |
| 1.2x | - | **100.00%** | ~88.5% | ~56.5% |
| 2.0x | - | **100.00%** | ~88.5% | ~56.5% |

## Verification
Reproduce locally with single command:
```bash
dotnet run --project src/Kairos103.Cli -- benchmark
```
