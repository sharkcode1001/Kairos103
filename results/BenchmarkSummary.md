# Consolidated Technical Benchmark Report: Kairos103

* **Date:** 2026-10-12 16:43:53 UTC
* **Platform:** .NET 10.0 (Asynchronous C#, Headless CLI)
* **Architecture:** Modular Rank-Order Network with Coupled N=4 Subregions

## 1. Time-Warp Invariance Summary
Experimental results validate stability against affine temporal transformations ($t' = \alpha t + \beta$).

### A. Dataset: SYNTHETIC

### Benchmark Results: synthetic

| Warp Factor | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| 0.5x | 99.00% | 31.00% | 30.00% |
| 0.8x | 98.00% | 49.50% | 38.50% |
| 1.0x | 98.50% | 80.50% | 98.50% |
| 1.2x | 98.00% | 39.00% | 0.00% |
| 2.0x | 90.00% | 24.50% | 0.50% |



#### Computational Efficiency Metrics
- **Kairos Front-End MACs (continuous):** 61.440.000
- **Kairos Discrete Spikes (events):** 63.999
- **Kairos Readout Accumulations:** 64.000
- **Mini-Transformer MACs:** 897.568.000
- **Simple LSTM MACs:** 155.648.000
- **Allocated Heap Memory:** 449.370.192 bytes

### B. Dataset: ECG

### Benchmark Results: ecg

| Warp Factor | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| 0.5x | 100.00% | 90.00% | 56.50% |
| 0.8x | 100.00% | 94.50% | 74.00% |
| 1.0x | 100.00% | 97.00% | 73.50% |
| 1.2x | 100.00% | 88.50% | 82.50% |
| 2.0x | 100.00% | 88.50% | 85.00% |



#### Computational Efficiency Metrics
- **Kairos Front-End MACs (continuous):** 61.440.000
- **Kairos Discrete Spikes (events):** 58.787
- **Kairos Readout Accumulations:** 31.744
- **Mini-Transformer MACs:** 1.711.072.000
- **Simple LSTM MACs:** 218.720.000
- **Allocated Heap Memory:** 690.794.344 bytes

