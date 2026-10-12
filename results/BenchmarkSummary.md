# Consolidated Technical Benchmark Report: Kairos103

* **Date:** 2026-10-12 22:33:24 UTC
* **Platform:** .NET 10.0 (Asynchronous C#, Headless CLI)
* **Architecture:** Modular Rank-Order Network with Coupled N=4 Subregions

## 1. Time-Warp Invariance Summary
Experimental results validate stability against affine temporal transformations ($t' = \alpha t + \beta$).

### A. Dataset: SYNTHETIC

### Benchmark Results: synthetic

| Warp Factor | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| 0.5x | 99.00% | 45.38% | 75.62% |
| 0.8x | 98.88% | 63.62% | 97.38% |
| 1.0x | 98.75% | 99.88% | 99.88% |
| 1.2x | 98.38% | 45.62% | 99.38% |
| 2.0x | 91.62% | 46.38% | 51.75% |



#### Computational Efficiency Metrics
- **Kairos Front-End MACs (continuous):** 245.760.000
- **Kairos Discrete Spikes (events):** 255.994
- **Kairos Readout Accumulations:** 256.000
- **Mini-Transformer MACs:** 3.590.272.000
- **Simple LSTM MACs:** 622.592.000
- **Allocated Heap Memory:** 1.590.273.672 bytes

### B. Dataset: ECG

### Benchmark Results: ecg

| Warp Factor | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| 0.5x | 100.00% | 92.00% | 66.25% |
| 0.8x | 99.75% | 95.25% | 64.00% |
| 1.0x | 100.00% | 97.25% | 65.50% |
| 1.2x | 100.00% | 84.50% | 62.50% |
| 2.0x | 100.00% | 91.25% | 65.50% |



#### Computational Efficiency Metrics
- **Kairos Front-End MACs (continuous):** 122.880.000
- **Kairos Discrete Spikes (events):** 117.881
- **Kairos Readout Accumulations:** 63.476
- **Mini-Transformer MACs:** 3.422.144.000
- **Simple LSTM MACs:** 437.440.000
- **Allocated Heap Memory:** 1.332.848.432 bytes

