# Informe Técnico Consolidado de Benchmarking: Kairos103

* **Fecha:** 2026-10-12 01:52:33 UTC
* **Plataforma:** .NET 10.0 (C# asíncrono, Headless CLI)
* **Arquitectura:** Red Rank-Order con Subregiones N=4 acopladas

## 1. Resumen de Invarianza a Time-Warping
Los resultados experimentales confirman la estabilidad ante transformaciones temporales afines ($t' = \alpha t + \beta$).

### A. Dataset: SYNTHETIC

### Resultados de Benchmark: synthetic

| Factor Warp | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| 0.5x | 99.00% | 31.00% | 30.00% |
| 0.8x | 98.00% | 49.50% | 38.50% |
| 1.0x | 98.50% | 80.50% | 98.50% |
| 1.2x | 98.00% | 39.00% | 0.00% |
| 2.0x | 90.00% | 24.50% | 0.50% |



#### Métricas de Eficiencia Computacional
- **Kairos Front-End MACs (continuos):** 61.440.000
- **Kairos Spikes Discretos (eventos):** 63.999
- **Kairos Acumulaciones (clasificador):** 64.000
- **Mini-Transformer MACs:** 897.568.000
- **Simple LSTM MACs:** 162.512.000
- **Memoria Heap Asignada:** 449.370.880 bytes

### A. Dataset: ECG

### Resultados de Benchmark: ecg

| Factor Warp | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| 0.5x | 100.00% | 90.00% | 56.50% |
| 0.8x | 100.00% | 94.50% | 74.00% |
| 1.0x | 100.00% | 97.00% | 73.50% |
| 1.2x | 100.00% | 88.50% | 82.50% |
| 2.0x | 100.00% | 88.50% | 85.00% |



#### Métricas de Eficiencia Computacional
- **Kairos Front-End MACs (continuos):** 61.440.000
- **Kairos Spikes Discretos (eventos):** 58.787
- **Kairos Acumulaciones (clasificador):** 31.744
- **Mini-Transformer MACs:** 1.711.072.000
- **Simple LSTM MACs:** 228.368.000
- **Memoria Heap Asignada:** 690.794.968 bytes

