# Informe Técnico Consolidado de Benchmarking: Kairos103

* **Fecha:** 2026-10-12 01:27:16 UTC
* **Plataforma:** .NET 10.0 (C# asíncrono, Headless CLI)
* **Arquitectura:** Red Rank-Order con Subregiones N=4 acopladas

## 1. Resumen de Invarianza a Time-Warping
Los resultados experimentales confirman la estabilidad ante transformaciones temporales afines ($t' = \alpha t + \beta$).

### A. Dataset: SYNTHETIC

### Resultados de Benchmark: synthetic

| Factor Warp | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| 0.5x | 99.00% | 31.00% | 30.00% |
| 0.8x | 98.00% | 39.50% | 38.50% |
| 1.0x | 98.50% | 53.00% | 98.50% |
| 1.2x | 98.00% | 25.00% | 0.00% |
| 2.0x | 90.00% | 24.50% | 0.50% |



#### Métricas de Eficiencia Computacional
- **Kairos Front-End MACs (continuos):** 61.440.000
- **Kairos Spikes Discretos (eventos):** 63.999
- **Kairos Acumulaciones (clasificador):** 64.000
- **Mini-Transformer MACs:** 897.504.000
- **Simple LSTM MACs:** 162.512.000
- **Memoria Heap Asignada:** 449.242.880 bytes

### A. Dataset: ECG

### Resultados de Benchmark: ecg

| Factor Warp | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| 0.5x | 100.00% | 92.50% | 56.50% |
| 0.8x | 100.00% | 95.50% | 74.00% |
| 1.0x | 100.00% | 96.00% | 73.50% |
| 1.2x | 100.00% | 74.00% | 82.50% |
| 2.0x | 100.00% | 68.50% | 85.00% |



#### Métricas de Eficiencia Computacional
- **Kairos Front-End MACs (continuos):** 61.440.000
- **Kairos Spikes Discretos (eventos):** 58.787
- **Kairos Acumulaciones (clasificador):** 31.744
- **Mini-Transformer MACs:** 1.711.040.000
- **Simple LSTM MACs:** 228.368.000
- **Memoria Heap Asignada:** 690.666.968 bytes

