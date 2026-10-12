# Informe Técnico de Benchmarking: Kairos103

* **Fecha:** 2026-10-12 01:07:04 UTC
* **Plataforma:** .NET 10.0 (C# asíncrono, Headless CLI)
* **Arquitectura:** Red Rank-Order con Subregiones N=4 acopladas
* **Dataset Evaluado:** SYNTHETIC

## 1. Resumen de Invarianza a Time-Warping
Los resultados experimentales confirman la estabilidad ante transformaciones temporales afines ($t' = \alpha t + \beta$).

### Resultados de Benchmark: synthetic

| Factor Warp | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| 0.5x | 99.00% | 31.00% | 40.50% |
| 0.8x | 98.00% | 59.00% | 36.50% |
| 1.0x | 98.50% | 53.50% | 41.00% |
| 1.2x | 98.00% | 34.50% | 40.00% |
| 2.0x | 90.00% | 24.50% | 43.50% |


## 2. Métricas de Eficiencia Computacional
- **Kairos Front-End MACs (continuos):** 61.440.000
- **Kairos Spikes Discretos (eventos):** 63.999
- **Kairos Acumulaciones (clasificador):** 64.000
- **Mini-Transformer MACs:** 897.504.000
- **Simple LSTM MACs:** 162.512.000
- **Memoria Heap Asignada:** 449.233.528 bytes
