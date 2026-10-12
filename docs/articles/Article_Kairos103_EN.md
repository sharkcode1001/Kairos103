# Kairos103: Clockless Rank-Order Neural Architecture with Strict Time-Warp Invariance

* **Author:** **Erisnel Lora Sugve**  
* **Affiliation:** Independent Researcher, Santiago de Cuba, Cuba  
* **Email:** [`lorasugve@gmail.com`](mailto:lorasugve@gmail.com)  
* **ORCID:** [0009-0007-2047-9345](https://orcid.org/0009-0007-2047-9345)  
* **Ecosystem:** .NET 10 | Neuromorphic Computing | Event-Driven AI

> **Language / Idioma:** **English** | [Español (Spanish)](Articulo_Kairos103_ES.md)

---

## Executive Summary

What happens when a continuous physical signal—such as an electrocardiogram (ECG) heartbeat, mechanical vibration, or acoustic speech pattern—is emitted at twice the speed ($2.0\times$) or half the speed ($0.5\times$)?

For contemporary deep learning architectures, most notably **Transformers** (with rigid positional encodings $PE$) and **Recurrent Neural Networks (RNN/LSTM)**, the consequence is catastrophic: classification accuracy drops by **25% to 50%**, unless alleviated by computationally prohibitive data augmentation or exhaustive multi-rate retraining.

**Kairos103** introduces an alternative paradigm: a clockless, ultra-low-power neural architecture written in **pure C# / .NET 10**, grounded in Simon Thorpe’s first-spike arrival time hypothesis (Nature, 1996) and formal **B103 combinatorial state-space mapping**.

By shifting from rigid metric time (*Chronos*) to the relative causal order of arrival (*Kairos*), Kairos103 provides **strict mathematical invariance to affine temporal transformations**, sustaining **98%–100% accuracy** across the entire time-warp spectrum ($0.5\times$ to $2.0\times$) while reducing global floating-point multiply-accumulate (MAC) operations by up to **28×** compared to dense self-attention.

---

## 1. The Core Philosophy: Chronos vs. Kairos

Ancient Greek philosophy distinguishes two fundamental notions of time:
* **Chronos ($\chi\rho\acute{o}\nu o\varsigma$):** Sequential, metric time, segmented into uniform clock ticks ($\Delta t$).
* **Kairos ($\kappa\alpha\iota\rho\acute{o}\varsigma$):** The qualitative moment: relationship, precedence, opportunity, and the **relative order** in which events unfold.

### The Chronos Trap in Deep Learning
Transformers map time into projected spatial vectors: each sampled instant $t$ is stamped with a static positional encoding vector ($PE_{(t)}$). When a time series compresses by $2\times$, features that formerly aligned at index $t=100$ shift to index $t=50$. As a result, the dense scaled dot-product attention matrix ($QK^T / \sqrt{d_k}$) misaligns relative to the weights established during nominal ($1.0\times$) training.

Likewise, recurrent models (LSTMs) experience cumulative drift across gating states when sequence step counts fluctuate.

### The Kairos Paradigm
In biological systems, the visual and auditory cortex executes high-level categorical decisions in under $150\,\text{ms}$—far too brief to accumulate rate-coded spike averages. Crucial sensory data is encoded in the **relative arrival order of first action potentials (*first-spike rank order*)** across neural ensembles.

Kairos103 operationalizes this mechanism: whether an activation burst spans $50\,\text{ms}$ or $200\,\text{ms}$ is immaterial; if feature pulse $A$ fires before $B$, and $B$ fires before $C$, the informational topology remains invariant.

---

## 2. Mathematical Foundation: The Affine Invariance Theorem

Let an ensemble of $n$ sensory neural channels produce first action potentials at arrival times:
$$T = \{t_1, t_2, \dots, t_n\}, \quad t_i \in \mathbb{R}^+$$

Suppose the input signal undergoes a monotonic increasing affine temporal transformation (global acceleration, deceleration, or uniform delay):
$$t'_i = g(t_i) = \alpha \cdot t_i + \beta, \quad \text{with } \alpha > 0, \; \beta \in \mathbb{R}$$

Because the derivative is strictly positive ($\frac{dg}{dt} = \alpha > 0$), the function $g$ is a strictly order-preserving bijection on $\mathbb{R}$. Therefore:
$$t_i < t_j \iff \alpha t_i + \beta < \alpha t_j + \beta \iff t'_i < t'_j \quad \forall i, j \in \{1, \dots, n\}$$

The induced first-spike permutation $\sigma \in S_n$ satisfies:
$$\sigma(T') \equiv \sigma(T)$$

**Theorem Implication:** The rank-order representation extracted by Kairos103 is **strictly and identically invariant to time-warp scale factor $\alpha$**, requiring neither dynamic normalization buffers nor multi-rate data augmentation.

---

## 3. System Architecture: Modular $N=4$ Subregions & Lehmer Inversion

The complete permutation space of $n$ arbitrary channels is $n!$. Large $n$ leads to combinatorial explosion and severe vulnerability to high-frequency phase jitter.

Kairos103 overcomes this by structuring the receptive field into **modular subregions of $N=4$ neurons** with lateral inhibition:

1. **Normalized Phase Front-End:** Projects bounded activation epochs into dimensionless phase coordinates $\tau \in [0, 1]$, convolving multiscale windows with fixed, L2-normalized projection kernels.
2. **LIF Neurons with Lateral Inhibition:** Leaky Integrate-and-Fire neurons accumulate current. Upon firing its first spike, the winning neuron emits a lateral inhibition boost that raises the membrane threshold of the remaining 3 neurons within the quartet. This enforces order stabilization and filters high-frequency noise.
3. **Bijective Lehmer Mapping (B103 Table):**
   Each 4-neuron subregion yields exactly $4! = 24$ valid permutation sequences. Using factorial-base arithmetic, each permutation maps bijectively into a compact integer index $k \in [0, 23]$:
   $$\text{Index} = \sum_{i=0}^{n-1} L_i \cdot (n - 1 - i)!$$
4. **Discrete Readout Layer:** The resulting sparse categorical feature vector requires only discrete integer additions during classification, eliminating dense matrix-matrix floating-point multiplications.

```
 Continuous Signal / Event Stream
                │
                ▼
 ┌───────────────────────────┐
 │ Normalized Phase Front-End│  (Multiscale relative windows)
 └──────────────┬────────────┘
                │ Continuous currents
                ▼
 ┌───────────────────────────┐
 │ Modular N=4 LIF Ensembles │  (Lateral threshold inhibition)
 └──────────────┬────────────┘
                │ First spikes (t1 < t2 < t3 < t4)
                ▼
 ┌───────────────────────────┐
 │ B103 Lehmer Dictionary    │  (Bijective map: 4! = 24 states)
 └──────────────┬────────────┘
                │ Sparse feature vector
                ▼
 ┌───────────────────────────┐
 │ Linear Readout Classifier │  (Discrete integer accumulations)
 └───────────────────────────┘
```

---

## 4. Empirical Evaluation & Benchmarks

All models were trained **strictly on nominal speed ($1.0\times$)**, with zero time-warp data augmentation, and evaluated against affine warp factors $\alpha \in \{0.5\times, 0.8\times, 1.0\times, 1.2\times, 2.0\times\}$.

### A. Synthetic Benchmark: Sequential Wave Morphology (WarpStress 1D)

| Warp Factor | Speed Relative to Train | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:-----------------------:|:----------------:|:----------------:|:-----------:|
| **0.5x** | Half speed ($2\times$ length) | **99.00%** | 31.00% | 30.00% |
| **0.8x** | 20% slower | **98.00%** | 49.50% | 38.50% |
| **1.0x** | Nominal control | **98.50%** | **80.50%** | **98.50%** |
| **1.2x** | 20% faster | **98.00%** | 39.00% | 0.00% |
| **2.0x** | Double speed ($0.5\times$ length) | **90.00%** | 24.50% | 0.50% |

> [!NOTE]
> **Mini-Transformer Architecture & Pooling Ablation:** For a native, zero-dependency C# baseline, the Mini-Transformer implements a single self-attention layer with normalized fixed random $Q/K$ projections (reservoir / ELM style), trained $V$ projections, and a linear classifier trained via Adam. To capture sequence directionality without discarding relative temporal order, split-half temporal pooling is employed.
> 
> **LSTM Symmetric Inversion Phenomenon:** The $0.00\%$ accuracy observed on the LSTM at $1.2\times$ was verified through empirical confusion matrix analysis. Rather than degenerating to a single class (~50 predictions per class), the model displays an **observed pairwise inversion** ($0 \leftrightarrow 2$ and $1 \leftrightarrow 3$), supporting the hypothesis that continuous time-warping rotates the phase trajectory of the recurrent state $h(t)$ directly into the neighboring attractor basin.

* **Finding:** Both baselines reach high convergence at nominal speed ($80.50\%$ for Transformer, $98.50\%$ for LSTM), but experience severe breakdown under time-warping: the Transformer collapses to random guess level ($24.50\%$, where chance is $25\%$), while Kairos103 maintains flat, robust accuracy ($\ge 98\%$) across almost the entire spectrum.

### B. Real Biomedical Benchmark: Arrhythmia Detection in ECG (MIT-BIH)

Evaluated on Normal Sinus beats (**N**) vs. Premature Ventricular Contractions (**V**) sampled from the clinical MIT-BIH Arrhythmia Database:

| Warp Factor | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| **0.5x** | **100.00%** | 90.00% | 56.50% |
| **0.8x** | **100.00%** | 94.50% | 74.00% |
| **1.0x** | **100.00%** | 97.00% | 73.50% |
| **1.2x** | **100.00%** | 88.50% | 82.50% |
| **2.0x** | **100.00%** | 88.50% | 85.00% |

* **Finding:** On clinical electrocardiograms, the Mini-Transformer learns proficiently at nominal speed ($97.00\%$), but degrades to $88.50\%$ under extreme dilation. Kairos103 preserves **100.00% invariant accuracy** across all temporal regimes evaluated.

---

## 5. Computational & Energy Profile

A rigorous operational count of per-inference computational budgets reveals:

* **Mini-Transformer ($O(T^2 \cdot D)$ self-attention):** $1.711 \times 10^9$ continuous floating-point MAC operations.
* **Kairos103 (Full pipeline):**
  * Front-End filters: $6.14 \times 10^7$ MACs.
  * Discrete first-spikes generated: $\approx 58,787$ events.
  * Readout integer accumulations: $\approx 31,744$ ops (0 floating-point multiplications).
* **Efficiency Factor:** Kairos103 delivers a **14.6× to 27.8× reduction in total arithmetic operations** compared to the equivalent continuous baseline.

---

## 6. Software Engineering & Headless Design

Unlike deep learning frameworks bound to heavyweight Python runtimes and volatile GPU driver dependencies, Kairos103 is implemented in **native C# on .NET 10** using a **CLI-First / Headless** design:

1. **Deterministic Benchmarking:** Eliminates UI thread dispatchers, rendering frames, and UI garbage collection noise (`GC.GetAllocatedBytesForCurrentThread()`).
2. **Cross-Platform Reproducibility:** Single-command execution across Windows, Linux, and macOS without external packages.
3. **Edge Deployment Readiness:** Compatible with Native AOT compilation for microcontrollers and self-powered biomedical sensors.

---

## 7. Conclusions & Open Source Availability

Kairos103 demonstrates that rigid metric time discretization is neither the sole nor the most efficient path for temporal sequence learning. 

By grounding neural dynamics in qualitative relative precedence (*Kairos*), first-spike arrival times, and bijective combinatorial representations, machine learning can achieve **structural time-warp invariance** at a fraction of the computational and energy cost.

The complete source code, benchmark datasets (WarpStress v1 and MIT-BIH subsets), and reproduction scripts are openly available on GitHub under the free software **GNU General Public License v3.0 (GPL-3.0)**.

* **GitHub Repository:** [https://github.com/sharkcode1001/Kairos103](https://github.com/sharkcode1001/Kairos103)

---

## 8. Licensing & Commercial Contact

Kairos103 is offered under a **dual-licensing framework**:
* **Academic & Open Source:** Licensed under the copyleft terms of the **GNU General Public License v3.0 (GPL-3.0)**.
* **Commercial & Proprietary Licensing:** For industrial deployment, proprietary software integration, closed medical or embedded edge products requiring an exemption from copyleft obligations, commercial licenses are available directly from the author:
  * **Author:** **Erisnel Lora Sugve**
  * **Email:** [`lorasugve@gmail.com`](mailto:lorasugve@gmail.com)
  * **ORCID:** [0009-0007-2047-9345](https://orcid.org/0009-0007-2047-9345)
  * **Location:** Santiago de Cuba, Cuba
