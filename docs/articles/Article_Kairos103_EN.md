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

By shifting from rigid metric time (*Chronos*) to the relative causal order of arrival (*Kairos*), Kairos103 provides **strict mathematical invariance of spike ordering under affine temporal transformations** alongside high end-to-end empirical robustness, sustaining **98%–100% accuracy** across moderate time-warp scaling ($0.5\times$ to $1.2\times$) and retaining **91.62%** under extreme compression ($2.0\times$), while reducing global floating-point multiply-accumulate (MAC) operations by up to **28×** compared to dense self-attention without requiring time-warp data augmentation.

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

## 2. Mathematical Foundation & Methodological Dissection of Invariance

To evaluate temporal invariance with scientific rigor, the system must be analyzed across three distinct levels:

1. **Strict Mathematical Invariance of Rank-Order Encoding:** Formally proven for continuous arrival timestamps.
2. **Physical Front-End Stability & Discretization Dynamics:** Effects of discrete sampling ($\Delta t$), finite windowing, and membrane leak decay ($\tau_{\text{leak}}$).
3. **End-to-End Classifier Generalization:** Empirical accuracy achieved by the linear combinatorial readout on synthetic and clinical benchmarks.

### 2.1. The Affine Invariance Theorem of First-Spike Order
Let an ensemble of $n$ sensory neural channels produce first action potentials at arrival times:
$$T = \{t_1, t_2, \dots, t_n\}, \quad t_i \in \mathbb{R}^+$$

Suppose the input signal undergoes a monotonic increasing affine temporal transformation (global acceleration, deceleration, or uniform delay):
$$t'_i = g(t_i) = \alpha \cdot t_i + \beta, \quad \text{with } \alpha > 0, \; \beta \in \mathbb{R}$$

Because the derivative is strictly positive ($\frac{dg}{dt} = \alpha > 0$), the function $g$ is a strictly order-preserving bijection on $\mathbb{R}$. Therefore:
$$t_i < t_j \iff \alpha t_i + \beta < \alpha t_j + \beta \iff t'_i < t'_j \quad \forall i, j \in \{1, \dots, n\}$$

The induced first-spike permutation $\sigma \in S_n$ satisfies:
$$\sigma(T') \equiv \sigma(T)$$

**Theorem Implication:** In the idealized space of continuous first-spike timestamps, the rank-order representation extracted by Kairos103 is **analytically identical by construction**.

### 2.2. Front-End Sensory Transduction and Discretization Boundaries
In real-world digital series, signals are sampled at finite intervals $\Delta t$. Moreover, leaky integrate-and-fire (LIF) units continuously decay membrane voltage with a passive leakage constant $\tau_{\text{leak}}$.
When an unaugmented signal undergoes extreme compression ($\alpha = 2.0\times$), the effective frequencies double relative to the fixed leak constant $\tau_{\text{leak}}$, and finite window boundaries may clip subtle tail events. This boundary effect accounts transparently for why synthetic accuracy at $2.0\times$ measures **$91.62\%$** rather than a theoretical $100\%$, while still decisively outperforming continuous baselines that experience complete breakdown.

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
| **0.5x** | Half speed ($2\times$ length) | **99.00%** | 45.38% | 75.62% |
| **0.8x** | 20% slower | **98.88%** | 63.62% | 97.38% |
| **1.0x** | Nominal control | **98.75%** | **99.88%** | **99.88%** |
| **1.2x** | 20% faster | **98.38%** | 45.62% | 99.38% |
| **2.0x** | Double speed ($0.5\times$ length) | **91.62%** | 46.38% | 51.75% |

> [!NOTE]
> **Methodological Role of Baselines:** To ensure autonomous, zero-dependency reproducibility in .NET without external Python or GPU overhead, the Mini-Transformer and Simple LSTM serve as **canonical native reference baselines (zero time-warp augmentation)**. Both achieve near-flawless nominal convergence ($99.88\%$), confirming robust representation learning on standard inputs. Their sharp degradation under scale variation ($45\%–51\%$) highlights the structural fragility of clock-dependent coordinate representations (*Chronos*) compared to causal rank ordering (*Kairos*).

* **Finding:** Kairos103 maintains flat, robust accuracy ($\ge 98.3\%$) across the operational range $0.5\times - 1.2\times$, and retains **$91.62\%$** under extreme $2.0\times$ temporal compression, outperforming continuous baselines by over 40 percentage points.

### B. Real Biomedical Benchmark: Arrhythmia Detection in ECG (MIT-BIH)

Evaluated on Normal Sinus beats (**N**) vs. Premature Ventricular Contractions (**V**) sampled from the clinical MIT-BIH Arrhythmia Database:

| Warp Factor | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| **0.5x** | **100.00%** | 92.00% | 66.25% |
| **0.8x** | **99.75%** | 95.25% | 64.00% |
| **1.0x** | **100.00%** | 97.25% | 65.50% |
| **1.2x** | **100.00%** | 84.50% | 62.50% |
| **2.0x** | **100.00%** | 91.25% | 65.50% |

* **Finding:** On clinical electrocardiograms, the Mini-Transformer learns proficiently at nominal speed ($97.25\%$), but degrades to $84.50\% - 91.25\%$ under temporal compression. Kairos103 preserves near-perfect retention ($\ge 99.75\%$) across all scale regimes.

---

## 5. Computational & Energy Profile

A rigorous operational count of per-inference computational budgets reveals:

* **Mini-Transformer ($O(T^2 \cdot D)$ self-attention):** $3.42 \times 10^9 - 3.59 \times 10^9$ continuous floating-point MAC operations across the full evaluation suite.
* **Simple LSTM:** $4.37 \times 10^8 - 6.22 \times 10^8$ continuous MACs.
* **Kairos103 (Full pipeline):**
  * Front-End filters: $1.22 \times 10^8 - 2.45 \times 10^8$ continuous MACs.
  * Discrete first-spikes generated: $\approx 117,881 - 255,994$ events.
  * Readout integer accumulations: $\approx 63,476 - 256,000$ operations (0 floating-point multiplications).
* **Efficiency Factor:** Kairos103 delivers a **14.6× to 27.8× reduction in total arithmetic operations** compared to the equivalent continuous baseline.

> [!TIP]
> **Notes on Energy and Memory Metrics:** 
> 1. Arithmetic metrics denote **algorithmic operational counts** (MACs vs. discrete integer accumulations). Direct physical energy dissipation (Joules/Watts) is scheduled for physical silicon telemetry phases on dedicated neuromorphic chips (e.g., Intel Loihi or asynchronous FPGAs).
> 2. Console-reported heap allocations reflect cumulative **garbage collection (GC) churn** throughout massive sequential batch evaluations in the .NET CLI test harness. Per-sample inference memory allocation remains in the low kilobyte range, suitable for embedded microcontrollers without dynamic heap managers.

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
