# Kairos103: Clockless Rank-Order Neural Architecture with Strict Time-Warp Invariance

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Platform: .NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com/)
[![Application: Console CLI](https://img.shields.io/badge/Interface-CLI%20First%20(Scientific)-black.svg)](#)
[![Focus: Neuromorphic Computing](https://img.shields.io/badge/Focus-Event--Driven%20%7C%20Rank--Order-green.svg)](#)

> **Language / Idioma:** **English** | [Español (Spanish)](README_ES.md)
>
> [!NOTE]
> **Active Research Workshop / Kaggle Competition Sprint:**
> This repository is currently undergoing an intensive competitive engineering workshop against the Kaggle competition: *[NPPE2-T2-26 - ECG Heartbeat Arrhythmia Classification](https://www.kaggle.com/competitions/nppe-2-t-2-26-ecg-heartbeat-arrhythmia-classification)*. Experimental pipeline commands (`train-ecg`, `predict-kaggle`), threshold calibrations, and neuromorphic feature extractors are actively iterating. Once this workshop sprint concludes, all experimental components will be systematically assembled, decoupled, and consolidated into their final canonical production architecture.

---

## 1. Overview and Philosophy

**Kairos103** is an asynchronous (*clockless*) neural architecture developed in **C# / .NET 10**, engineered for sparse event-driven inference and computational efficiency under a **CLI-First / Headless** philosophy: a modular scientific console engine designed to guarantee high reproducibility, minimal memory contention from graphical interfaces, and controlled benchmarking.

In classical Greek philosophy:
* **Chronos** ($\chi\rho\acute{o}\nu o\varsigma$) represents metric, sequential time measured by a rigid clock ($\Delta t$).
* **Kairos** ($\kappa\alpha\iota\rho\acute{o}\varsigma$) represents the qualitative moment: relationship, precedence, opportunity, and the **relative order** in which events unfold.

Dominant contemporary deep learning architectures in their standard formulations—most notably vanilla **Transformers** (with rigid positional encodings $PE$) and standard **RNNs/LSTMs**—depend primarily on a *Chronos* framework: they discretize reality into uniform frames or time steps $\Delta t$. When an unaugmented signal accelerates, decelerates, or undergoes non-uniform temporal fluctuations (*time-warping*), these networks experience significant performance degradation unless retrained with computationally intensive data augmentation.

**Kairos103** operates under the *Kairos* paradigm: it dispenses with global clocks and absolute temporal coordinates. Information resides in the relative firing permutations ($n!$) among local neural subregions, achieving **strict mathematical invariance to global affine dilations** and executing inference via sparse event spikes without dense floating-point matrix multiplications.

---

## 2. Architectural Decision: Why CLI-First (Console)?

To scientifically validate the architecture against established baselines such as Transformers and LSTMs, a monolithic GUI application is deliberately eschewed in favor of a layered **Scientific Console** architecture:

1. **Rigorous Metric Isolation (Zero UI Overhead):**
   * Measuring inference latency in microsecond precision or heap allocations (`GC.GetAllocatedBytesForCurrentThread()`) in WPF or WinForms introduces statistical noise from the UI dispatcher, render loops, and XAML data bindings.
   * The headless console eliminates graphical dispatcher and render thread contention, ensuring that benchmark measurements reflect pure neural computation and core runtime execution without UI framework noise.
2. **Automated Massive Sweeps (Sweeps & CI):**
   * Enables automated, reproducible batch experiment execution via command-line arguments (evaluating across the 5 canonical time-warp scale factors: $0.5\times, 0.8\times, 1.0\times, 1.2\times, 2.0\times$, plus jitter sweeps).
3. **Automated Reporting and Plot Generation:**
   * The CLI automatically outputs confusion matrices, comparative curves (*Accuracy vs. Time-Warp Factor*), and Markdown tables ready for technical publication or repository display.
4. **Future Decoupling:**
   * By keeping core logic pure in `Kairos103.Core`, a graphical visualizer (WPF / Avalonia) can be added in the future as an optional consumer without touching the computational engine.

---

## 3. The Fundamental Problem of Transformers in Temporal Sequences

1. **Dependence on Rigid Sampling Frames ($\Delta t$):**
   Transformers convert time into spatial representations by projecting each sampled instant into a continuous vector and adding an index-dependent positional encoding vector ($PE_t$). If a signal is emitted at twice the speed ($2\times$) or half the speed ($0.5\times$), the indices shift drastically, and scaled dot-product attention ($QK^T / \sqrt{d_k}$) misaligns relative to the weights learned during nominal training.
2. **Quadratic Complexity and Inefficient Power Consumption:**
   Dense self-attention computes $O(T^2)$ relationships across the entire sequence length, recalculating attention even during static signal periods devoid of new information.
3. **Infeasibility on Edge Devices and Microcontrollers:**
   The memory footprint of multi-head attention and large embedding tables prevents deployment on low-power microcontrollers and autonomous edge sensors.

---

## 4. Scientific Hypothesis and What We Demonstrate

### 4.1. Global Affine Time-Scale Invariance Theorem
Let an ensemble of $n$ sensory neural channels produce their first action potentials (*first-spike arrival times*) at instants:
$$T = \{t_1, t_2, \dots, t_n\}$$

If the input signal undergoes a monotonic increasing affine temporal transformation (global acceleration, deceleration, or uniform delay):
$$t'_i = \alpha \cdot t_i + \beta, \quad \text{with } \alpha > 0, \; \beta \in \mathbb{R}$$

Then, the induced relative arrival order $\sigma \in S_n$ satisfies:
$$t_i < t_j \iff t'_i < t'_j \quad \forall i, j$$

$$\sigma(T') \equiv \sigma(T)$$

**Consequence:** In the arrival space of continuous first spikes, the internal rank-dictionary representation of Kairos103 is **analytically identical by construction** under any monotonic increasing affine dilation ($\alpha > 0$). In discrete digital signals with finite sampling, the phase-normalized front-end preserves between $90\%$ and $99\%$ permutation code concordance, exhibiting negligible degradation compared to clock-dependent models.

> [!NOTE]
> **Formal Theoretical Scope:** The theorem proves exact invariance for *global monotonic affine transformations* ($t' = \alpha t + \beta$). Non-uniform, local non-linear phase deformations (e.g., intra-burst warping) are not covered by the analytical theorem itself, but are empirically mitigated through modular $N=4$ subregions and lateral threshold inhibition.

### 4.2. Concrete Empirical Objectives to Demonstrate
1. **Global Affine Scaling Invariance ($0.5\times$ to $2.0\times$):**
   * On bounded temporal sequence classification tasks (sensory bursts, vibration, or ECG complexes delimited by their activation interval), Kairos103 maintains nominal accuracy within $\pm 2\%$ in moderate scaling ($0.8\times - 1.2\times$) and $\ge 90\%$ retention (worst-case drop $\le 10\%$) under extreme scaling ($0.5\times, 2.0\times$) without retraining or data augmentation, whereas baselines with rigid positional encodings like Transformers and LSTMs suffer drops exceeding $20\%-30\%$.
2. **Discrete Computational Footprint and MAC Breakdown:**
   * At the network and readout classifier stage, inference is resolved entirely through accumulated sums over sparse events ($\sim 10^4$ discrete integer additions without dense matrix multiplications).
   * Accounting for the complete end-to-end pipeline (including continuous front-end sensory filters), Kairos103 achieves an estimated $14.6\times$ (Synthetic) to $27.8\times$ (ECG) reduction in total arithmetic MAC/FLOP operations compared to the quadratic dense self-attention of an equivalent Transformer.
   * *Methodological Note on Energy:* Reported efficiency represents algorithmic arithmetic operation counts (MACs/FLOPs); direct physical energy (Joules/Watts) measurements on edge silicon remain a goal for dedicated hardware deployment phases.
3. **Jitter Tolerance via Modular Subregions:**
   * Partitioning into local modules of $N=4$ neurons with adaptive lateral inhibition filters high-frequency phase noise, maintaining code stability above $97\%$.

---

## 5. Processing Pipeline Architecture

```
 Continuous Signal / Event Stream (Sensory or Synthetic)
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 1. Front-End: Local Temporal Filters                    │
 │    (Burst Stimulation / Adaptive Zero-Crossing)         │
 └──────────────────────────┬──────────────────────────────┘
                            │ Asynchronous pulses
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 2. Modular Subregions N=4                               │
 │    - LIF Neurons with Commitment-Based Firing           │
 │    - Lateral Inhibition via Threshold Jumping           │
 └──────────────────────────┬──────────────────────────────┘
                            │ First spikes (t1 < t2 < t3 < t4)
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 3. Permutation Extractor (Rank Dictionary)              │
 │    - Normalized Rank Vector (1/rank)                    │
 │    - Local Combinatorial Capacity: 4! = 24              │
 └──────────────────────────┬──────────────────────────────┘
                            │ Sparse feature vector
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 4. Hierarchical Coincidence & Readout Layer             │
 │    - Relative Co-occurrence Detection                   │
 │    - Supervised Linear / Sparse Classifier              │
 └─────────────────────────────────────────────────────────┘
```

---

## 6. Solution Structure (.NET 10 Solution)

```text
Kairos103/
├── src/
│   ├── Kairos103.Core/               # Class Library: Pure UI-agnostic neural engine
│   │   ├── Neurons/
│   │   │   ├── LifNeuron.cs          # Leaky Integrate-and-Fire with dynamic threshold
│   │   │   └── ModularSubregion.cs   # N=4 neuron unit with lateral inhibition
│   │   ├── Encoding/
│   │   │   ├── RankOrderEncoder.cs   # Maps arrival times to rank vectors (1/rank)
│   │   │   └── CombinatorialTable.cs # Injective combinatorial mapping B103
│   │   └── Models/
│   │       ├── KairosNetwork.cs      # Modular asynchronous network graph
│   │       └── PairwiseTrainer.cs    # Supervised pairwise learning rule
│   │
│   ├── Kairos103.Benchmarks/         # Class Library: Comparison protocols and baselines
│   │   ├── Baselines/
│   │   │   ├── MiniTransformer.cs    # Multi-head self-attention baseline with PE
│   │   │   └── SimpleLstm.cs         # Classic recurrent baseline
│   │   ├── Generators/
│   │   │   ├── SyntheticSignals.cs   # Pattern generator with time-warp (0.5x - 2.0x)
│   │   │   └── NoiseInjector.cs      # Temporal jitter and drift injection
│   │   └── Metrics/
│   │       ├── EfficiencyTracker.cs  # Spike counter vs. MACs and heap allocation tracker
│   │       └── RobustnessMetrics.cs  # Accuracy degradation curves vs. time-warp
│   │
│   └── Kairos103.Cli/                # Console App: Main research entry point
│       ├── Program.cs                # CLI command router (benchmark, sweep, export)
│       ├── Commands/
│       │   ├── RunBenchmarkCommand.cs
│       │   ├── SweepNoiseCommand.cs
│       │   └── ExportReportCommand.cs
│       └── Reporting/
│           └── AsciiPlotter.cs       # Terminal ASCII charts and SVG/MD exporter
│
├── tests/
│   └── Kairos103.Tests/              # xUnit: Mathematical and injectivity unit tests
│       ├── InvarianceTests.cs        # Formal verification: sigma(t) == sigma(alpha * t + beta)
│       └── CombinatorialTests.cs     # B103 bound verification: sum(C(N,k)*2^k) >= 2^R
│
├── docs/
│   ├── MathTheory.md                 # Formal mathematical proof of permutation spaces
│   └── ExperimentalProtocol.md       # Step-by-step benchmark reproduction protocol
│
├── results/                          # Reproducible experimental results and vector figures
│   ├── BenchmarkSummary.md           # CLI-generated quantitative benchmark report
│   └── WarpRobustness.svg            # Accuracy vs. time-warp scale curves
│
├── Kairos103.slnx
├── README.md
├── README_ES.md
└── LICENSE
```

---

## 7. Command Line Interface (CLI Usage)

The console application provides direct commands for terminal reproducibility:

```bash
# === BLACK-BOX WORKFLOW (TRAIN, PREDICT, EVALUATE) ===

# 1. Train your model on any 1D time-series CSV and save to disk
dotnet run --project src/Kairos103.Cli -- train --data datasets/synthetic/warpstress_train.csv --output model.kairos --epochs 25

# 2. Predict and infer new signals in real-time using the saved model
dotnet run --project src/Kairos103.Cli -- predict --model model.kairos --data datasets/synthetic/warpstress_test_1.0x.csv --output predictions.csv

# 3. Detailed evaluation with Confusion Matrix and Latency
dotnet run --project src/Kairos103.Cli -- evaluate --model model.kairos --data datasets/synthetic/warpstress_test_2.0x.csv --export-json metrics.json

# === BENCHMARKS & COMPETITIVE LEADERBOARDS ===

# 4. Generate official submission package for leaderboards (Papers With Code / TinyML)
dotnet run --project src/Kairos103.Cli -- online-bench --output results

# 5. Run comparative benchmark against Transformer and LSTM
dotnet run --project src/Kairos103.Cli -- benchmark --fast

# 6. Run temporal jitter tolerance sweep
dotnet run --project src/Kairos103.Cli -- jitter-sweep --max-jitter-ms 50 --step 10
```

---

## 8. Experimental Protocol for Reproducibility

The entire benchmark suite is designed to run on standard hardware with the following configuration:
* **Runtime & SDK:** .NET 10.0 SDK (tested on version `10.0.101` across Windows, Linux, and macOS).
* **Reference Baselines:** Mini-Transformer and Simple LSTM are implemented natively in C# without data augmentation (*zero time-warp augmentation*) to test intrinsic architectural robustness.

1. **Phase 1: Nominal Training ($1.0\times$)**
   * Kairos103 and baselines (Mini-Transformer and LSTM) are trained on the dataset at nominal sampling rate.
2. **Phase 2: Time-Warp Stress Test Sweep**
   * Inference is evaluated across scale factors $\alpha \in \{0.5, 0.8, 1.0, 1.2, 2.0\}$.
   * Evaluated under progressive temporal Gaussian noise (*jitter* $\sigma_j \in [0, 50\,\text{ms}]$).
3. **Phase 3: Computational Footprint and Operation Accounting**
   * Strict event accounting: total additions per spike versus floating-point operations (FLOPs/MACs) of continuous models.
   * Heap memory allocation: reported batch memory reflects total managed GC churn in the CLI test harness, while per-sample inference memory footprint is in the low kilobyte range (<4 KB).
   * Direct physical energy telemetry (Joules/Watts) is deferred to dedicated neuromorphic silicon phases.

---

## 9. References and Foundations

* **Thorpe, S., Fize, D., & Marlot, C. (1996).** *Speed of processing in the human visual system.* Nature. (Foundations of the first-spike hypothesis).
* **Lora Sugve, E. (2026).** *Learning to Use Spike Order: Rank-Order Dictionaries Beat Binary Features by 17 Points at a Fraction of the Events.* SNN103 Technical Preprint.
* **B103 Theory (2026):** *Combinatorial State-Space Mapping with Semantic Activity Encodings.*

---

## 10. License and Commercial Licensing

This project is free and open-source software licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the `LICENSE` file for the full license text.

### Commercial Licensing / Dual Licensing
For enterprise, industrial, proprietary products, or commercial use cases requiring exemptions from the copyleft terms of the GNU GPL v3, custom commercial licenses are available directly from the author:

* **Author:** **Erisnel Lora Sugve**
* **Affiliation:** Independent Researcher, Santiago de Cuba, Cuba
* **Email:** [`lorasugve@gmail.com`](mailto:lorasugve@gmail.com)
* **ORCID:** [0009-0007-2047-9345](https://orcid.org/0009-0007-2047-9345)
