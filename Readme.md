# Kairos103: Clockless Rank-Order Neural Architecture with Strict Time-Warp Invariance

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Platform: .NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com/)
[![Application: Console CLI](https://img.shields.io/badge/Interface-CLI%20First%20(Scientific)-black.svg)](#)
[![Focus: Neuromorphic Computing](https://img.shields.io/badge/Focus-Event--Driven%20%7C%20Rank--Order-green.svg)](#)

> **Language / Idioma:** **English** | [Español (Spanish)](README_ES.md)

---

## 1. Overview and Philosophy

**Kairos103** is an ultra-low-power, asynchronous (*clockless*) neural architecture developed in **C# / .NET 10**, designed under a **CLI-First / Headless** philosophy: a modular scientific console engine engineered to guarantee total reproducibility, zero memory contamination from graphical interfaces, and maximum benchmarking precision.

In classical Greek philosophy:
* **Chronos** ($\chi\rho\acute{o}\nu o\varsigma$) represents metric, sequential time measured by a rigid clock ($\Delta t$).
* **Kairos** ($\kappa\alpha\iota\rho\acute{o}\varsigma$) represents the qualitative moment: relationship, precedence, opportunity, and the **relative order** in which events unfold.

Dominant contemporary deep learning architectures—most notably **Transformers** (with rigid positional encodings $PE$) and **RNNs/LSTMs**—depend strictly on a *Chronos* framework: they discretize reality into uniform frames or time steps $\Delta t$. When a signal accelerates, decelerates, or undergoes non-uniform temporal fluctuations (*time-warping*), these networks collapse or require computationally prohibitive retraining and data augmentation.

**Kairos103** operates under the *Kairos* paradigm: it dispenses with global clocks and absolute temporal coordinates. Information resides in the relative firing permutations ($n!$) among local neural subregions, achieving **strict mathematical invariance to temporal dilations** and executing inference via sparse event spikes without dense floating-point matrix multiplications.

---

## 2. Architectural Decision: Why CLI-First (Console)?

To scientifically validate the architecture against established baselines such as Transformers and LSTMs, a monolithic GUI application is deliberately eschewed in favor of a layered **Scientific Console** architecture:

1. **Rigorous Metric Isolation (Zero UI Overhead):**
   * Measuring inference latency in microsecond precision or heap allocations (`GC.GetAllocatedBytesForCurrentThread()`) in WPF or WinForms introduces statistical noise from the UI dispatcher, render loops, and XAML data bindings.
   * The headless console guarantees that 100% of measured CPU cycles and memory bytes belong exclusively to the neural network.
2. **Automated Massive Sweeps (Sweeps & CI):**
   * Enables automated, reproducible batch experiment execution via command-line arguments (e.g., 10 random seeds $\times$ 7 time-warp scale factors $\times$ 5 noise levels).
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

### 4.1. Time-Warp Invariance Theorem
Let an ensemble of $n$ sensory neural channels produce their first action potentials (*first-spike arrival times*) at instants:
$$T = \{t_1, t_2, \dots, t_n\}$$

If the input signal undergoes a monotonic increasing affine temporal transformation (global acceleration, deceleration, or uniform delay):
$$t'_i = \alpha \cdot t_i + \beta, \quad \text{with } \alpha > 0, \; \beta \in \mathbb{R}$$

Then, the induced relative arrival order $\sigma \in S_n$ satisfies:
$$t_i < t_j \iff t'_i < t'_j \quad \forall i, j$$

$$\sigma(T') \equiv \sigma(T)$$

**Consequence:** In the arrival space of continuous first spikes, the internal rank-dictionary representation of Kairos103 is **analytically identical by construction** under any monotonic increasing affine dilation ($\alpha > 0$). In discrete digital signals with finite sampling, the phase-normalized front-end preserves between $90\%$ and $99\%$ permutation code concordance, exhibiting negligible degradation compared to clock-dependent models.

### 4.2. Concrete Empirical Objectives to Demonstrate
1. **Time-Warp Invariance ($0.5\times$ to $2.0\times$) in Bounded Epochs:**
   * On bounded temporal sequence classification tasks (sensory bursts, vibration, or ECG complexes delimited by their activation interval), Kairos103 maintains accuracy degradation under $2\%$ without retraining or data augmentation, whereas baselines with rigid positional encodings like Transformers and LSTMs suffer drops exceeding $20\%-30\%$.
2. **Discrete Computational Efficiency and MAC Breakdown:**
   * At the network and readout classifier stage, inference is resolved entirely through accumulated sums over sparse events ($\sim 10^4$ discrete integer additions without dense multiplications).
   * Accounting for the complete pipeline (including the continuous front-end sensory analog/digital filters), Kairos103 reduces total compute by over an order of magnitude ($\sim 14.6\times$ fewer global MAC/FLOP operations compared to the quadratic dense self-attention $O(T^2 \cdot D)$ of an equivalent Transformer).
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
# 1. Run quick comparative benchmark (Kairos103 vs Transformer vs LSTM)
dotnet run --project src/Kairos103.Cli -- benchmark --fast

# 2. Run full benchmark across all datasets and scale factors
dotnet run --project src/Kairos103.Cli -- benchmark --warp-factors 0.5,0.8,1.0,1.2,2.0 --dataset synthetic

# 3. Run temporal jitter tolerance sweep
dotnet run --project src/Kairos103.Cli -- jitter-sweep --max-jitter-ms 50 --step 10

# 4. Export technical report and vector plots (Markdown + SVG to ./results)
dotnet run --project src/Kairos103.Cli -- export-plots --output ./results/
```

---

## 8. Experimental Protocol for Reproducibility

The entire benchmark suite is designed to run on any standard developer machine with the .NET 10 SDK installed:

1. **Phase 1: Nominal Training ($1.0\times$)**
   * Kairos103 and baselines (Mini-Transformer and LSTM) are trained on the dataset at nominal sampling rate.
2. **Phase 2: Time-Warp Stress Test Sweep**
   * Inference is evaluated across scale factors $\alpha \in \{0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0\}$.
   * Evaluated under progressive temporal Gaussian noise (*jitter* $\sigma_j \in [0, 50\,\text{ms}]$).
3. **Phase 3: Energy and Computational Footprint Measurement**
   * Strict event accounting: total additions per spike versus floating-point operations (FLOPs/MACs) of continuous models.
   * Heap memory allocation during steady-state inference (`GC.GetAllocatedBytesForCurrentThread()`).

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
