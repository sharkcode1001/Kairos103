# Experimental Protocol: Time-Warp Invariance (Kairos103 vs. Baselines)

> **Language / Idioma:** **English** | [Español (Spanish)](ExperimentalProtocol_ES.md)

This document formalizes the scientific benchmarking rules and conditions established prior to experimental execution.

---

## 1. Experimental Variables

* **Inference Temporal Scaling Factors ($Test$):**
  $$\alpha \in \{0.5, 0.8, 1.0, 1.2, 2.0\}$$
* **Noise Control:**
  * All test sets include an identical level of temporal *jitter* (Gaussian perturbation preserving temporal monotonicity) and Gaussian amplitude noise ($\sigma_a = 0.05$).
  * The test condition $\alpha = 1.0\times$ acts as a control group to isolate the effect of jitter from time dilation.
* **Training Condition ($Train$):**
  * Exclusively at nominal speed ($1.0\times$), without jitter and with slight residual amplitude noise ($\sigma_a = 0.03$).
  * **Zero data augmentation with time-warp:** Invariance must strictly emerge from the asynchronous rank-order encoding itself, rather than perturbation memorization.

---

## 2. Evaluation Metrics

For each model (Kairos103, Transformer, LSTM), across each speed factor and for each dataset:

1. **Classification Accuracy ($Accuracy$ vs. $\alpha$):**
   * Degradation curve under dilation and compression.
2. **Computational Cost per Decision:**
   * **Continuous Baselines (Transformer / LSTM):** Number of accumulated floating-point multiply-accumulate operations (MACs/FLOPs) per inference and resident heap memory.
   * **Kairos103:** Strict two-component breakdown:
     * *Sensory Front-End Transduction:* Continuous sampling MACs over local temporal windows.
     * *Network & Classifier:* Number of discrete events (first-spike pulses) + integer accumulation operations during readout (zero floating-point multiplications).
     * *Total Compute:* Sum of both stages benchmarked against the dense self-attention budget of the baseline.

---

## 3. Formal Success Criteria

To verify the experimental hypothesis, **all** of the following conditions must be satisfied:

1. **Kairos103 Global Affine Invariance:** Achieved accuracy remains within a $\pm 2\%$ margin relative to the nominal $1.0\times$ baseline under moderate temporal scaling ($0.8\times$ to $1.2\times$), and retains $\ge 90\%$ accuracy (worst-case drop $\le 10\%$) under extreme scaling ($0.5\times$ and $2.0\times$), across both datasets.
2. **Baseline Collapse:** Continuous models (Transformer or LSTM) suffer an accuracy drop $\ge 15\%$ at extreme factors ($0.5\times$ or $2.0\times$) compared to their performance at $1.0\times$.
3. **Operational / Energy Efficiency:** Kairos103 executes inference requiring $\le \frac{1}{10}$ (order of magnitude reduction) of the end-to-end compute per decision demanded by dense self-attention (Mini-Transformer).

> **Formal Scope:** Experiments strictly evaluate *Global Affine Time-Scale Invariance* ($t' = \alpha t + \beta, \, \alpha > 0$). Non-linear local deformations (e.g., intra-cycle piecewise warping) are classified as separate targets for future multidimensional extensions.


---

## 4. Stopping Rules and Scientific Integrity

* **Hypothesis Rejection:** If Kairos103 fails to maintain a substantially flat accuracy curve across temporal factors, the outcome will be reported transparently without cherry-picking or post-hoc hyperparameter tuning.
* **Nominal Quality Lower Bound:** If Kairos103 fails to achieve a minimum of $80\%$ accuracy under the nominal $1.0\times$ control condition, warp sweeps will be halted until the temporal encoding or modular subregion lateral inhibition scheme is refined.

---

## 5. Recurrent Baseline Dynamics Under Temporal Compression

In recurrent models (LSTM), temporal acceleration ($2.0\times$) halves the sequence length ($T \to T/2$). On physiological signals (ECG), this reduction shortens the temporal horizon over which hidden state and forget-gate error accumulate, occasionally producing a paradoxical accuracy boost (e.g., $73.50\% \to 85.00\%$). This is formally classified as a sequence-length truncation artifact rather than affine temporal robustness, as evidenced by catastrophic collapse under temporal dilation ($0.5\times$, where horizon lengthens) and phase distortion on synthetic signals.

