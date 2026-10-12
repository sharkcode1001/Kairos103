# Mathematical Foundations: Permutation Spaces and Affine Invariance

> **Language / Idioma:** **English** | [Español (Spanish)](MathTheory_ES.md)

## 1. The Time-Warp Invariance Theorem

Let an asynchronous dynamic system consist of $n$ sensory neural channels. For a given stimulus pattern $S(t)$, we define the set of first action potential arrival times (*first-spike arrival times*) as:

$$T = \{t_1, t_2, \dots, t_n\}, \quad t_i \in \mathbb{R}^+$$

### 1.1. Monotonic Increasing Affine Transformation (Time-Warping)
Consider a global temporal distortion where the signal accelerates, decelerates, or shifts according to:

$$t'_i = g(t_i) = \alpha \cdot t_i + \beta, \quad \alpha > 0, \; \beta \in \mathbb{R}$$

Because the derivative of $g$ is strictly positive:
$$\frac{dg}{dt} = \alpha > 0$$

The function $g$ is a strictly order-preserving bijection on $\mathbb{R}$. Consequently:

$$t_i < t_j \iff \alpha \cdot t_i + \beta < \alpha \cdot t_j + \beta \iff t'_i < t'_j \quad \forall i, j \in \{1, \dots, n\}$$

### 1.2. Invariance of the Induced Permutation
We define the firing permutation $\sigma \in S_n$ as the bijection ordering the channels by arrival time:

$$t_{\sigma(1)} \le t_{\sigma(2)} \le \dots \le t_{\sigma(n)}$$

Since the relative order remains identical for any $\alpha > 0$ and $\beta \in \mathbb{R}$:

$$\sigma(T') \equiv \sigma(T)$$

The discrete representation extracted by the rank dictionary is **formally and identically invariant to temporal compression or dilation factor $\alpha$**.

### 1.3. Bounded-Epoch Domain Condition
The theorem is strictly valid over sensory bursts or bounded epochs where the physical event is delimited by an activation interval $[t_{\text{start}}, t_{\text{end}}]$. The front-end projection maps the temporal support of the signal onto a dimensionless phase coordinate $\tau = \frac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}} \in [0, 1]$. Affine invariance manifests in its purest form across any signal where internal morphology and causal precedence determine class membership, regardless of event emission speed.

---

## 2. Modular Partitioning into $N=4$ Subregions

The full combinatorial permutation space for $n$ channels is $|S_n| = n!$.
For large $n$, $n!$ exhibits severe combinatorial explosion, and susceptibility to high-frequency *jitter* escalates due to collisions among temporally close channels.

Kairos103 decomposes the space into $M$ independent local modular subregions of fixed size $N = 4$:

* **Local capacity:** Each subregion produces $4! = 24$ possible first-spike configurations.
* **Cartesian product capacity:** For $M$ modules:
  $$|\Omega| = (4!)^M = 24^M$$
* **Jitter robustness:** Locally coupled lateral inhibition within each quartet ensures that only neurons with dominant activation fix the firing order, filtering phase noise with over $97\%$ stability.
