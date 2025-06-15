# Theoretical Foundations of the Compact Spectral Multiplier (CSM)

The Compact Spectral Multiplier (CSM) is a randomized feature map that approximates high-dimensional multilinear kernels via CountSketch and FFT, enabling efficient inner-product estimation with bounded variance. It is designed to replace expensive outer products in bilinear or polynomial mechanisms while preserving kernel expectations in case of highly correlated data.

---

## 1. Core Definitions and Notation

Let $x = (x_0, \dots, x_{K-1})$, with each $x_k \in \mathbb{R}^{d_k}$. Define:

* $d'$: common sketch dimension (shared across all modes)
* $K$: number of modes (tensor factors)

For each mode $k \in [K]$, generate random hash and sign functions:

$$
h_k : [d_k] \to [d'], \quad s_k : [d_k] \to \{-1, +1\},
$$

with 2-wise independence and uniform collision probability $\Pr(h_k(i) = h_k(j)) = 1/d'$.

The **CountSketch** of $x_k$ is:

$$
\text{CS}(x_k)[j] := \sum_{t: h_k(t)=j} s_k(t) x_k[t], \quad j \in [d'].
$$

We define the **unitary discrete Fourier transform**:

$$
\text{FFT}(u)[j] := \frac{1}{\sqrt{d'}} \sum_{t=0}^{d'-1} u[t] e^{-2\pi i jt / d'}.
$$

Define the **CSM feature map**:

$$
\Phi(x) := \text{IFFT} \left( \bigodot_{k=0}^{K-1} \text{FFT}(\text{CS}(x_k)) \right) \in \mathbb{C}^{d'}.
$$

The **inner product estimator** is:

$$
S(x, y) := \langle \Phi(x), \Phi(y) \rangle, \quad \widehat{Z}(x, y) := d'^{K-1} S(x, y).
$$

---

## 2. Expectation and Unbiasedness

**Theorem (Unbiased Estimator).** Under 2-wise independent hashes and signs:

$$
\mathbb{E}[\widehat{Z}(x, y)] = \prod_{k=0}^{K-1} \langle x_k, y_k \rangle =: Z(x, y).
$$

*Interpretation:* The CSM estimator is unbiased for the multilinear kernel formed by dot-products across modes, despite using low-dimensional projections.

---

## 3. Variance Behavior and Error Bound

Define similarity of each mode as:

$$
\rho_k := \frac{\langle x_k, y_k \rangle}{\|x_k\| \cdot \|y_k\|} \in [-1,1],
$$

and the relative variance term:

$$
\xi_k := \frac{c_t (1 - \rho_k^2)}{d' \rho_k^2}, \quad\text{with } c_t \le 2.
$$

**Lemma (Product Variance):**

$$
\text{Var}[\widehat{Z}] = Z^2 \left( \prod_{k=0}^{K-1}(1 + \xi_k) - 1 \right).
$$

**NRMSE Bound:**

$$
\mathrm{NRMSE} := \frac{\sqrt{\mathbb{E}[(\widehat{Z} - Z)^2]}}{|Z|} \lesssim \frac{c_t}{2d'} \sum_{k=0}^{K-1} \frac{1 - \rho_k^2}{\rho_k^2}.
$$

This explains the **variance explosion** when input modes have moderate or low similarity (e.g. $\rho_k = 0.5$).

---

## 4. Summary

The Compact Spectral Multiplier:

* Combines CountSketch + FFT across $K$ modes.
* Produces an unbiased kernel estimate via rescaled inner product: $\widehat{Z} = d'^{K-1} \langle \Phi(x), \Phi(y) \rangle$.
* Accumulates variance multiplicatively across modes, with magnitude governed by $\rho_k$ and $d'$.

CSM is most effective when the input mode similarities $\rho_k$ are moderately high. In low-similarity regimes ($\rho_k \ll 1$), the relative variance $\xi_k$ becomes large and may dominate the estimation error. In such cases, increasing $d'$ or falling back to exact dot-product computation may be necessary.

---
