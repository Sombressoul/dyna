# CPSF — Appendix A: Ray–Trace Operator $\tau$

## A.0. Conventions and Dependencies (from Canon)

* Dimension: $N\ge 2$; domain $(\mathbb C/\mathbb Z)^N$ with unit periods componentwise.
* Ray: $\ell=(z,\vec d)$ with $z\in\mathbb C^N$ and $|\vec d|=1$.
* Contribution: $C_j=(\ell_j,\hat T_j,\sigma_j^{\parallel},\sigma_j^{\perp},\alpha_j)$ with $\ell_j=(z_j,\vec d_j)$, $\hat T_j\in\mathbb C^S$, $\sigma_j^{\parallel},\sigma_j^{\perp}>0$, $\alpha_j\in\mathbb R_{\ge 0}$.
* Lift $\mathrm{lift}(\cdot)$ and wrap $\mathrm{wrap}(\cdot)$: **wrap applies to the real part only**, mapping $\Re(\cdot)$ to $[-\tfrac12,\tfrac12)^N$ (nearest representative). The imaginary part is left unchanged.
* Directional term $\delta_{\vec d}(\vec d,\vec d_j)\in\mathbb C^N$: tangent to $\vec d_j$ and vanishes when $\vec d\parallel\vec d_j$; smooth in $(\vec d,\vec d_j)$.
* Frame $R(\vec d_j)\in U(N)$; its first column $b$ aligns to $\vec d_j$. Extended frame $R_{\mathrm{ext}}=\operatorname{diag}(R,R)\in U(2N)$.
* Embedding $\iota(u,v)=[u;v]\in\mathbb C^{2N}$.
* Covariance blocks (per-contribution $j$):
  $S_{0,j}\;=\;\sigma_j^{\perp} I_N + (\sigma_j^{\parallel}-\sigma_j^{\perp})\, b\,b^{\!*},\qquad \Sigma_j\;=\;\operatorname{diag}(S_{0,j},S_{0,j}).$
* Quadratic form and envelope: $q(w)=\langle \Sigma_j^{-1}w,w\rangle$, $\rho(q)=e^{-\pi q}$ . All inner products are Hermitian; batch broadcasting implicit.

## A.1. Torus–Periodic Spatial Kernel

Let $\Delta z:=\mathrm{lift}(z)-\mathrm{lift}(z_j)\in\mathbb C^N$ and $\delta:=\delta_{\vec d}(\vec d,\vec d_j)$. Define the **periodized spatial Gaussian** for contribution $j$:

$$
G_j^{\mathbb T}(\Delta z)\;:=\;\sum_{n\in\mathbb Z^N} \exp\!\big( -\pi\,(\Delta z+n)^{\!*} S_{0,j}^{-1} (\Delta z+n) \big),\quad S_{0,j}\succ0.
$$

Because $S_{0,j}\succ0$, the series is absolutely and uniformly convergent on compact sets; thus $G_j^{\mathbb T}$ is smooth and $1$‑periodic in $\Re(\Delta z)$.

## A.2. Definition of the Ray–Trace Operator

Given $\mathcal C=\{C_j\}$ and a query ray $\ell=(z,\vec d)$, define the **exact torus field**

$$
\boxed{\;\tau(\ell\mid\mathcal C)\;=\;\sum_j \alpha_j\,\eta_j(\ell)\,\hat T_j\;\in\mathbb C^S\;}
$$

Assume that $\mathcal C$ is finite or that the series $\sum_j \alpha_j \cdot \eta_j(\ell) \cdot \hat T_j$ is absolutely convergent; all subsequent interchanges of sums and evaluations rely on this.

with the **exact envelope**

$$
\eta_j(\ell)\;=\;\sum_{n\in\mathbb Z^N}\rho\!\big(q(\,\iota(\Delta z+n,\,\delta)\,)\big)\quad\text{(periodization applied to the spatial block).}
$$

> **Remark.** The *nearest‑image* practice of replacing $\Delta z$ by $\Delta z^{\mathbb T}:=\mathrm{wrap}(\Delta z)$ yields a fast approximation (Sec. A.6). In this definition the torus periodicity is exact and handled analytically; no lattice materialization is required at evaluation time.

## A.3. Dual (Poisson) Representation

By the Poisson summation formula applied to the Gaussian with covariance $S_{0,j}$, one has the absolutely convergent **dual series**

$$
G_j^{\mathbb T}(\Delta z)\;=\; (\det S_{0,j}^{-1})^{-\tfrac12}\,\sum_{k\in\mathbb Z^N} \exp\!\big(-\pi\,k^{\!*} S_{0,j}\,k\big)\,\exp\!\big(2\pi i\,\langle k,\,\Delta z\rangle\big).
$$

Here the periodization acts on the real part $\Re(\Delta z)$; the imaginary part contributes only via $e^{2\pi i\langle k,\Delta z\rangle}$ and preserves absolute convergence.
Hence

$$
\eta_j(\ell)\;=\;\exp\!\big(-\pi\,\delta^{\!*} S_{0,j}^{-1}\,\delta\big)\,(\det S_{0,j}^{-1})^{-\tfrac12}\sum_{k\in\mathbb Z^N} \exp\!\big(-\pi\,k^{\!*} S_{0,j}\,k\big)\,e^{2\pi i\langle k,\Delta z\rangle}.
$$

This representation is often preferable for broad kernels (large $\sigma$) due to rapid decay in $k$‑space.

## A.4. Equivalence to the Canonical Periodized Field

Let $\psi_j^{\mathbb T}(z,\vec d)$ denote the canonical CPSF periodized contribution (as in the base canon). With $S_{0,j}\succ0$ and absolute convergence of the Gaussian periodization,

$$
\boxed{\;T(z,\vec d)\;=\;\sum_j \alpha_j\,\psi_j^{\mathbb T}(z,\vec d)\,\hat T_j\;\equiv\;\tau(\ell\mid\mathcal C)\;,\quad \ell=(z,\vec d).\;}
$$

*Proof.* By the canon, the periodized contribution is

$$
\psi_j^{\mathbb T}(z,\vec d)\;=\;\sum_{n\in\mathbb Z^N} \exp\!\big(-\pi\, q(\,\iota(\Delta z+n,\,\delta)\,)\big),
$$

with $\Delta z=\mathrm{lift}(z)-\mathrm{lift}(z_j)$, $\delta=\delta_{\vec d}(\vec d,\vec d_j)$, and $q(w)=\langle \Sigma_j^{-1}w, w\rangle$ for $\Sigma_j=\operatorname{diag}(S_{0,j},S_{0,j})$, $S_{0,j}\succ0$. Since $\Sigma_j^{-1}=\operatorname{diag}(S_{0,j}^{-1},S_{0,j}^{-1})$, one has

$$
q\big(\iota(u,\delta)\big)\;=\;u^{\!*}S_{0,j}^{-1}u\; +\; \delta^{\!*}S_{0,j}^{-1}\delta.
$$

Therefore, for each fixed $j$,

$$
\psi_j^{\mathbb T}(z,\vec d)\;=\;\exp\!\big(-\pi\,\delta^{\!*}S_{0,j}^{-1}\delta\big)\,\sum_{n\in\mathbb Z^N}\exp\!\big(-\pi\,(\Delta z+n)^{\!*}S_{0,j}^{-1}(\Delta z+n)\big)\;=\;\eta_j(\ell),
$$

where $\eta_j$ is exactly the envelope defined in A.2. Absolute convergence of the Gaussian periodization for $S_{0,j}\succ0$ justifies factoring out the $n$‑independent angular term and termwise evaluation. Summing over $j$ and using linearity yields

$$
T(z,\vec d)\;=\;\sum_j \alpha_j\,\psi_j^{\mathbb T}(z,\vec d)\,\hat T_j\;=\;\sum_j \alpha_j\,\eta_j(\ell)\,\hat T_j\;=\;\tau(\ell\mid\mathcal C),\quad \ell=(z,\vec d).
$$

$\square$

## A.5. Structural Properties

1. **Toroidality.** $\tau(\ell\mid\mathcal C)$ is invariant under $z\mapsto z+n$, $n\in\mathbb Z^N$ (by construction of $G_j^{\mathbb T}$).
2. **Linearity.** Linear in ${\hat T_j}$ and in ${\alpha_j}$.
3. **Axial anisotropy.** Controlled by $(\sigma_j^{\parallel},\sigma_j^{\perp})$ and $R_{\mathrm{ext}}(\vec d_j)$; isotropic limit $\sigma_j^{\parallel}=\sigma_j^{\perp}$.
4. **Equivariance of frames.** For $R\mapsto R \cdot \mathrm{diag}(e^{i\phi},U)$ with $U\in U(N-1)$, one has $b\mapsto e^{i\phi}b$ and thus $S_{0,j}$ (hence $\Sigma_j$ and $\tau$) unchanged.
5. **Interference.** Coherent superposition $\sum_j \alpha_j \cdot \eta_j(\ell) \cdot \hat T_j$.
6. **Angular smoothness.** $\delta_{\vec d}$ is smooth and vanishes at $\vec d\parallel\vec d_j$; thus $\eta_j$ is smooth in both $z$ and $\vec d$.

## A.6. Nearest–Image Approximation and Rigorous Bounds

Define the **nearest‑image envelope** by evaluating the non‑periodized Gaussian at the nearest representative

$$
\eta_j^{\mathrm{nearest}}(\ell)\;:=\;\exp\!\big(-\pi\,\delta^{\!*} S_{0,j}^{-1}\,\delta\big)\;\exp\!\big(-\pi\,(\Delta z^{\mathbb T})^{\!*} S_{0,j}^{-1}\,\Delta z^{\mathbb T}\big),\qquad \Delta z^{\mathbb T}:=\mathrm{wrap}(\Delta z).
$$

Let

$$
\Delta q_{\min}:=\min_{n\in\mathbb Z^N\setminus\{0\}}\Big[\,(\Delta z+n)^{\!*}S_{0,j}^{-1}(\Delta z+n)-\Delta z^{\!*}S_{0,j}^{-1}\Delta z\,\Big].
$$

Then

$$
\big\|\eta_j(\ell)-\eta_j^{\mathrm{nearest}}(\ell)\big\|\;\le\; \Theta(S_{0,j})\,e^{-\pi\,\Delta q_{\min}}.
$$

Here $\Theta(S_{0,j})$ is a finite constant depending only on $S_{0,j}$ and $N$ (the tail of the associated theta–series).
In particular, the nearest‑image error is $\le \varepsilon$ whenever $\Delta q_{\min}\ge \pi^{-1}\log(1/\varepsilon)$. For the **dual truncation**, keeping only

$$
\mathcal K_\varepsilon\;=\;\{\,k\in\mathbb Z^N:\ k^{\!*}S_{0,j}\,k\le \pi^{-1}\log(1/\varepsilon)\,\}
$$

controls the error to $\mathcal O(\varepsilon)$; $|\mathcal K_\varepsilon|$ grows polynomially in $N$ for fixed $\varepsilon$.

## A.7. Line–Integral Variant (Optional)

For $[a,b]\subset\mathbb R$ define

$$
\mathcal T(\ell;[a,b])\;=\;\int_a^b \tau\big((z+t\vec d)\mid\mathcal C\big)\,dt.
$$

With Gaussian envelopes the integral admits closed forms of error‑function type or is well approximated by low‑order quadratures. The definition of $\tau$ itself is unchanged.

## A.8. Regularity and Differentiability

All maps in A.0 are smooth; $S_{0,j}\succ0$ ensures strong convexity of the quadratic. Hence $\eta_j$ and $\tau$ are $C^\infty$ jointly in $(z,\vec d)$ and parameters $(\sigma_j^{\parallel},\sigma_j^{\perp})$, with gradients flowing through $R_{\mathrm{ext}}$ and the assembly $\iota$.

## A.9. Summary of Exact vs Approximate Forms

* **Exact (torus‑periodic):** A.2 with $G_j^{\mathbb T}$; A.3 gives the dual series (no lattice materialization).
* **Approximate (nearest‑image):** A.6 with explicit exponentially small error; recommended as the default *compute backend* when $\Delta q_{\min}$ is large.

---

**End of Appendix A.**
