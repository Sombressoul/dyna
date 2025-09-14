# CPSF — Appendix A: Ray–Trace Operator $\tau$

## A.0. Conventions (from Canon)

* Dimension: $N\ge 2$; domain $(\mathbb C/\mathbb Z)^N$ with unit periods componentwise.
* Ray: $\ell=(z,\vec d)$ with $z\in\mathbb C^N$, $|\vec d|=1$.
* Contribution: $C_j=(\ell_j,\hat T_j,\sigma_j^{\parallel},\sigma_j^{\perp},\alpha_j)$ with $\ell_j=(z_j,\vec d_j)$, $\hat T_j\in\mathbb C^S$, $\sigma_j^{\parallel},\sigma_j^{\perp}>0$, $\alpha_j\in\mathbb R$.
* Lift/wrap: $\mathrm{lift}$ to $\mathbb C^N$; **wrap applies to $\Re(\cdot)$ only**, mapping each real coordinate to $[-\tfrac12,\tfrac12)$; $\Im(\cdot)$ is free (non-periodized).
* Directional offset: $\delta_{\vec d}(\vec d,\vec d_j)\in\mathbb C^N$, smooth in $(\vec d,\vec d_j)$, tangent to $\vec d_j$, vanishes for $\vec d\parallel\vec d_j$.
* Frame: $R(\vec d_j)\in U(N)$; first column $b$ aligned with $\vec d_j$. Extended $R_{\mathrm{ext}}=\mathrm{diag}(R,R)\in U(2N)$.
* Spatial covariance block (per $j$):

  $$
  S_{0,j}=\sigma_j^{\perp}I_N+(\sigma_j^{\parallel}-\sigma_j^{\perp})\,b\,b^{*},\qquad
  \Sigma_j=\mathrm{diag}(S_{0,j},S_{0,j}).
  $$
* Quadratic form and Gaussian: $q(w)=\langle \Sigma_j^{-1}w,w\rangle$, $\rho(q)=e^{-\pi q}$. Hermitian inner products; batch broadcasting implicit. &#x20;

---

## A.1. Torus–Periodic Spatial Kernel

Let $\Delta z:=\mathrm{lift}(z)-\mathrm{lift}(z_j)\in\mathbb C^N$ and $\delta:=\delta_{\vec d}(\vec d,\vec d_j)$. Define the periodized spatial Gaussian

$$
G_j^{\mathbb T}(\Delta z):=\sum_{n\in\mathbb Z^N}\exp\!\Big(-\pi\,(\Delta z+n)^{*}S_{0,j}^{-1}(\Delta z+n)\Big),\quad S_{0,j}\succ 0.
$$

The series converges absolutely and uniformly on compact sets; $G_j^{\mathbb T}$ is smooth and $1$-periodic in $\Re(\Delta z)$. &#x20;

---

## A.2. Ray–Trace Operator

Given $\mathcal C=\{C_j\}$ and $\ell=(z,\vec d)$, define

$$
\boxed{\;\tau(\ell\mid\mathcal C)=\sum_j \alpha_j\,\eta_j(\ell)\,\hat T_j\in\mathbb C^S\;}
$$

with the exact envelope

$$
\eta_j(\ell)=\sum_{n\in\mathbb Z^N}\rho\!\Big(q\big(\iota(\Delta z+n,\delta)\big)\Big),\qquad
q\big(\iota(u,\delta)\big)=u^{*}S_{0,j}^{-1}u+\delta^{*}S_{0,j}^{-1}\delta.
$$

Absolute convergence (for $S_{0,j}\succ0$) justifies factoring the $\delta$-term. &#x20;

---

## A.3. Dual (Poisson) Representation with Complex Coordinates

Poisson summation in $\mathbb Z^N$ yields

$$
G_j^{\mathbb T}(\Delta z)=(\det S_{0,j}^{-1})^{-\tfrac12}\sum_{k\in\mathbb Z^N}
\exp\!\Big(-\pi\,k^{*}S_{0,j}k\Big)\,\exp\!\big(2\pi i\langle k,\Delta z\rangle\big).
$$

Write $a_j=\sigma_j^{\perp}{}^{-1}$, $c_{\mathrm{ang},j}=(\sigma_j^{\parallel}-\sigma_j^{\perp})/(\sigma_j^{\parallel}\sigma_j^{\perp})=1/\sigma_j^{\perp}-1/\sigma_j^{\parallel}$,
$u=\Im(\Delta z)$, and $x=\Re(\Delta z)$. Then

$$
\begin{aligned}
\eta_j(\ell)
&=\exp\!\big(-\pi\,\delta^{*}S_{0,j}^{-1}\delta\big)\,(\det S_{0,j}^{-1})^{-\tfrac12}
\sum_{k\in\mathbb Z^N}\exp\!\big(-\pi\,k^{*}S_{0,j}k\big)\,e^{2\pi i\langle k,x\rangle}\;\cdot\; \Xi_j(u),\\[2mm]
\Xi_j(u)
&=\exp\!\Big(-\pi\,a_j\,\|u\|^2+\pi\,c_{\mathrm{ang},j}\,\big|\langle b,u\rangle\big|^2\Big),
\end{aligned}
$$

i.e. **only $\Re(\Delta z)$** enters the oscillatory factor; the **free imaginary coordinate** contributes via the real Gaussian multiplier $\Xi_j(u)$. &#x20;

---

## A.4. Equivalence to the Canonical Periodized Field

Let $\psi_j^{\mathbb T}(z,\vec d)$ be the canonical periodized contribution. For $S_{0,j}\succ0$ and absolute convergence of the Gaussian periodization,

$$
\boxed{\;T(z,\vec d)=\sum_j \alpha_j\,\psi_j^{\mathbb T}(z,\vec d)\,\hat T_j\equiv \tau(\ell\mid\mathcal C),\quad \ell=(z,\vec d).\;}
$$

Sketch: $q(\iota(u,\delta))=u^{*}S_{0,j}^{-1}u+\delta^{*}S_{0,j}^{-1}\delta$ implies $\psi_j^{\mathbb T}=\exp(-\pi\,\delta^{*}S_{0,j}^{-1}\delta)\,G_j^{\mathbb T}$, and Poisson gives A.3. Linearity in $j$ completes the identity. &#x20;

---

## A.5. Nearest–Image Approximation

Let $x^{\mathbb T}=\mathrm{wrap}(x)\in[-\tfrac12,\tfrac12)^N$. The nearest–image envelope evaluates the non-periodized Gaussian at $(x^{\mathbb T},u)$:

$$
\eta_j^{\mathrm{near}}(\ell)=\exp\!\big(-\pi\,\delta^{*}S_{0,j}^{-1}\delta\big)\,
\exp\!\Big(-\pi\,\big((x^{\mathbb T})^{*}S_{0,j}^{-1}x^{\mathbb T}+u^{*}S_{0,j}^{-1}u\big)\Big).
$$

Let

$$
\Delta q_{\min}:=\min_{n\in\mathbb Z^N\setminus\{0\}}
\Big[(x+n)^{*}S_{0,j}^{-1}(x+n)-x^{*}S_{0,j}^{-1}x\Big].
$$

Then there exists $\Theta(S_{0,j})<\infty$ such that

$$
\big|\eta_j(\ell)-\eta_j^{\mathrm{near}}(\ell)\big|
\;\le\;\Theta(S_{0,j})\,e^{-\pi\,\Delta q_{\min}}.
$$

Thus the nearest–image error is $\le\varepsilon$ whenever $\Delta q_{\min}\ge \pi^{-1}\log(1/\varepsilon)$. &#x20;

---

## A.6. Dual Truncation

For $\varepsilon\in(0,1)$ define the truncation set

$$
\mathcal K_{\varepsilon}:=\Big\{k\in\mathbb Z^N:\;k^{*}S_{0,j}k\le \pi^{-1}\log(1/\varepsilon)\Big\}.
$$

Then

$$
\sum_{k\notin\mathcal K_{\varepsilon}}
\exp\!\big(-\pi\,k^{*}S_{0,j}k\big)
\;\le\;\mathcal O(\varepsilon),
$$

and the truncated dual series attains $\mathcal O(\varepsilon)$ error. For fixed $\varepsilon$, $|\mathcal K_{\varepsilon}|$ grows polynomially in $N$. &#x20;

---

## A.7. Structural Properties

1. **Toroidality:** $\tau(\ell)$ is invariant under $z\mapsto z+n$, $n\in\mathbb Z^N$.
2. **Linearity:** Linear in $\hat T_j$ and $\alpha_j$.
3. **Axial anisotropy:** Controlled by $(\sigma_j^{\parallel},\sigma_j^{\perp})$ via $S_{0,j}$. Isotropic limit $\sigma_j^{\parallel}=\sigma_j^{\perp}$.
4. **Frame equivariance:** $R\mapsto R\cdot\mathrm{diag}(e^{i\phi},U)$, $U\in U(N-1)$, leaves $S_{0,j}$ and $\tau$ invariant.
5. **Angular smoothness:** $\delta_{\vec d}$ is smooth, $\delta_{\vec d}=0$ for $\vec d\parallel\vec d_j$; hence $\eta_j$ is smooth in $(z,\vec d)$. &#x20;

---

## A.8. Line–Integral Variant

For $[a,b]\subset\mathbb R$,

$$
\mathcal T(\ell;[a,b])=\int_a^b \tau\big((z+t\vec d)\mid\mathcal C\big)\,dt.
$$

With Gaussian envelopes the integral admits error-function forms or low-order quadrature; the definition of $\tau$ is unchanged. &#x20;

---

## A.9. Complexity (Backends)

* Nearest–image: $\mathcal O(M)$ per query; no frequency set; recommended when $\Delta q_{\min}$ is large.
* Dual (Poisson): $\mathcal O\big(M(|\mathcal K_{\varepsilon}|+N)\big)$ with symmetric $k$-set; recommended for broad kernels (fast spectral decay). &#x20;
