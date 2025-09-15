# Appendix B — PHC (Poisson–Hermite–Clenshaw) evaluator for T-field — `T_PHC`

Poisson–Hermite–Clenshaw evaluator for T-field.

## B.0 Notation

Batches $B$, sources $M$, dimension $N\ge 2$, output $S$.
Inputs (per source $j$): $\hat T_j\in\mathbb C^S$, $\alpha_j\in\mathbb R$, $\sigma_{\parallel,j},\sigma_{\perp,j}>0$, direction $\vec d_j\in\mathbb C^N$, position $z_j\in\mathbb C^N$. Query $(z,\vec d)$ with $z\in\mathbb C^{B\times N}$, $\vec d\in\mathbb C^{B\times N}$. Lift $\Delta z=\mathrm{lift}(z)-\mathrm{lift}(z_j)$. Wrap only $\Re(\Delta z)$ into $[-\tfrac12,\tfrac12)$; $\Im(\Delta z)$ is free. Let $b_j:=\vec d_j/\|\vec d_j\|$ with real/imag parts $b_j^R,b_j^I$. Define

$$
a_j:=\sigma_{\perp,j}^{-1},\quad
\gamma_j:=\frac{\sigma_{\perp,j}}{\sigma_{\parallel,j}},\quad
\kappa_j:=\sqrt{\frac{(\sigma_{\parallel,j}-\sigma_{\perp,j})\,\sigma_{\perp,j}}{\pi\,\sigma_{\parallel,j}}},\quad
\kappa_j^{\mathrm{eff}}:=\kappa_j/\sqrt{\gamma_j},\quad
c_{\mathrm{ang},j}:=\frac{\sigma_{\parallel,j}-\sigma_{\perp,j}}{\sigma_{\parallel,j}\sigma_{\perp,j}}.
$$

Angular offset $\delta\vec d:=\vec d-\vec d_j$. Gaussian–Hermite 1D nodes/weights $(\tau_q,w_q)_{q=1}^Q$; 2D tensor nodes $(\tau_{q_1},\tau_{q_2})$ and weights $w_{q_1}w_{q_2}$. Normalization

$$
\mathcal N_j=\frac{1}{\pi}\,\gamma_j^{-1/2}.
$$

---

## B.1 Phase factorization (double trigonometric split)

For each $(b,j,n)$ and $(q_1,q_2)$,

$$
A_{bjn}:=2\pi\,\Re(\Delta z_{bjn}),\quad
\phi_{jn}(q_1):=2\pi\,\kappa_j^{\mathrm{eff}}\,\tau_{q_1}\,b^{R}_{jn},\quad
\psi_{jn}(q_2):=2\pi\,\kappa_j^{\mathrm{eff}}\,\tau_{q_2}\,b^{I}_{jn}.
$$

Set

$$
x_{bjn}(q_1,q_2):=\cos\!\big(A_{bjn}-\phi_{jn}(q_1)-\psi_{jn}(q_2)\big)
= \cos A_{bjn}\,(\cos\phi\cos\psi)-\sin A_{bjn}\,(\sin\phi\sin\psi),
$$

assembled from precomputed $\cos/\sin$ of $A,\phi,\psi$ with an outer-product over $q_1,q_2$.&#x20;

---

## B.2 Chebyshev expansion in $k$ and truncation

Let $T_k$ be the Chebyshev polynomial of the first kind, $T_k(\cos\theta)=\cos(k\theta)$. Coefficients

$$
c_{j,k}=\exp\!\big(-\pi\,k^2/a_j^{-1}\big)=\exp\!\big(-\pi k^2/a_j\big),\quad
\rho_j:=e^{-2\pi/a_j},\quad r_{j,k}:=\frac{c_{j,k+1}}{c_{j,k}}=e^{-(2k+1)\pi/a_j}.
$$

Per $(b,j,n,q_1,q_2)$ and truncation $K_j$,

$$
S_{bjn}(q_1,q_2;K_j)=\sum_{k=1}^{K_j} c_{j,k}\,T_k\!\big(x_{bjn}(q_1,q_2)\big).
$$

Closed forms for $K_j\le 4$: $c_{j,1}=e^{-\pi/a_j},c_{j,2}=e^{-4\pi/a_j},c_{j,3}=e^{-9\pi/a_j},c_{j,4}=e^{-16\pi/a_j}$. For $K_j>4$, Clenshaw with seed

$$
c_{j,K_j}=
\begin{cases}
\rho_j^{K_j^2/2}, & K_j\ \text{even},\\
e^{-\pi/a_j}\,\rho_j^{(K_j^2-1)/2},& K_j\ \text{odd}.
\end{cases}
$$

Tail bound (geometric):

$$
\sum_{m\ge 1} c_{j,K_j+m}\,\big|T_{K_j+m}(x)\big|
\;\le\; \frac{c_{j,K_j}\,r_{j,K_j}}{1-r_{j,K_j}}\!.
$$

Tolerance allocation: with total $\varepsilon_{\mathrm{total}}$, set

$$
\varepsilon_{\theta}:=\frac{\varepsilon_{\mathrm{total}}}{2\,N\,Q^2},\qquad
\frac{c_{j,K_j}\,r_{j,K_j}}{1-r_{j,K_j}}\le \varepsilon_{\theta}.
$$

The $K$-bucketing sorts sources by $K_j$ and processes groups with shared $K$.&#x20;

---

## B.3 Per-node log contribution and quadrature LSE

Per $(b,j,q_1,q_2)$ define

$$
\Lambda_{bj}(q_1,q_2)=\sum_{n=1}^{N}\log\!\big(1+2\,S_{bjn}(q_1,q_2;K_j)\big)\;-\;\frac{N}{2}\,\log a_j.
$$

2D GH aggregation in log-domain:

$$
\mathcal L_{bj}=\log\sum_{q_1,q_2=1}^{Q}\exp\!\Big(\Lambda_{bj}(q_1,q_2)+\log w_{q_1}+\log w_{q_2}\Big),
\qquad
\eta_{bj}=\mathcal N_j\,e^{\mathcal L_{bj}}.
$$

Numerical guard: $\log(1+\max(2S,-1+\mathrm{tiny}))$, $\log\max(w_q,\mathrm{tiny})$, $\log\max(a_j,\mathrm{tiny})$.&#x20;

---

## B.4 Imaginary-coordinate multiplier (free $\Im\Delta z$)

Let $u_{bjn}:=\Im(\Delta z_{bjn})$ (no wrapping). Global (non-quadrature) multiplier:

$$
\log \Xi_{bj} \;=\; -\pi\,a_j\sum_{n=1}^N u_{bjn}^2 \;+\; \pi\,c_{\mathrm{ang},j}\,\Big|\sum_{n=1}^N \overline{b_{jn}}\,u_{bjn}\Big|^2.
$$

Add to the envelope:

$$
\log\eta_{bj}\ \leftarrow\ \log\eta_{bj} + \log \Xi_{bj}.
$$

Only $\Re(\Delta z)$ enters trigonometric phases; $\Im(\Delta z)$ contributes via this real Gaussian factor.&#x20;

---

## B.5 Directional angular factor

With $\delta\vec d_{bj}:=\vec d_b-\vec d_j$,

$$
q^{\mathrm{ang}}_{bj}=\frac{\|\delta\vec d_{bj}\|^2}{\sigma_{\perp,j}}
-\frac{\sigma_{\parallel,j}-\sigma_{\perp,j}}{\sigma_{\parallel,j}\sigma_{\perp,j}}
\big|\langle b_j,\delta\vec d_{bj}\rangle\big|^2,\qquad
\log\mathrm{Ang}_{bj}=-\pi\,q^{\mathrm{ang}}_{bj}.
$$

This factor is accumulated outside of quadrature; it is purely real.&#x20;

---

## B.6 Final weight and assembly

Log-weight per $(b,j)$:

$$
\log W_{bj}=\log|\alpha_j|+\log\eta_{bj}+\log\mathrm{Ang}_{bj},\qquad
\mathrm{sign}(\alpha_j)\in\{\pm 1,0\}.
$$

Stable exponentiation via row-wise $\max$: for each $b$, set $m_b=\max_j \log W_{bj}$ and compute

$$
\tilde W_{bj}=\exp(\log W_{bj}-m_b)\cdot \mathrm{sign}(\alpha_j),\quad
\mathsf{scale}_b=e^{m_b}.
$$

Output (two GEMMs):

$$
\Re T_{b,:}\;=\;\mathsf{scale}_b\cdot \sum_{j}\tilde W_{bj}\,\Re\hat T_{j,:},\qquad
\Im T_{b,:}\;=\;\mathsf{scale}_b\cdot \sum_{j}\tilde W_{bj}\,\Im\hat T_{j,:}.
$$

This matches the implemented accumulation strategy (log-domain + LSE + GEMM).&#x20;

---

## B.7 Streaming, tiling, caching

* Source tiling: $j\in[m_0,m_1)$.
* Coordinate tiling: $n\in[n_0,n_1)$; accumulate $\sum_n \log(1+2S_{bjn})$ without materializing $B\times m_c\times N\times Q^2$.
* Quadrature tiling: block $(q_1,q_2)$ via outer-products of $(j,n,q_1)$ and $(j,n,q_2)$ trigs; accumulate per-block LSE.
* $K$-bucketing: sort by $K_j$; per-bucket Chebyshev (closed $K\le 4$, Clenshaw $K>4$).
* Cache GH $\{\tau_q,\log w_q\}$ per dtype/device/$Q$.&#x20;

---

## B.8 Numerical guards and Lipschitz control

* Machine guard $\mathrm{tiny}$ by dtype.
* Denominator guards in Clenshaw tail: $\max(1-r_{j,K_j},\mathrm{tiny})$.
* Lipschitz: $f(x)=\log(1+2x)$ on $x>-1/2$ has $|f'(x)|\le 2$. If $|\Delta S_{bjn}|\le \varepsilon_\theta$ pointwise, then $|\Delta \Lambda_{bj}|\le 2\,N\,Q^2\,\varepsilon_\theta$. Choose $\varepsilon_\theta=\varepsilon_{\mathrm{total}}/(2NQ^2)$.&#x20;

---

## B.9 Differentiability

All continuous inputs $(z,z_j,\vec d,\vec d_j,\alpha_j,\sigma_{\parallel},\sigma_{\perp})$ pass through smooth primitives ($\cos,\sin,\exp,\log,\log1p$, LSE, GEMM). Non-smooth control (sorting by $K_j$, integer $K_j$) is piecewise-constant and does not obstruct backpropagation through the continuous path.&#x20;

---

## B.10 Complexity

Let $Q$ be the 1D GH size. Time $\Theta(B\,M\,N\,Q^2)$ (constant factors reduced by fused trigs and tiling). Peak memory per source-tile $\mathcal O(B\,m_c + B\,m_c\,n_s + m_c\,S)$; quadrature cache $\mathcal O(Q)$.&#x20;
