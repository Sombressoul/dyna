# Appendix B — Poisson-only `T_HS_Theta`: Mathematics and Theory (N ≥ 2)

## 1. Notation and assumptions

Dimensions: $B$ (batch), $M$ (sources), $N\ge 2$ (direction size), $S$ (output size).
Inputs:

$$
\begin{aligned}
&z\in\mathbb{R}^{B\times N},\quad z_j\in\mathbb{R}^{M\times N},\\
&\vec d\in\mathbb{C}^{B\times N},\quad \vec d_j\in\mathbb{C}^{M\times N},\\
&\widehat T_j\in\mathbb{C}^{M\times S},\quad \alpha_j\in\mathbb{R}^M,\\
&\sigma_{\parallel},\sigma_{\perp}\in\mathbb{R}^M,\quad \sigma_{\parallel,j}>0,\ \sigma_{\perp,j}>0.
\end{aligned}
$$

1D Gauss–Hermite nodes/weights for weight $e^{-t^2}$:
$(\tau_q,w_q)_{q=1}^Q$ (Golub–Welsch).
2D tensor product rule: nodes $(\tau_{q_1},\tau_{q_2})$, weights $w_{q_1}w_{q_2}$.
Fractional reduction: $\operatorname{frac}(x):=\mathrm{frac}(x+\tfrac12)-\tfrac12\in[-\tfrac12,\tfrac12)$.

## 2. Anisotropy and normalization (Poisson-only)

$$
a_j:=\frac{1}{\sigma_{\perp,j}},\qquad 
\gamma_j:=\frac{\sigma_{\perp,j}}{\sigma_{\parallel,j}},\qquad 
\kappa_j:=\sqrt{\frac{(\sigma_{\parallel,j}-\sigma_{\perp,j})\,\sigma_{\perp,j}}{\pi\,\sigma_{\parallel,j}}}.
$$

Unit direction $b_j:=\vec d_j/\|\vec d_j\|\in\mathbb{C}^N$; real/imag parts $b_j^R,b_j^I\in\mathbb{R}^N$.
Effective amplitude $\kappa^{\mathrm{eff}}_j:=\kappa_j\,\gamma_j^{-1/2}$.
Phase shifts for batch $b$, source $j$, component $n$:

$$
\Delta z_{b,j,n}:=\operatorname{frac}\!\left(z_{b,n}-z_{j,n}\right)\in[-\tfrac12,\tfrac12).
$$

Angular factor:

$$
q_{b,j}\ :=\ \frac{\|\vec d_b-\vec d_j\|^2}{\sigma_{\perp,j}}
\;-\;\frac{\sigma_{\parallel,j}-\sigma_{\perp,j}}{\sigma_{\parallel,j}\sigma_{\perp,j}}\,
\bigl|\langle b_j,\ \vec d_b-\vec d_j\rangle\bigr|^2,
\qquad 
\mathrm{Ang}_{b,j}:=\exp(-\pi\,q_{b,j}).
$$

Normalization:

$$
\mathcal{N}_j:=\frac{1}{\pi}\,\gamma_j^{-1/2}.
$$

## 3. Phase factorization and double trigonometric split

For $q_1,q_2\in\{1,\dots,Q\}$ and $n\in\{1,\dots,N\}$:

$$
\phi_{j,n}(q_1):=2\pi\,\tau_{q_1}\,\kappa_j^{\mathrm{eff}}\, b^R_{j,n},\qquad
\psi_{j,n}(q_2):=2\pi\,\tau_{q_2}\,\kappa_j^{\mathrm{eff}}\, b^I_{j,n}.
$$

$$
\theta_{b,j,n}(q_1,q_2):=2\pi\,\Delta z_{b,j,n}-\bigl(\phi_{j,n}(q_1)+\psi_{j,n}(q_2)\bigr).
$$

Define $x_{b,j,n}(q_1,q_2):=\cos\theta_{b,j,n}(q_1,q_2)$ via

$$
\cos(\theta)=\cos A\cos B+\sin A\sin B,\quad
\cos(\phi+\psi)=\cos\phi\cos\psi-\sin\phi\sin\psi,
$$

with $A=2\pi\,\Delta z_{b,j,n}$, $B=\phi_{j,n}(q_1)+\psi_{j,n}(q_2)$.
Thus $\cos A,\sin A$ are computed per $(b,j,n)$; $\cos\phi,\sin\phi$ per $(j,n,q_1)$; $\cos\psi,\sin\psi$ per $(j,n,q_2)$.

## 4. Spectral series and tail control

Chebyshev polynomials of the first kind: $T_k(\cos\vartheta)=\cos(k\vartheta)$, $|T_k(x)|\le 1$ for $|x|\le 1$.
Coefficients:

$$
c_{j,k}:=\exp\!\bigl(-\pi\,k^2\,a_j^{-1}\bigr),\qquad 
r_{j,k}:=\frac{c_{j,k+1}}{c_{j,k}}=\exp\!\bigl(-(2k+1)\pi\,a_j^{-1}\bigr),\quad 
\rho_j:=e^{-2\pi a_j^{-1}}.
$$

Per $(b,j,n,q_1,q_2)$ and truncation $K\in\mathbb{N}$:

$$
S_{b,j,n}(q_1,q_2;K):=\sum_{k=1}^{K} c_{j,k}\,T_k\!\bigl(x_{b,j,n}(q_1,q_2)\bigr).
$$

**Closed form for $K\le 4$** via one base $e_j:=e^{-\pi a_j^{-1}}$:

$$
c_{j,1}=e_j,\quad c_{j,2}=e_j^{4},\quad c_{j,3}=e_j^{9},\quad c_{j,4}=e_j^{16}.
$$

**Clenshaw for $K>4$** with

$$
c_{j,K}=
\begin{cases}
\rho_j^{K^2/2},& K\ \text{even},\\[2pt]
e_j\,\rho_j^{(K^2-1)/2},& K\ \text{odd}.
\end{cases}
$$

**Tail bound (geometric):**

$$
\sum_{m\ge 1} c_{j,K+m}\,\bigl|T_{K+m}(x)\bigr|
\;\le\; \frac{c_{j,K}\,r_{j,K}}{1-r_{j,K}}.
$$

## 5. Log-domain stabilization and error allocation

Per $(b,j,q_1,q_2)$, sum over $n=1,\dots,N$:

$$
\Lambda_{b,j}(q_1,q_2):=\sum_{n=1}^{N}\log\!\bigl(1+2\,S_{b,j,n}(q_1,q_2;K_j)\bigr)-\tfrac12\,N\,\log a_j.
$$

2D quadrature (log-sum-exp):

$$
\mathcal{L}_{b,j}:=\log\sum_{q_1,q_2=1}^{Q}\exp\!\Bigl(\Lambda_{b,j}(q_1,q_2)+\log w_{q_1}+\log w_{q_2}\Bigr).
$$

Set

$$
\eta_{b,j}:=\mathcal{N}_j\,\exp(\mathcal{L}_{b,j}).
$$

**Lipschitz control.** For $f(x)=\log(1+2x)$ on $x>-1/2$:

$$
|f'(x)|=\frac{2}{1+2x}\le 2.
$$

Hence, if $|\Delta S_{b,j,n}(q_1,q_2)|\le \varepsilon_\theta$ pointwise, then

$$
\bigl|\Delta \Lambda_{b,j}\bigr|\le 2\,N\,Q^2\,\varepsilon_\theta.
$$

Choose

$$
\varepsilon_\theta:=\frac{\varepsilon_{\mathrm{total}}}{2\,N\,Q^2}.
$$

**Stopping rule (strict):**

$$
\frac{c_{j,K}\,r_{j,K}}{1-r_{j,K}}\ \le\ \varepsilon_\theta.
$$

## 6. Final assembly

Weights:

$$
W_{b,j}:=\alpha_j\,\mathrm{Ang}_{b,j}\,\eta_{b,j}\ \in\ \mathbb{R}_{\ge 0}.
$$

Output, for $s=1,\dots,S$:

$$
\mathrm{Re}\,T_{b,s}\ =\ \sum_{j=1}^{M} W_{b,j}\,\mathrm{Re}\,\widehat T_{j,s},\qquad
\mathrm{Im}\,T_{b,s}\ =\ \sum_{j=1}^{M} W_{b,j}\,\mathrm{Im}\,\widehat T_{j,s}.
$$

(Implementation: two GEMMs for real/imag parts.)

## 7. Streaming and tiling

* Source tiling: $j\in[m_0,m_1)$ with $m_c=m_1-m_0$.
* $K$-bucketing: sort $j$ by $K_j$, process groups of equal $K$.
* Coordinate tiling: $n\in[n_0,n_1)$; accumulate $\sum_n \log(1+2S_{b,j,n})$ without materializing $(B\times m_c\times N\times Q^2)$.
* Quadrature tiling: outer-product assembly of $\cos(\phi+\psi),\sin(\phi+\psi)$ from $(j,n,q_1)$ and $(j,n,q_2)$; log-sum-exp accumulator $(A_{b,j},S_{b,j})$ updated per $(q_1,q_2)$-tile.

## 8. Numerical guards

* Machine guard $\texttt{tiny}$ by dtype.
* $\log(\max(w_q,\texttt{tiny}))$, $\log(\max(a_j,\texttt{tiny}))$, $\log(1+\max(2S,-1+\texttt{tiny}))$.
* Denominator guard $\max(1-r_{j,K},\texttt{tiny})$.
* Phase reduction via $\operatorname{frac}$ on $[-\tfrac12,\tfrac12)$.
* If available: use $\mathrm{sincos}$ for paired trigonometry.

## 9. Complexity and memory

Let $Q$ be 1D quadrature size (2D has $Q^2$ nodes).
Time:

$$
\Theta\!\bigl(B\,M\,N\,Q^2\bigr)
\quad (\text{constant factors reduced by 1D trig split and streaming}).
$$

Peak memory (per source tile):

$$
\mathcal{O}\!\bigl(B\,m_c + B\,m_c\,n_s + m_c\,S\bigr),
$$

with $m_c$ = source-chunk size, $n_s$ = coordinate-tile size. Quadrature cache: $\mathcal{O}(Q)$.

## 10. Differentiability

All continuous inputs $(z,z_j,\vec d,\vec d_j,\alpha_j,\sigma_{\parallel},\sigma_{\perp})$ pass through smooth operations (trigonometry, algebra, $\exp,\log,\log1p$, LSE, GEMM). The scheme is differentiable almost everywhere w\.r.t. these parameters. Discrete components (sorting by $K_j$, integer $K_j$ selection) are piecewise-constant by design for truncation control.

## 11. Tolerance verification

Local truncation:

$$
\frac{c_{j,K}\,r_{j,K}}{1-r_{j,K}}\ \le\ \varepsilon_\theta,\qquad 
\varepsilon_\theta=\frac{\varepsilon_{\mathrm{total}}}{2\,N\,Q^2}.
$$

Global contribution to the log-sum bounded by $\varepsilon_{\mathrm{total}}$ via the 2-Lipschitz bound of $x\mapsto\log(1+2x)$ on $x>-1/2$.
2D GH product rule achieves the standard polynomial-moment exactness up to degree $2Q-1$ per axis.
