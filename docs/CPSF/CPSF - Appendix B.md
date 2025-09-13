# Appendix B — HS-Θ: Rank-1 Periodic Gaussian via 1D Quadrature and 1D Θ–Sums

## B.0 Notation

Let $N\in\mathbb{N}$, $S\in\mathbb{N}$. Real torus $\mathbb{T}^N=\mathbb{R}^N/\mathbb{Z}^N$.
For $u\in\mathbb{R}$, define the wrap $\mathrm{wrap}(u)=u-\mathrm{round}(u)\in[-\tfrac12,\tfrac12]$; componentwise for vectors.
Data (per contribution $j=1,\dots,M$): $z_j\in\mathbb{R}^N$, $b_j\in\mathbb{R}^N$ with $\|b_j\|_2=1$, $\sigma_{\parallel,j}>\sigma_{\perp,j}>0$, $\alpha_j\in\mathbb{R}_{\ge 0}$, $\hat T_j\in\mathbb{C}^S$.
Query: $z\in\mathbb{R}^N$, direction $d\in\mathbb{R}^N$ (unit).
Set $\Delta z=\mathrm{wrap}(z-z_j)\in[-\tfrac12,\tfrac12]^N$, $\Delta d=\mathrm{wrap}(d-b_j)$ if needed.

Define constants (per $j$):

$$
a_j=\frac{1}{\sigma_{\perp,j}},\quad
c_j=\frac{1}{\sigma_{\perp,j}}-\frac{1}{\sigma_{\parallel,j}}>0,\quad
\gamma_j=\frac{\sigma_{\perp,j}}{\sigma_{\parallel,j}}\in(0,1],\quad
\kappa_j=\sqrt{\frac{\sigma_{\parallel,j}-\sigma_{\perp,j}}{\pi\,\sigma_{\parallel,j}}}.
\tag{B.0.1}
$$

1D periodic Gaussian (Jacobi theta, spatial form):

$$
\theta_{a}(u)=\sum_{n\in\mathbb{Z}} e^{-\pi a\,(n+u)^2},\qquad a>0,\;u\in\mathbb{R}.
\tag{B.0.2}
$$

Spectral (Poisson) form:

$$
\theta_{a}(u)=\frac{1}{\sqrt{a}}\sum_{k\in\mathbb{Z}} e^{-\pi k^2/a}\,e^{2\pi i k u}.
\tag{B.0.3}
$$

Angular factor (rank-1 anisotropy in direction mismatch):

$$
\mathrm{ang}_j(d,b_j)=\exp\!\Big(-\pi\Big[\tfrac{1}{\sigma_{\perp,j}}\|\Delta d\|^2-\tfrac{\sigma_{\parallel,j}-\sigma_{\perp,j}}{\sigma_{\parallel,j}\sigma_{\perp,j}}\,|\langle b_j,\Delta d\rangle|^2\Big]\Big).
\tag{B.0.4}
$$

Field:

$$
T(z,d)=\sum_{j=1}^M \alpha_j\,\eta_j(\Delta z;b_j,\sigma_{\parallel,\perp,j})\,\mathrm{ang}_j(d,b_j)\,\hat T_j.
\tag{B.0.5}
$$

## B.1 Rank-1 periodic Gaussian (lattice form)

$$
\eta_j(\Delta z)=\sum_{n\in\mathbb{Z}^N} \exp\!\Big(-\pi\big[a_j\,\|x\|_2^2-c_j\,|\langle b_j,x\rangle|^2\big]\Big),\quad x=\Delta z+n.
\tag{B.1.1}
$$

## B.2 HS factorization and coordinatewise separation

Hubbard–Stratonovich (real, rank-1):

$$
e^{\pi c_j\,(\langle b_j,x\rangle)^2}
=\frac{1}{\sqrt{\pi}}\int_{-\infty}^{\infty} e^{-t^2}\,e^{\,2\sqrt{\pi c_j}\,t\,\langle b_j,x\rangle}\,dt.
\tag{B.2.1}
$$

Insert (B.2.1) into (B.1.1), exchange sum and integral, and complete squares coordinatewise:

$$
\eta_j(\Delta z)=\frac{1}{\sqrt{\pi}}\int_{-\infty}^{\infty}e^{-t^2}
\prod_{i=1}^N\Bigg(\sum_{n_i\in\mathbb{Z}} \exp\!\Big(-\pi a_j\,(n_i+\Delta z_i-\mu_{j,i}(t))^2\Big)\Bigg)\cdot
\exp\!\Big(\tfrac{\pi c_j}{a_j}\,t^2\,\sum_{i=1}^N b_{j,i}^2\Big)\,dt,
\tag{B.2.2}
$$

with $\mu_{j,i}(t)=\dfrac{\sqrt{\pi c_j}}{\pi a_j}\,t\,b_{j,i}= \kappa_j\,t\,b_{j,i}$ and $\sum b_{j,i}^2=1$. Using $c_j/a_j=1-\sigma_{\perp,j}/\sigma_{\parallel,j}=1-\gamma_j$:

$$
\boxed{\;
\eta_j(\Delta z)=\frac{1}{\sqrt{\pi}}\int_{-\infty}^{\infty}
\exp\!\big(-\gamma_j\,t^2\big)\,\prod_{i=1}^{N}\theta_{a_j}\!\big(\Delta z_i-\kappa_j\,t\,b_{j,i}\big)\,dt\; }.
\tag{B.2.3}
$$

## B.3 1D quadrature (Gauss–Hermite scaling)

Let $\{\tau_k,w_k\}_{k=1}^K$ be Gauss–Hermite nodes/weights for $\int_{-\infty}^{\infty}e^{-\tau^2}\phi(\tau)d\tau$. Substitute $t=\tau/\sqrt{\gamma_j}$:

$$
\eta_j(\Delta z)\approx \frac{1}{\sqrt{\pi}}\sum_{k=1}^{K} w_k\,
\prod_{i=1}^{N}\theta_{a_j}\!\Big(\Delta z_i-\kappa_j\,\tfrac{\tau_k}{\sqrt{\gamma_j}}\,b_{j,i}\Big).
\tag{B.3.1}
$$

Error (quadrature) decreases super-algebraically with $K$ for analytic integrands; choose $K$ to meet target $\varepsilon_{\mathrm{quad}}$.

## B.4 1D theta evaluation policy

For each scalar call $\theta_{a}(u)$:

**Direct (spatial) truncation** for $a\gtrsim a_{\mathrm{thr}}$:

$$
\theta_{a}(u)\approx \sum_{n=-W}^{W} e^{-\pi a\,(n+u)^2},\quad
\mathcal{E}^{\mathrm{dir}}_{\theta}\le 2\sum_{n=W+1}^{\infty} e^{-\pi a\,(n-|u|)^2}
\;\le\; \frac{2}{\sqrt{\pi a}}\int_{W-|u|}^{\infty} e^{-\pi a\,t^2}dt.
\tag{B.4.1}
$$

**Poisson (spectral) truncation** for $a\lesssim a_{\mathrm{thr}}$:

$$
\theta_{a}(u)=\frac{1}{\sqrt{a}}\Big(1+2\sum_{k=1}^{K'} e^{-\pi k^2/a}\cos(2\pi k u)\Big),
\quad
\mathcal{E}^{\mathrm{sp}}_{\theta}\le \frac{2}{\sqrt{a}}\sum_{k=K'+1}^{\infty} e^{-\pi k^2/a}.
\tag{B.4.2}
$$

Target per-call relative error:

$$
\max(\mathcal{E}^{\mathrm{dir}}_{\theta},\mathcal{E}^{\mathrm{sp}}_{\theta})\ \le\ \varepsilon_{\theta},
\quad \varepsilon_{\theta}:=\frac{\varepsilon_{\mathrm{tot}}}{2\,N\,K}.
\tag{B.4.3}
$$

## B.5 Total error budget

Let $\varepsilon_{\mathrm{tot}}\in(0,1)$ (e.g. $10^{-3}$). Choose $K$ and $(W\text{ or }K')$ such that

$$
\varepsilon_{\mathrm{quad}}\ \le\ \frac{\varepsilon_{\mathrm{tot}}}{2},\qquad
\varepsilon_{\theta}\ \le\ \frac{\varepsilon_{\mathrm{tot}}}{2 N K}
\ \ \Rightarrow\ \ 
\big|\eta_j-\hat\eta_j\big|\ \le\ \varepsilon_{\mathrm{tot}}\cdot C_j,
\tag{B.5.1}
$$

with $C_j:=\max_{t}\prod_{i}\theta_{a_j}(\Delta z_i-\kappa_j t b_{j,i})$ (bounded for fixed $a_j$). Consequently for field

$$
\|T-\hat T\|\ \le\ \sum_{j=1}^{M}\alpha_j\,\varepsilon_{\mathrm{tot}}\,C_j\,\|\hat T_j\|.
\tag{B.5.2}
$$

Using fixed $\varepsilon_{\mathrm{tot}}$ and bounded $\alpha_j,C_j$, the error is controlled linearly by $\sum \alpha_j \|\hat T_j\|$.

## B.6 Differentiability

All quantities are $C^{\infty}$ in $(\Delta z,b_j,\sigma_{\parallel,j},\sigma_{\perp,j})$ away from wrap boundaries.

Per-coordinate derivatives (direct form):

$$
\partial_u \theta_{a}(u)= -2\pi a\sum_{n\in\mathbb{Z}}(n+u)\,e^{-\pi a(n+u)^2},
\quad
\partial_a \theta_{a}(u)= -\pi\sum_{n\in\mathbb{Z}}(n+u)^2\,e^{-\pi a(n+u)^2}.
\tag{B.6.1}
$$

Stable ratios:

$$
\partial_u \log\theta_a(u)= -2\pi a\,\frac{\sum (n+u)\,e^{-\pi a(n+u)^2}}{\sum e^{-\pi a(n+u)^2}}.
\tag{B.6.2}
$$

Chain rules:

$$
\partial_{\Delta z_i}\log\eta_j = 
\frac{1}{\eta_j}\cdot \frac{1}{\sqrt{\pi}}\int e^{-\gamma_j t^2}
\Big(\prod_{r}\theta_{a_j}(\cdot)\Big)\,\partial_{u}\log\theta_{a_j}(u_i)\,dt,
\quad u_i=\Delta z_i-\kappa_j t b_{j,i}.
\tag{B.6.3}
$$

$$
\partial_{b_{j,i}}\log\eta_j = 
-\kappa_j\frac{1}{\eta_j}\cdot \frac{1}{\sqrt{\pi}}\int e^{-\gamma_j t^2}
\Big(\prod_{r}\theta_{a_j}(\cdot)\Big)\,t\,\partial_{u}\log\theta_{a_j}(u_i)\,dt.
\tag{B.6.4}
$$

$$
\partial_{\sigma_{\perp,j}}\ \text{and}\ \partial_{\sigma_{\parallel,j}}:\ \text{via } a_j,\kappa_j,\gamma_j\ \text{and (B.6.1)–(B.6.2)}.
\tag{B.6.5}
$$

Angular factor derivatives: from (B.0.4) explicitly; gradient w\.r.t. $b_j$ additionally via $\partial b_j/\partial d_j=\frac{1}{\|d_j\|}(I-b_j b_j^\top)$.

## B.7 Algorithm (single batch, single chunk)

**Inputs:** $z\in\mathbb{R}^{B\times N}$, $\{z_j,b_j,\sigma_{\parallel,\perp,j},\alpha_j,\hat T_j\}_{j=1}^{M_c}$.
**Parameters:** $K\in\mathbb{N}$, $\varepsilon_{\mathrm{tot}}$, theta policy $\in\{\mathrm{direct},\mathrm{poisson},\mathrm{auto}\}$.
**Outputs:** $T\in\mathbb{C}^{B\times S}$.

1. $\Delta z = \mathrm{wrap}(z[:,None,:]-z_j[None,:,:])\in\mathbb{R}^{B\times M_c\times N}$.
2. Precompute $a_j,\kappa_j,\gamma_j$ for $j=1..M_c$.
3. Gauss–Hermite nodes $\{\tau_k,w_k\}_{k=1}^K$.
4. For $k=1..K$:
   4.1. $t_{j,k}=\tau_k/\sqrt{\gamma_j}$.
   4.2. $u_{b,j,i,k}=\Delta z_{b,j,i}-\kappa_j\,t_{j,k}\,b_{j,i}$.
   4.3. $\ell_{b,j,k}=\sum_{i=1}^N \log\theta_{a_j}(u_{b,j,i,k})$ (log-domain).
   4.4. $v_{b,j,k}= \exp(\ell_{b,j,k})$.
5. $\hat\eta_{b,j}=\tfrac{1}{\sqrt{\pi}}\sum_{k=1}^K w_k\,v_{b,j,k}$.
6. $\mathrm{ang}_{b,j}=\mathrm{ang}_j(d_b,b_j)$ via (B.0.4).
7. Accumulate $T_{b,:} \mathrel{+}= \sum_{j=1}^{M_c} \alpha_j\,(\hat\eta_{b,j}\,\mathrm{ang}_{b,j})\,\hat T_j[:]$.

**Complexity:** $O(B\,M_c\,K\,N + B\,M_c\,S)$ flops.
**Memory (peak):** $O(B\,M_c\,N)$ reals + transient $O(B\,M_c\,K)$.

## B.8 Numerical policies

– **Theta policy:** choose per $a_j$: direct if $a_j\ge a_{\mathrm{thr}}$ else Poisson; $a_{\mathrm{thr}}\sim 1$ (tunable).
– **Truncation:** pick $W$ or $K'$ from (B.4.1)–(B.4.2) to meet (B.4.3).
– **Stability:** accumulate $\sum_i \log\theta$; across $k$ use log-sum-exp if needed.
– **Precision:** FP32 (or TF32) for main path; optionally FP16/bf16 for $\theta$ with compensated summation.

## B.9 Parameter selection for $\varepsilon_{\mathrm{tot}}=10^{-3}$

Let $K\in\{12,16\}\Rightarrow \varepsilon_{\mathrm{quad}}\ll 10^{-4}$.
Per-coordinate theta budget $\varepsilon_{\theta}= \frac{10^{-3}}{2 N K}$.

**Direct form:** choose $W$ minimal s.t.

$$
\frac{2}{\sqrt{\pi a_j}}\int_{W-1}^{\infty} e^{-\pi a_j t^2}dt \ \le\ \varepsilon_{\theta}.
\tag{B.9.1}
$$

Approximation:

$$
W \ \approx\ \sqrt{\frac{1}{\pi a_j}\,\log\!\Big(\frac{1}{\varepsilon_{\theta}}\Big)}\ +\ 1.
\tag{B.9.2}
$$

**Poisson form:** choose $K'$ minimal s.t.

$$
\frac{2}{\sqrt{a_j}}\sum_{k=K'+1}^\infty e^{-\pi k^2/a_j}\ \le\ \varepsilon_{\theta}
\ \Rightarrow\
K' \ \approx\ \sqrt{\frac{a_j}{\pi}\,\log\!\Big(\frac{1}{\varepsilon_{\theta}}\Big)}.
\tag{B.9.3}
$$

## B.10 Special cases

– Isotropic case $\sigma_{\parallel,j}=\sigma_{\perp,j}\Rightarrow \gamma_j=1,\ \kappa_j=0$:
$\eta_j(\Delta z)=\prod_{i=1}^{N}\theta_{a_j}(\Delta z_i)$ (no quadrature).
– Large-$N$: linear scaling in $N$; accuracy via (B.9.1)–(B.9.3).

## B.11 Output assembly

For a full set $j=1..M$, tile $M$ into chunks $M_c$, apply B.7, accumulate $T$.
Overall complexity $O(B\,M\,K\,N + B\,M\,S)$.
Differentiable w\.r.t. $(z,b_j,\sigma_{\parallel,\perp,j},\alpha_j,\hat T_j,d)$ per B.6.
