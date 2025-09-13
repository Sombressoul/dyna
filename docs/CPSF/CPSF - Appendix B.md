# Appendix B — HS-Θ (Hubbard–Stratonovich + θ-sums) for rank-1 anisotropic periodic Gaussians

This appendix documents the algorithm implemented by `T_HS_theta(...)`. Notation matches the code.

## B.1. Data, shapes, and basic notation

* Inputs (devices/dtypes as in code):

  * $z \in \mathbb{C}^{B\times N}$,
  * $z_j \in \mathbb{C}^{M\times N}$,
  * $\mathrm{vec\_d} \in \mathbb{C}^{B\times N}$,
  * $\mathrm{vec\_d\_j} \in \mathbb{C}^{M\times N}$,
  * $\widehat T_j \in \mathbb{C}^{M\times S}$,
  * $\alpha_j \in \mathbb{R}^{M}$,
  * $\sigma_{\parallel},\sigma_{\perp}\in\mathbb{R}^M_{>0}$ (component-wise for $j=1,\dots,M$).

* Output: $T(z,\mathrm{vec\_d}) \in \mathbb{C}^{B\times S}$.

* Toroidal wrap (component-wise):

$$
\mathrm{wrap}(u) \;:=\; u-\mathrm{round}(u) \;\in [-\tfrac12,\tfrac12).
$$

* Directional difference on the torus:

$$
\Delta_d(\mathrm{vec\_d},\mathrm{vec\_d\_j})\in\mathbb{C}^{B\times M\times N},
$$

implemented by `delta_vec_d` (fallback: plain difference if unavailable).

* Unit complex direction for each contribution $j$:

$$
b_j \;:=\; \frac{\mathrm{vec\_d\_j}[j,:]}{\|\mathrm{vec\_d\_j}[j,:]\|_2}
\;\in\;\mathbb{C}^N,
\quad
b_j = b_{R,j} + i\,b_{I,j},
\quad
\|b_j\|_2=1.
$$

* Anisotropy scalars (per $j$):

$$
a_j := \frac{1}{\sigma_{\perp,j}},\qquad
\gamma_j := \frac{\sigma_{\perp,j}}{\sigma_{\parallel,j}}\in(0,1],\qquad
\kappa_j^2 := \frac{(\sigma_{\parallel,j}-\sigma_{\perp,j})\,\sigma_{\perp,j}}{\pi\,\sigma_{\parallel,j}}\;\ge 0.
$$

Scaled quantities used in code:

$$
\tilde\kappa_j := \frac{\kappa_j}{\sqrt{\gamma_j}},\qquad
\mathrm{norm\_fac}_j := \frac{1}{\pi\sqrt{\gamma_j}}.
$$

* Gauss–Hermite (1D) nodes and weights (for $\int_{\mathbb{R}} e^{-\tau^2}f(\tau)\,d\tau$):
  $\{\tau_q,w_q\}_{q=1}^{Q}$, with $Q=$ `quad_nodes`. Tensor grid $\mathcal{Q}=\{(\tau_p,\tau_q)\}_{p,q=1}^Q$, weights $w_{pq}=w_p w_q$.

## B.2. HS factorization and coordinate separation

For each contribution $j$ and quadrature node $(\tau_1,\tau_2)$, define per-coordinate shifted phases

$$
u_{n}^{(j)}(\tau_1,\tau_2)
\;:=\;
\mathrm{wrap}\!\left(\operatorname{Re}(z-z_j)_{n} \;-\; \tilde\kappa_j\big(\tau_1\,b_{R,j,n} + \tau_2\,b_{I,j,n}\big)\right),
\quad n=1,\dots,N.
$$

Then the HS-separated factor for fixed $(\tau_1,\tau_2)$ is the product of 1D periodic Gaussians:

$$
\eta_j(z\mid \tau_1,\tau_2)
\;=\;
\prod_{n=1}^{N}\theta_{a_j}\!\left(u_{n}^{(j)}(\tau_1,\tau_2)\right).
$$

The full $\eta_j$ combines quadrature:

$$
\boxed{\quad
\eta_j(z)\;=\;\mathrm{norm\_fac}_j \sum_{(\tau_1,\tau_2)\in\mathcal{Q}}
w(\tau_1)w(\tau_2)\;\prod_{n=1}^{N}\theta_{a_j}\!\big(u_{n}^{(j)}(\tau_1,\tau_2)\big)
\quad}
$$

with accumulation in the log-domain (Sec. B.5).

An independent “angular” factor, identical to `Tau_dual`, is

$$
\boxed{\;
\mathrm{ang}_j(z,\mathrm{vec\_d})
=
\exp\!\Big(
-\pi\Big[
\sigma_{\perp,j}^{-1}\,\|\Delta_d\|_2^2 \;-\;
\frac{\sigma_{\parallel,j}-\sigma_{\perp,j}}{\sigma_{\parallel,j}\sigma_{\perp,j}}\,
\big|\langle b_j,\Delta_d\rangle\big|^2
\Big]\Big),
\;}
$$

where $\Delta_d=\Delta_d(\mathrm{vec\_d},\mathrm{vec\_d\_j}[j,:])\in\mathbb{C}^{B\times N}$.

## B.3. Periodic Gaussian $\theta_a(u)$ and truncations

For $u\in[-\tfrac12,\tfrac12)$, $a>0$:

**Direct form (lattice in real space):**

$$
\theta_a(u)=\sum_{n\in\mathbb{Z}} e^{-\pi a (u+n)^2}
\;\approx\;
\sum_{|n|\le W} e^{-\pi a (u+n)^2}.
$$

**Poisson form (via Poisson summation):**

$$
\theta_a(u)
=
\frac{1}{\sqrt{a}}+\frac{2}{\sqrt{a}}\sum_{k=1}^{\infty} e^{-\pi k^2/a}\cos(2\pi k u)
\;\approx\;
\frac{1}{\sqrt{a}}+\frac{2}{\sqrt{a}}\sum_{k=1}^{K} e^{-\pi k^2/a}\cos(2\pi k u).
$$

**Form and radii selection.** With a per-$\theta$ error budget

$$
\varepsilon_\theta := \frac{\varepsilon_{\mathrm{tot}}}{2\,N\,Q^2},
$$

the code uses

$$
W \;\approx\; \Big\lceil \sqrt{\frac{-\log \varepsilon_\theta}{\pi a}}\Big\rceil + 1,\qquad
K \;\approx\; \Big\lceil \sqrt{\frac{a}{\pi}(-\log \varepsilon_\theta)}\Big\rceil.
$$

Mode:

$$
\texttt{auto}:~ \text{Direct if } a\ge a_{\mathrm{thr}},~ \text{Poisson otherwise};\quad
\texttt{direct}/\texttt{poisson}:~ forced.
$$

## B.4. Quadrature and log-domain accumulation

Let $L_{j}(\tau_1,\tau_2):=\sum_{n=1}^N \log \theta_{a_j}(u_n^{(j)}(\tau_1,\tau_2))$.
Then

$$
\eta_j(z)=\mathrm{norm\_fac}_j \sum_{(\tau_1,\tau_2)\in\mathcal{Q}} \exp\!\big(L_j(\tau_1,\tau_2)+\log w(\tau_1) + \log w(\tau_2)\big).
$$

Numerically:

* the sum over $n$ is carried as $\sum \log(\cdot)$ in chunks of coordinates;
* the sum over $(\tau_1,\tau_2)$ is accumulated with online log-sum-exp (Sec. B.5).

## B.5. Streaming accumulators and chunking

Chunking parameters (as in code): `m_chunk` over contributions $j$, `n_chunk` over coordinates $n$, `q_chunk` over GH-node pairs $(\tau_1,\tau_2)$. `q_chunk` is chosen adaptively from available memory.

**Online LSE over a chunk $\mathcal{Q}_{\mathrm{chunk}}$.** Maintain $(A_j,S_j)$ such that

$$
\sum_{(\tau_1,\tau_2)\in \mathcal{Q}_{\le}} e^{L_j(\tau_1,\tau_2)}
= e^{A_j} S_j.
$$

For a new chunk with values $\{L_j^{(q)}\}_{q\in\mathcal{Q}_{\mathrm{chunk}}}$,

$$
A_j^{\mathrm{new}} = \max\big(A_j,\max_{q} L_j^{(q)}\big),\quad
S_j^{\mathrm{new}} = S_j\,e^{A_j-A_j^{\mathrm{new}}} + \sum_{q} e^{L_j^{(q)}-A_j^{\mathrm{new}}}.
$$

At the end,

$$
\eta_j(z) = \mathrm{norm\_fac}_j \, e^{A_j} S_j.
$$

**Direct/Poisson partition.** In `auto`, contributions split into $\,J_{\mathrm{dir}}=\{j:a_j\ge a_{\mathrm{thr}}\}$ and $J_{\mathrm{poi}}=\{j:a_j< a_{\mathrm{thr}}\}$. For each $q$-chunk, compute $\theta_a$ **separately** on these subsets:

* Direct (stream over offsets $n\in[-W_j,W_j]$; mask $|n|\le W_j$ per $j$):

$$
\theta_{a_j}(u) \approx \sum_{|n|\le W_j} e^{-\pi a_j (u+n)^2}.
$$

* Poisson (stream over harmonics $k=1..K_j$; $k=0$ term $1/\sqrt{a_j}$ added once):

$$
\theta_{a_j}(u) \approx \frac{1}{\sqrt{a_j}} + 
\frac{2}{\sqrt{a_j}}\sum_{k=1}^{K_j} e^{-\pi k^2/a_j}\cos(2\pi k u).
$$

Per-coordinate products $\prod_n \theta_{a_j}(\cdot)$ are accumulated as $\sum_n \log \theta_{a_j}(\cdot)$. Before $\log$, values are clamped below by machine-`tiny`.

## B.6. Angular factor and spectral aggregation

For each $j$,

$$
\mathrm{ang}_j(z,\mathrm{vec\_d}) = \exp\!\Big(
-\pi\big[\sigma_{\perp,j}^{-1}\,\|\Delta_d\|_2^2
- \tfrac{\sigma_{\parallel,j}-\sigma_{\perp,j}}{\sigma_{\parallel,j}\sigma_{\perp,j}}
|\langle b_j,\Delta_d\rangle|^2\big]\Big),
$$

$$
\eta_j(z) = \mathrm{norm\_fac}_j \, e^{A_j} S_j.
$$

Define weights $w_j(z):=\alpha_j\,\mathrm{ang}_j(z,\mathrm{vec\_d})\,\eta_j(z)\in\mathbb{R}^{B}$. The final field is

$$
\boxed{\quad
T(z,\mathrm{vec\_d})
\;=\;
\sum_{j=1}^{M} w_j(z)\;\widehat T_j
\;\in\;\mathbb{C}^{B\times S}.
\quad}
$$

## B.7. Numerical details

* **Log domain:** products in $n$ → sums of logs; quadrature sum → online LSE accumulators $(A_j,S_j)$.
* **Truncation:** $W_j,K_j$ from $\varepsilon_\theta=\varepsilon_{\mathrm{tot}}/(2NQ^2)$ (B.3).
* **Mode selection:** `poisson` is preferable for small $a_j$ (large $\sigma_{\perp,j}$); `direct` for large $a_j$. `auto` uses the threshold $a_{\mathrm{thr}}$.
* **Memory:** no materialization of $(B,mc,Nc,qc,K)$ / $(B,mc,Nc,qc,\mathrm{No})$ / $(B,mc,Q^2)$; only streaming buffers of shape $(B,mc,Nc,qc)$. `q_chunk` is set adaptively from free memory (with min/max caps).
* **Safeguards:** all $\theta_a$ are clamped below by dtype-dependent `tiny` before $\log$.

## B.8. Pseudocode (one $m$-chunk)

```
Input: z(B,N), z_j(mc,N), vec_d(B,N), vec_d_j(mc,N), T_hat_j(mc,S),
       alpha_j(mc), σ∥(mc), σ⊥(mc), Q, eps_tot, a_thr, n_chunk
Output: ΔT(B,S)

Precompute per j:
  a=1/σ⊥, γ=σ⊥/σ∥, κ^2=((σ∥−σ⊥)σ⊥)/(πσ∥), κ_eff=κ/√γ, norm_fac=1/(π√γ)
  choose mode_j ∈ {Direct, Poisson}; compute W_j or K_j from eps_tot/(2 N Q^2)

dz = wrap( Re(z) − Re(z_j) ) ∈ ℝ^{B×mc×N}
ang_fac(B,mc) from Δd(vec_d, vec_d_j)  # same as Tau_dual

A_lse(B,mc) = −∞ ; S_lse(B,mc) = 0

for (τ1,τ2) in GH grid, in q-chunks:
  for n in [1..N] in n-chunks:
    for j in J_dir:                # a_j ≥ a_thr
      u_d(B,|J_dir|,Nc,qc) = dz − κ_eff(j)*(τ1 b_R + τ2 b_I)
      θ_direct via streaming over offsets n∈[−W_j..W_j] (mask |n|≤W_j), keep axis qc
      accumulate log over coordinates Nc
    for j in J_poi:                # a_j < a_thr
      u_p(B,|J_poi|,Nc,qc) = dz − κ_eff(j)*(τ1 b_R + τ2 b_I)
      θ_poisson = 1/√a_j + streaming sum k=1..K_j of exp(−πk^2/a_j) cos(2πk u_p)
      accumulate log over coordinates Nc

  L_dir = log_acc_dir + log w(τ1) + log w(τ2)  # shape (B,|J_dir|)
  L_poi = log_acc_poi + log w(τ1) + log w(τ2)  # shape (B,|J_poi|)
  online-LSE merge (A_lse,S_lse) with L_dir and L_poi

eta(B,mc) = norm_fac · exp(A_lse) · S_lse
ΔT(B,S) += (alpha(mc) ⊙ ang_fac(B,mc) ⊙ eta(B,mc)) · T_hat_j(mc,S)
```
