#  CPSF — Appendix A: Poisson–dual computation of the CPSF field

## A.0. Data, operators, constraints

Let $N\ge 2$. For each index $j$ the data are

$$
z\in\mathbb C^N,\quad z_j\in\mathbb C^N,\quad \vec d\in\mathbb C^N,\quad \vec d_j\in\mathbb C^N,\quad
\alpha_j\in\mathbb R_{>0},\quad \sigma_{\parallel j},\sigma_{\perp j}\in\mathbb R_{>0},\quad \hat T_j\in\mathbb C^S.
$$

Define $\delta z:=z-z_j$, $\delta\vec d:=\mathrm{delta\_vec\_d}(\vec d,\vec d_j)$. Let $R(\vec d_j)\in\mathrm U(N)$ and
$\mathcal R(\vec d_j):=\mathrm{diag}(R(\vec d_j),R(\vec d_j))$.
Let

$$
D_j=\mathrm{diag}(\sigma_{\parallel j},\sigma_{\perp j},\dots;\ \sigma_{\parallel j},\sigma_{\perp j},\dots),
\quad
\Sigma_j=\mathcal R(\vec d_j)^\dagger D_j\,\mathcal R(\vec d_j),
$$

and for $w\in\mathbb C^{2N}$ set $q_j(w)=\langle \Sigma_j^{-1}w,w\rangle$, $\rho_j(q)=e^{-\pi q}$.
Only the positional block is periodized. The periodization lattice is $\mathbb Z^{2N}$ acting on $\delta z$ by complex–componentwise integer shifts; $\delta\vec d$ is not periodized.&#x20;

Denote the complex concatenation $\iota(u,v)=[u;v]\in\mathbb C^{2N}$.&#x20;

## A.1. Canonical field and block factorization

The canonical (toroidal) contribution is

$$
\eta_j(z,\vec d)=\sum_{n\in\mathbb Z^{2N}}\rho_j\!\Big(q_j\big(\iota(\delta z+n,\ \delta\vec d)\big)\Big).
\tag{A.1}
$$

Since $\Sigma_j^{-1}$ is block–diagonal, $\Sigma_j^{-1}=\mathrm{diag}(A^{(\mathrm{pos})}_j,A^{(\mathrm{dir})}_j)$ with

$$
A^{(\mathrm{pos})}_j=A^{(\mathrm{dir})}_j
=R(\vec d_j)\,\mathrm{diag}(\sigma_{\parallel j}^{-1},\sigma_{\perp j}^{-1},\dots)\,R(\vec d_j)^\dagger,
$$

one has

$$
q_j\big(\iota(\delta z+n,\ \delta\vec d)\big)
=(\delta z+n)^\dagger A^{(\mathrm{pos})}_j(\delta z+n)+\delta\vec d^\dagger A^{(\mathrm{dir})}_j\,\delta\vec d,
$$

and hence the exact factorization

$$
\eta_j
=
\underbrace{e^{-\pi\,\delta\vec d^\dagger A^{(\mathrm{dir})}_j\,\delta\vec d}}_{C^{(\mathrm{dir})}_j}
\cdot
\underbrace{\sum_{n\in\mathbb Z^{2N}} e^{-\pi\,(\delta z+n)^\dagger A^{(\mathrm{pos})}_j(\delta z+n)}}_{\Theta^{(\mathrm{pos})}_j}.
\tag{A.2}
$$



The CPSF field is

$$
T(z,\vec d)=\sum_j\big(\alpha_j\,\mathrm{Re}\,\eta_j(z,\vec d)\big)\,\hat T_j.
\tag{A.3}
$$



## A.2. Poisson–dual representation of the positional sum

For $d=2N$ and any $t>0$ the Poisson identity yields

$$
\Theta^{(\mathrm{pos})}_j
=\frac{1}{t^{d/2}\sqrt{\det A^{(\mathrm{pos})}_j}}\;
\sum_{k\in\mathbb Z^{2N}}
\exp\!\Big(-\tfrac{\pi}{t}\,k^\top (A^{(\mathrm{pos})}_j)^{-1}k\Big)\,
e^{\,2\pi i\,k\cdot b_{\mathrm{pos}}},
\quad
b_{\mathrm{pos}}:=\mathrm{frac}(\delta z),
\tag{A.4}
$$

where $\mathrm{frac}(\cdot)$ is the componentwise reduction to the unit torus. Equivalently, the real–space form is

$$
\Theta^{(\mathrm{pos})}_j
=\sum_{n\in\mathbb Z^{2N}} e^{-\pi\,t\,(\delta z+n)^\dagger A^{(\mathrm{pos})}_j(\delta z+n)}.
\tag{A.5}
$$

The two series are equal for all $t>0$.&#x20;

In the CPSF implementation, the dual quadratic form and phase are realized via

$$
y=R(\vec d_j)^\dagger k\in\mathbb C^N,\qquad
\mathsf{quad}_k=\sigma_{\perp j}\sum_{i=1}^N |y_i|^2+(\sigma_{\parallel j}-\sigma_{\perp j})\,|y_1|^2,
$$

$$
\mathrm{phase}(k)=\exp\!\big(2\pi i\,k\cdot \mathrm{frac}(\delta z)\big),
$$

so that (A.4) is computed as

$$
\Theta^{(\mathrm{pos})}_j
=\Big(\sigma_{\parallel j}\,\sigma_{\perp j}^{\,N-1}\Big)\,t^{-N}
\sum_{k\in\mathbb Z^{2N}} \exp\!\big(-(\pi/t)\,\mathsf{quad}_k\big)\,\mathrm{phase}(k),
\tag{A.6}
$$

using the complex–$N$ convention $\big(\det_{\mathbb R}A^{(\mathrm{pos})}_j\big)^{-1/2}=\det_{\mathbb C}\Sigma^{(\mathrm{pos})}_j=\sigma_{\parallel j}\sigma_{\perp j}^{N-1}$.&#x20;

## A.3. Directional factor

The non–periodized directional factor is

$$
C^{(\mathrm{dir})}_j
=\exp\!\Big(-\pi\,q_j\big(\iota(0,\delta\vec d)\big)\Big)
=\exp\!\big(-\pi\,\delta\vec d^\dagger A^{(\mathrm{dir})}_j\,\delta\vec d\big).
\tag{A.7}
$$

This factor has no lattice sum.&#x20;

## A.4. Final Poisson–dual formula for $\eta_j$ and $T$

Combining (A.2), (A.6), (A.7),

$$
\eta_j(z,\vec d)
=
\bigg[\exp\!\big(-\pi\,\delta\vec d^\dagger A^{(\mathrm{dir})}_j\,\delta\vec d\big)\bigg]\,
\bigg[\Big(\sigma_{\parallel j}\,\sigma_{\perp j}^{\,N-1}\Big)\,t^{-N}
\sum_{k\in\mathbb Z^{2N}} e^{-(\pi/t)\,\mathsf{quad}_k}\,e^{2\pi i\,k\cdot \mathrm{frac}(\delta z)}\bigg],
\tag{A.8}
$$

and

$$
T(z,\vec d)=\sum_j\big(\alpha_j\,\mathrm{Re}\,\eta_j(z,\vec d)\big)\,\hat T_j.
\tag{A.9}
$$

Equations (A.8)–(A.9) are the source–of–truth for the procedures `T_PD_window` and `T_PD_full`.&#x20;

## A.5. Discrete truncation bounds (L∞ windows)

Let $d=2N$, $A_d(m)=(2m{+}1)^d-(2m{-}1)^d$. For a dual L∞ window $\|k\|_\infty\le L$,

$$
\sum_{\|k\|_\infty>L}\exp\!\Big(-\tfrac{\pi}{t}\,k^\top (A^{(\mathrm{pos})}_j)^{-1}k\Big)
\ \le\
\sum_{m=L+1}^\infty A_{d}(m)\,\exp\!\Big(-\tfrac{\pi}{t}\,\sigma_{\min j}\,m^2\Big),
\quad \sigma_{\min j}=\min(\sigma_{\parallel j},\sigma_{\perp j}).
\tag{A.10}
$$

For a real L∞ window $\|n\|_\infty\le L$,

$$
\sum_{\|n\|_\infty>L} e^{-\pi\,t\,(\delta z+n)^\dagger A^{(\mathrm{pos})}_j(\delta z+n)}
\ \le\
\sum_{m=L+1}^\infty A_{d}(m)\,\exp\!\Big(-\pi\,t\,\tfrac{m^2}{\sigma_{\max j}}\Big),
\quad \sigma_{\max j}=\max(\sigma_{\parallel j},\sigma_{\perp j}).
\tag{A.11}
$$

These bounds control truncation errors for either representation; no lattice appears in the directional factor.&#x20;

## A.6. Invariance properties

(i) Torus periodicity: replacing $\delta z$ by $\delta z+n_0$, $n_0\in\mathbb Z^{2N}$, leaves $\eta_j$ invariant (phase $e^{2\pi i k\cdot n_0}\equiv 1$). 

(ii) No directional lattice: $C^{(\mathrm{dir})}_j$ depends solely on $\delta\vec d$. 

(iii) $t$–invariance for the infinite sums: (A.4)=(A.5) for all $t>0$.&#x20; 

## A.7. Algorithmic form (streamed over packs)

Let packs be $\{(\_,\_,\mathcal K_\nu)\}_\nu$ with $\mathcal K_\nu\subset\mathbb Z^{2N}$ finite. For each $\nu$,

$$
\Theta^{(\mathrm{pos})}_j[\nu]
=\Big(\sigma_{\parallel j}\,\sigma_{\perp j}^{\,N-1}\Big)\,t^{-N}
\sum_{k\in\mathcal K_\nu} e^{-(\pi/t)\,\mathsf{quad}_k}\,e^{2\pi i\,k\cdot \mathrm{frac}(\delta z)},
\quad
\eta_j[\nu]=C^{(\mathrm{dir})}_j\,\Theta^{(\mathrm{pos})}_j[\nu],
$$

$$
T_\nu=\sum_j \big(\alpha_j\,\mathrm{Re}\,\eta_j[\nu]\big)\,\hat T_j,\qquad
T=\sum_\nu T_\nu.
\tag{A.12}
$$

Optional early–stopping decisions are applied on $\{T_\nu\}$ via absolute/relative pack norms; the mathematical sum is order–invariant when no stopping is applied. This streaming realization corresponds to `T_PD_full`.&#x20;
