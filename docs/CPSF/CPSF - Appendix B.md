# Appendix B — Zero-Frame, Non-Periodized Field $T_{\mathrm{Zero}}$

## B.0 Data and constraints

Let $N\ge 2$. For each contributor $j$ we are given
$$
z\in\mathbb C^{N},\quad z_j\in\mathbb C^{N},\quad
\vec d\in\mathbb C^{N},\quad \vec d_j\in\mathbb C^{N},\quad
\alpha_j\in\mathbb R,\quad
\sigma_{\parallel j},\sigma_{\perp j}\in\mathbb R_{>0},\quad
\hat T_j\in\mathbb C^{S}.
$$

As in the CPSF canon, the lift is the identity, so $\tilde z=z$ and
$$
\delta z:=\tilde z-\tilde z_j=z-z_j.
$$
Directions are unit-norm in CPSF (the frame construction $R(\vec d)$ assumes $|\vec d|=1$), hence we treat $\vec d_j$ as normalized.

Define the Hermitian inner product $\langle u,v\rangle=\sum_k \overline{u_k},v_k$ and the directional tangent
$$
\delta\vec d := \mathrm{delta_vec_d}(\vec d,\vec d_j),
$$
which is orthogonal to $\vec d_j$ (tangency: $\langle \vec d_j,\delta\vec d\rangle=0$).

## B.1 Definition

The **zero-frame, non-periodized** contribution of index $j$ is
$$
\eta^{(0)}_j(z,\vec d)
=
\exp!\bigl(-\pi,q^{(\mathrm{pos})}_j(\delta z)\bigr);
\exp!\bigl(-\pi,q^{(\mathrm{dir})}_j(\delta\vec d)\bigr),
$$
with
$$
q^{(\mathrm{pos})}_j(\delta z)
=

\sigma_{\perp j}^{-1},|\delta z|_2^2

* \bigl(\sigma_{\parallel j}^{-1}-\sigma_{\perp j}^{-1}\bigr),
  \bigl|\langle \vec d_j,\delta z\rangle\bigr|^2,
  $$
  $$
  q^{(\mathrm{dir})}*j(\delta\vec d)
  =
  \sigma*{\perp j}^{-1},|\delta\vec d|_2^2
  \quad\text{(since $\delta\vec d\perp\vec d_j$).}
  $$

The field is the real-gain combination of the semantic vectors:
$$
T_{\mathrm{Zero}}(z,\vec d)=\sum_j \bigl(\alpha_j,\eta^{(0)}_j(z,\vec d)\bigr),\hat T_j
;\in;\mathbb C^{S}.
$$

This is exactly the algorithm implemented in $T_{\mathrm{Zero}}$ (elementwise exponentials of the two quadratic forms, followed by a weighted sum over $j$).

## B.2 Derivation from CPSF canon (zero-cell reduction)

Let $R(\vec d_j)\in \mathrm U(N)$ and let $b:=\vec d_j$ denote its first column (alignment). The CPSF covariance block in position is
$$
S_{0j} ;=; \sigma_{\perp j},I_N + (\sigma_{\parallel j}-\sigma_{\perp j}),b,b^{\dagger},
$$
and its inverse satisfies
$$
S_{0j}^{-1} ;=; \sigma_{\perp j}^{-1},I_N + (\sigma_{\parallel j}^{-1}-\sigma_{\perp j}^{-1}),b,b^{\dagger}.
$$
Hence
$$
\langle S_{0j}^{-1}\delta z,\delta z\rangle
= \sigma_{\perp j}^{-1}|\delta z|^2
+(\sigma_{\parallel j}^{-1}-\sigma_{\perp j}^{-1}),|\langle b,\delta z\rangle|^2,
$$
giving the stated $q^{(\mathrm{pos})}_j$.

For the directional factor, in the full CPSF quadratic form $q(w)$ with $w=[u;v]$, taking $u=0$ and $v=\delta\vec d$ yields
$$
q^{(\mathrm{dir})}*j(\delta\vec d)
= \frac{|\langle b,\delta\vec d\rangle|^2}{\sigma*{\parallel j}}

* \frac{|\delta\vec d|^2-|\langle b,\delta\vec d\rangle|^2}{\sigma_{\perp j}}.
  $$
  By tangency, $\langle b,\delta\vec d\rangle=0$, so
  $$
  q^{(\mathrm{dir})}*j=\sigma*{\perp j}^{-1}|\delta\vec d|^2,
  $$
  exactly as used above.

Thus $\eta^{(0)}_j=\exp(-\pi,q^{(\mathrm{pos})}_j)\exp(-\pi,q^{(\mathrm{dir})}*j)$ is the **single-cell (no lattice)** specialization of the canonical CPSF envelope; $T*{\mathrm{Zero}}$ implements this product and sums $\alpha_j\eta^{(0)}_j\hat T_j$.

## B.3 Properties (inherited from CPSF)

* **No periodization.** No $\mathbb Z^{2N}$ sum is taken; only the zero cell contributes (your “zero torus frame”).
* **Directional tangency.** Because $\delta\vec d\perp \vec d_j$, the directional quadratic uses only $\sigma_{\perp j}$.
* **Real, positive gain.** $\eta^{(0)}*j>0$, hence $T*{\mathrm{Zero}}$ multiplies $\hat T_j$ by a real scalar and sums (no extra $\operatorname{Re}$ is needed here).
* **Lift compatibility.** With the canonical identity lift, $\delta z=z-z_j$. If a non-trivial lift is introduced later, this definition remains valid with $\delta z=\tilde z-\tilde z_j$.
