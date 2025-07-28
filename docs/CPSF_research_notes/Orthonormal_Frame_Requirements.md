# Formal Requirements for Orthonormal Frames in CPSF

This document formalizes the strict mathematical conditions required for the orthonormal frame construction \( R(\vec{d}) \in \mathrm{U}(N) \) in the CPSF (Continuous Projective Semantic Fields) architecture. These requirements are necessary and sufficient to guarantee that all CPSF analytic and functional components behave as defined.

---

## 1. Domain and Codomain

Let \( \vec{d} \in \mathbb{S}^{2N-1}_{\text{unit}} := \{ \vec{d} \in \mathbb{C}^N : \|\vec{d}\| = 1 \} \). Then:

\[ R : \mathbb{S}^{2N-1}_{\text{unit}} \to \mathrm{U}(N) \]

---

## 2. Unitarity

\[ R(\vec{d})^\dagger R(\vec{d}) = I_N \quad \text{for all } \vec{d} \in \mathbb{S}^{2N-1}_{\text{unit}} \]

This guarantees that \( R(\vec{d}) \) preserves Hermitian inner products and norms.

---

## 3. Alignment Constraint

\[ R(\vec{d}) e_1 = \vec{d} \quad \text{where } e_1 = (1,0,\dots,0)^T \in \mathbb{C}^N \]

This ensures that the first column of \( R(\vec{d}) \) coincides with the projection direction \( \vec{d} \).

---

## 4. Orthogonality of Complement Columns

Let \( R(\vec{d}) = [\vec{d}, v_2, \dots, v_N] \). Then:

- \( \langle v_j, \vec{d} \rangle = 0 \) for all \( j = 2, \dots, N \);
- \( \langle v_j, v_k \rangle = \delta_{jk} \) for all \( j,k = 2, \dots, N \).

That is, \( \{v_2, \dots, v_N\} \) form an orthonormal basis of \( \vec{d}^\perp \subset \mathbb{C}^N \).

---

## 5. Smoothness

The map \( \vec{d} \mapsto R(\vec{d}) \in \mathrm{U}(N) \) is required to be \( C^\infty \)-smooth on \( \mathbb{S}^{2N-1}_{\text{unit}} \), either globally or via a smooth local trivialization with transition functions in \( \mathrm{U}(N-1) \).

This guarantees:
- \( \Sigma_j(\vec{d}) \in C^\infty \);
- \( \psi_j^{\mathbb{T}}(z, \vec{d}) \in C^\infty \);
- \( \Delta \hat{T}_j \in C^\infty \);
- and uniform convergence of all directional derivatives of \( T(z, \vec{d}) \).

---

## 6. Invariance under \( \mathrm{U}(N-1) \) Action

If \( Q \in \mathrm{U}(N-1) \), define:

\[
R'(\vec{d}) := R(\vec{d}) \cdot \begin{bmatrix} 1 & 0 \\ 0 & Q \end{bmatrix}
\]

Then:

\[ \Sigma_j(R'(\vec{d})) = \Sigma_j(R(\vec{d})) \]

This shows that the geometric covariance structure is invariant under any change of orthonormal basis in the subspace orthogonal to \( \vec{d} \), which implies that \( \psi_j^{\mathbb{T}} \), \( \Delta \hat{T}_j \), and \( T(z, \vec{d}) \) remain invariant.

---

## 7. Extended Frame Compatibility

Define:
\[ \mathcal{R}(\vec{d}) := \mathrm{diag}(R(\vec{d}), R(\vec{d})) \in \mathrm{U}(2N) \]

The same frame \( R(\vec{d}) \) is used for both position and direction components in \( \mathbb{C}^{2N} = \mathbb{C}_{\text{pos}}^N \oplus \mathbb{C}_{\text{dir}}^N \).

The diagonal structure ensures consistent alignment and covariance computation for the combined spatial-directional Gaussian envelope, as used in \( \Sigma_j \).

---

## 8. Local Trivialization (Topological Requirement)

A global smooth frame field \( R(\vec{d}) \) over \( \mathbb{S}^{2N-1} \) may not exist due to topological obstructions (e.g., non-parallelizability). Therefore:

The frame must be defined locally via an open cover \( \{U_\alpha\} \) with smooth local sections:

\[ R_\alpha : U_\alpha \to \mathrm{U}(N), \quad R_\alpha e_1 = \vec{d}, \quad \vec{d} \in U_\alpha \]

On overlaps \( U_\alpha \cap U_\beta \):

\[ R_\alpha(\vec{d}) = R_\beta(\vec{d}) \cdot \begin{bmatrix} 1 & 0 \\ 0 & Q_{\alpha\beta}(\vec{d}) \end{bmatrix}, \quad Q_{\alpha\beta} \in \mathrm{U}(N-1) \]

This ensures the transition compatibility of local frames and defines a smooth \( \mathrm{U}(N-1) \)-bundle structure.

---

## 9. Compatibility with CPSF Dynamics

All derivatives of \( R(\vec{d}) \) must be bounded and smooth in \( \vec{d} \), ensuring:

- \( C^\infty \)-dependence of \( \delta \vec{d} \in T_{\vec{d}_j} \mathbb{S}^{2N-1} \);
- differentiability of the embedding \( w := \iota(\tilde{z} - \tilde{z}_j, \delta \vec{d}) \);
- exponential decay and regularity of \( \rho_j(w) \);
- smooth convergence of the periodized sum \( \psi_j^{\mathbb{T}}(z, \vec{d}) \);
- and well-definedness of the semantic projection \( \Delta \hat{T}_j \).

This condition is essential for the variational structure of CPSF.

---

## Conclusion

The orthonormal frame \( R(\vec{d}) \) must satisfy all nine conditions above to guarantee the analytic and functional correctness of CPSF. These constraints ensure:

- Localized Gaussian envelopes \( \rho_j(w) \) and \( \psi_j^{\mathbb{T}} \) with smooth dependence on all inputs;
- Consistent semantic projection via \( \Delta \hat{T}_j \);
- Invariance of geometric behavior under internal frame rotations;
- Differentiable structure over \( \mathbb{T}_\mathbb{C}^N \times \mathbb{S}^{2N-1} \).

Violation of any condition may result in loss of differentiability, breakdown of localization, or invalid update dynamics.

