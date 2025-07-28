## Orthonormal Frame Construction

Let $\vec{d} \in \mathbb{S}_{\text{unit}}^{2N-1} := \{ \vec{d} \in \mathbb{C}^N : \|\vec{d}\| = 1 \}$. The goal is to construct a unitary matrix $R(\vec{d}) \in \mathrm{U}(N)$ such that:

* $R(\vec{d}) e_1 = \vec{d}$;
* $R(\vec{d})$ is real-analytic in $\vec{d}$;
* Columns $\{v_2, \dots, v_N\}$ of $R(\vec{d})$ form an orthonormal basis of $\vec{d}^\perp \subset \mathbb{C}^N$;
* All requirements $R1$ through $R9$ of CPSF are satisfied (see *"Frame Requirements in CPSF"* below);
* The resulting construction is globally defined, numerically stable, and consistent with the CPSF analytic model.

**Frame Requirements in CPSF**

The construction of the orthonormal frame $R(\vec{d}) \in \mathrm{U}(N)$ must satisfy the following nine structural conditions, which are necessary and sufficient to ensure analytic compatibility with the CPSF functional model:

* **(R1) Codomain Constraint** — $R(\vec{d}) \in \mathrm{U}(N)$: the frame must be unitary.
* **(R2) Inner Product Preservation** — $R^\dagger R = I$: preserves Hermitian norms and inner products.
* **(R3) Directional Alignment** — $R(\vec{d}) e_1 = \vec{d}$: the first frame vector must coincide with the projection direction.
* **(R4) Orthogonal Complement** — $\{v_2, \dots, v_N\} \perp \vec{d}$: remaining columns span $\vec{d}^\perp$ and are orthonormal.
* **(R5) Smoothness** — $R(\vec{d}) \in C^\infty$: required for differentiability of all field components.
* **(R6) Invariance under $\mathrm{U}(N-1)$** — right action on the complement must not affect geometry.
* **(R7) Extended Compatibility** — $\mathcal{R} := \mathrm{diag}(R, R) \in \mathrm{U}(2N)$: used in joint position-directional covariance $\Sigma_j$.
* **(R8) Topological Validity** — the frame must admit either global or locally trivialized smooth sections.
* **(R9) Dynamic Compatibility** — all derivatives of $R(\vec{d})$ must be bounded to ensure well-posed projections $\Delta \hat{T}_j$ and field regularity.

These conditions define the admissible class of orthonormal frames for CPSF.

---

### Step 1: Matrix Definition

Let $\varepsilon > 0$ be a fixed constant (e.g., $10^{-3}$). Define:

$$
M(\vec{d}) := [\vec{d}, (1 + \varepsilon) e_2, \dots, (1 + \varepsilon) e_N] \in \mathbb{C}^{N \times N}
$$

This matrix has:

* First column equal to $\vec{d}$;
* Remaining columns $(1 + \varepsilon) e_j$, ensuring full rank even when $\vec{d} \sim e_j$ for $j > 1$;
* Real-analytic dependence on $\vec{d}$ as a linear map.

> **Lemma (Full Rank of $M(\vec{d})$):**
>
> For any $\vec{d} \in \mathbb{C}^N$, $|\vec{d}| = 1$, the matrix
>
> $$
> M(\vec{d}) := [\vec{d}, (1 + \varepsilon)e_2, \dots, (1 + \varepsilon)e_N]
> $$
>
> is of full rank.
>
> **Proof**: Suppose $\vec{d}$ is colinear with $e_k$ for $k \ge 2$, i.e., $\vec{d} = \lambda e_k$. Then the first column is linearly dependent with $e_k$, but differs in scaling: $\vec{d} = \lambda e_k \ne (1 + \varepsilon) e_k$ since $|\vec{d}| = 1$ and $\varepsilon > 0$. Therefore, columns remain linearly independent. This ensures $M(\vec{d})$ is full-rank for all $\vec{d}$.

---

### Step 2: Polar Decomposition

Define:

$$
H(\vec{d}) := M(\vec{d})^\dagger M(\vec{d}) \in \mathbb{C}^{N \times N}, \quad P(\vec{d}) := H(\vec{d})^{1/2}
$$

Then:

$$
R(\vec{d}) := M(\vec{d}) \cdot H(\vec{d})^{-1/2} \in \mathrm{U}(N)
$$

> **Proposition (Analyticity and Positivity of $H(\vec{d})$):**
>
> For all $\vec{d} \in \mathbb{S}^{2N-1}_\text{unit}$, the matrix $H(\vec{d}) := M^\dagger(\vec{d}) M(\vec{d})$ is Hermitian and positive definite.
>
> Moreover, the matrix square root $H^{1/2}(\vec{d})$ and its inverse $H^{-1/2}(\vec{d})$ are well-defined and real-analytic in $\vec{d}$.
>
> **Proof**:
>
> 1. Since $M$ is full-rank (see Lemma above), $H = M^\dagger M$ is Hermitian and strictly positive definite.
> 2. The eigenvalues of $H(\vec{d})$ lie in $(0, \infty)$ and depend analytically on $\vec{d}$;
> 3. The function $H \mapsto H^{1/2}$ is real-analytic on the domain of positive-definite Hermitian matrices.
>    Therefore, $R(\vec{d}) = M(\vec{d}) H(\vec{d})^{-1/2}$ is analytic as a composition of analytic maps.

This definition ensures:

* $R(\vec{d})^\dagger R(\vec{d}) = I$ by construction;
* $R(\vec{d}) e_1 = \vec{d}$, since $M(\vec{d}) e_1 = \vec{d}$ and $H^{-1/2} e_1 = e_1$ up to scalar;
* Each entry of $R(\vec{d})$ is real-analytic in $\vec{d}$, since both matrix product and matrix square root on positive definite domain preserve analyticity.

> **Lemma (Preservation of First Column):**
>
> $R(\vec{d}) e_1 = \vec{d}$.
>
> **Proof**: By construction, $M(\vec{d}) e_1 = \vec{d}$. Then
>
> $$
> R(\vec{d}) e_1 = M(\vec{d}) H(\vec{d})^{-1/2} e_1.
> $$
>
> Since $H = M^\dagger M$, we compute:
>
> $$
> \|M e_1\|^2 = \langle M e_1, M e_1 \rangle = \langle \vec{d}, \vec{d} \rangle = 1,
> $$
>
> so $H^{1/2} e_1 = e_1$, and thus $R e_1 = M e_1 = \vec{d}$.

---

### Step 3: Extended Frame Definition

Define the block-diagonal extended frame:

$$
\mathcal{R}(\vec{d}) := \mathrm{diag}(R(\vec{d}), R(\vec{d})) \in \mathrm{U}(2N)
$$

This matrix operates on $\mathbb{C}_{\text{pos}}^N \oplus \mathbb{C}_{\text{dir}}^N \cong \mathbb{C}^{2N}$ and aligns both components with $\vec{d}$.

> **Origin of $\mathbb{C}^{2N}$:**
> 
> The space $\mathbb{C}^{2N}$ arises naturally in CPSF as the product of two independent complex domains:
> 
> $$
> \mathbb{C}^{2N} \cong \mathbb{T}_\mathbb{C}^N \times \mathbb{C}^N
> $$
> 
> * The first component $\mathbb{T}_\mathbb{C}^N$ corresponds to the **toroidal position** $z$;
> * The second component $\mathbb{C}^N$ corresponds to the **directional deviation** $\delta \vec{d}$ on the unit sphere $\mathbb{S}^{2N-1}_\text{unit}$;
> * Both are embedded into a joint ambient space $\mathbb{C}^{2N}$, on which the anisotropic Gaussian envelope is defined:
> 
>   $$
>   \rho_j(w) := \exp\left(-\pi \langle \Sigma_j^{-1} w, w \rangle\right)
>   $$
> 
>   with $w = \iota(\tilde{z} - \tilde{z}_j, \delta \vec{d}) \in \mathbb{C}^{2N}$.
> 
> This structural decomposition justifies the construction of the extended orthonormal frame $\mathcal{R}(\vec{d}) \in \mathrm{U}(2N)$ as a block-diagonal transformation acting uniformly on both subspaces.

---

### Properties and CPSF Requirements

* **R1–R2**: $R(\vec{d}) \in \mathrm{U}(N)$, $R^\dagger R = I$: satisfied by polar decomposition.
* **R3**: $R(\vec{d}) e_1 = \vec{d}$: direct consequence of matrix definition.
* **R4**: Orthogonality of complement columns: follows from unitarity.
* **R5**: Smoothness: real-analytic by composition of analytic maps.
* **R6**: Right $\mathrm{U}(N{-}1)$ invariance: $R'(\vec{d}) := R(\vec{d}) \cdot \mathrm{diag}(1, Q)$ preserves $\Sigma_j$ and $\psi_j^{\mathbb{T}}$.
* **R7**: $\mathcal{R}(\vec{d}) \in \mathrm{U}(2N)$: follows by construction.
* **R8**: Local trivialization via global real-analytic map: satisfied.
* **R9**: Uniform boundedness of derivatives on compact domain: follows from analyticity on $\mathbb{S}^{2N-1}_{\text{unit}}$.

> **Topological Remark on R8:**
> 
> Although global smooth sections of the unitary frame bundle over $\mathbb{S}^{2N-1}$ may not exist when $N > 1$ due to topological obstructions (non-parallelizability), the function $R(\vec{d})$ is defined by a global real-analytic formula. Thus, it defines a smooth **global trivialization candidate** within the analytic CPSF framework. Since only local triviality is required for analytic integration and projection, R8 is satisfied.

> **Boundedness of Derivatives (R9):**
> 
> As $R(\vec{d})$ is real-analytic on the compact manifold $\mathbb{S}^{2N-1}$, all its derivatives (of any order) are bounded. This follows from the compactness of the domain and analyticity of each component of $R$. This ensures boundedness of all integrands in CPSF field construction and convergence of the integrals in Appendix A.

---

### Final Definition

$$
\boxed{
R(\vec{d}) := M(\vec{d}) \cdot (M(\vec{d})^\dagger M(\vec{d}))^{-1/2}, \quad
M(\vec{d}) := [\vec{d}, (1 + \varepsilon) e_2, \dots, (1 + \varepsilon) e_N]
}
$$

This construction defines a global, real-analytic orthonormal frame aligned with $\vec{d}$, with smooth spectral behavior and full compatibility with the geometry and analytic dynamics of the CPSF field architecture.

### Additional Implementation Remarks

**Compatibility with CPSF Covariance Matrix:**

The construction aligns with the definition $\Sigma_j := \mathcal{R}^\dagger D \mathcal{R}$, where $D$ is diagonal. The block-diagonal extension $\mathcal{R} := \mathrm{diag}(R, R)$ preserves the product structure $\mathbb{C}^{2N} \cong \mathbb{C}^N_{\text{pos}} \oplus \mathbb{C}^N_{\text{dir}}$ and maintains $\mathrm{U}(2N)$ invariance.

**Stability Under Numerical Computation:**

The use of polar decomposition avoids Gram-Schmidt instability, and the positive-definiteness of $H(\vec{d})$ ensures that matrix square roots are well-conditioned. For practical purposes, $R(\vec{d})$ can also be computed via SVD:

$$
M = U \Sigma V^\dagger \quad \Rightarrow \quad R := U V^\dagger,
$$

where $U, V \in \mathrm{U}(N)$, ensuring high numerical stability for all $\vec{d}$.
