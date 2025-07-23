## Functional Role of $\Sigma_j$

This section defines the functional and geometric role of the matrix $\Sigma_j \in \mathbb{C}^{2N \times 2N}$, which governs the anisotropic localization of field contributions in the Continuous Projective Semantic Fields (CPSF).

---

### 1. Geometric Context and Projection Coordinates

Let $\ell_j := (z_j, \vec{d}_j) \in \mathbb{T}_\mathbb{C}^N \times \mathbb{S}_\mathbb{C}^{2N-1}$ be a projection coordinate as defined in *"CPSF: Core Terms — Projection Coordinates"*.

* $z_j \in \mathbb{T}_\mathbb{C}^N$ is a base point on the complex torus;
* $\vec{d}_j \in \mathbb{S}_\mathbb{C}^{2N-1} \subset \mathbb{C}^N$ is a unit-norm complex direction vector (i.e., $\|\vec{d}_j\| = 1$).

Let $(z, \vec{d}) \in \mathbb{T}_\mathbb{C}^N \times \mathbb{S}_\mathbb{C}^{2N-1}$ be an arbitrary projection coordinate. We use lifted representatives $\tilde{z}, \tilde{z}_j \in \mathbb{C}^N$ such that $\tilde{z} \equiv z \mod \Lambda$, and define the relative offset:

$$
  w := \iota(\tilde{z} - \tilde{z}_j, \vec{d} - \vec{d}_j), \quad \iota(u, v) := \begin{bmatrix} u \\ v \end{bmatrix} \in \mathbb{C}^{2N}
$$

> **Geometric remark**: While both $\vec{d}$ and $\vec{d}_j$ lie on the unit sphere $\mathbb{S}_\mathbb{C}^{2N-1}$, the difference $\vec{d} - \vec{d}_j$ does not. It is interpreted in the ambient space $\mathbb{C}^N$ as an approximation to angular deviation within the tangent space at $\vec{d}_j$, and is valid only under the assumption that $\vec{d} \approx \vec{d}_j$.

---

### 2. Aligned Orthonormal Frame

As in *"CPSF: Core Terms — Orthonormal Frame"*, define the unitary matrix $R(\vec{d}_j) \in \mathrm{U}(N)$ satisfying:

* $R(\vec{d}_j) e_1 = \vec{d}_j$
* The remaining columns span $\vec{d}_j^\perp$ and are orthonormal with respect to the Hermitian inner product:

$$
  \langle u, v \rangle := \sum_{k=1}^N \overline{u_k} v_k
$$

Define the extended block-diagonal frame:

$$
  \mathcal{R}(\vec{d}_j) := \mathrm{diag}(R(\vec{d}_j), R(\vec{d}_j)) \in \mathrm{U}(2N)
$$

This acts on $\mathbb{C}_{\text{pos}}^N \oplus \mathbb{C}_{\text{dir}}^N \cong \mathbb{C}^{2N}$, aligning both spatial and directional components with $\vec{d}_j$.

---

### 3. Attenuation and Covariance Matrix

Let $\sigma_j^{\parallel}, \sigma_j^{\perp} \in \mathbb{R}_{>0}$ be the longitudinal and transverse attenuation parameters respectively (see *"CPSF: Core Terms — Attenuation Parameters"*).

Define the diagonal attenuation matrix:

$$
  D_j := \mathrm{diag}(\sigma_j^{\parallel}, \underbrace{\sigma_j^{\perp}, \dotsc, \sigma_j^{\perp}}_{N-1}, \sigma_j^{\parallel}, \underbrace{\sigma_j^{\perp}, \dotsc, \sigma_j^{\perp}}_{N-1}) \in \mathbb{R}^{2N \times 2N}
$$

Then the geometric covariance matrix is:

$$
  \Sigma_j := \mathcal{R}(\vec{d}_j)^{\dagger} \cdot D_j \cdot \mathcal{R}(\vec{d}_j) \in \mathbb{C}^{2N \times 2N}
$$

By construction, $\Sigma_j$ is Hermitian and strictly positive definite. It defines an anisotropic Gaussian metric whose principal axes are aligned with the projection direction $\vec{d}_j$, and whose longitudinal and transverse variances are given by $\sigma_j^{\parallel}$ and $\sigma_j^{\perp}$, respectively (see *"CPSF: Core Terms — Geometric Covariance Matrix"*).

---

### 4. Gaussian Envelope and Periodization

Define the unnormalized anisotropic Gaussian envelope (see *"CPSF: Core Terms — Unnormalized Gaussian Envelope"*):

$$
  \rho_j(w) := \exp\left( -\pi \langle \Sigma_j^{-1} w, w \rangle \right)
$$

where $w \in \mathbb{C}^{2N}$ is the relative offset defined above.

To restore toroidal periodicity in $z \in \mathbb{T}_\mathbb{C}^N$, define the periodized envelope (see *"CPSF: Core Terms — Periodized Envelope"*):

$$
  \psi_j^{\mathbb{T}}(z, \vec{d}) := \sum_{n \in \Lambda} \rho_j\left( \iota(\tilde{z} - \tilde{z}_j + n, \vec{d} - \vec{d}_j) \right)
$$

This function is smooth, $\Lambda$-periodic in $z$, rapidly decaying in $\vec{d}$, and invariant under the choice of lifted representatives $\tilde{z}, \tilde{z}_j \in \mathbb{C}^N$. This invariance follows from the lattice summation over all $n \in \Lambda$, which eliminates dependence on the specific choice of representatives (see *"CPSF: Core Terms — Periodized Envelope"*).

The envelope $\psi_j^{\mathbb{T}}$ thus localizes the influence of each field contribution $C_j$ to a region in projection space whose shape is geometrically induced by $\Sigma_j$.

---

### 5. Field Construction and Semantic Projection

Let $\mathcal{J} \subset \mathbb{N}$ index the finite collection of field contributions $C_j := (\ell_j, \hat{T}_j, \sigma_j^{\parallel}, \sigma_j^{\perp}, \alpha_j)$ as defined in *"CPSF: Core Terms — Field Contribution"*.

Then the global semantic field (see *"CPSF: Core Terms — Global Field Response") is:

$$
  T(z, \vec{d}) := \sum_{j \in \mathcal{J}} \alpha_j \cdot \psi_j^{\mathbb{T}}(z, \vec{d}) \cdot \hat{T}_j \in \mathbb{C}^S
$$

Let $T^{\text{ref}}(z, \vec{d}) \in \mathbb{C}^S$ denote a reference field. Define the semantic error field:

$$
  \Delta T(z, \vec{d}) := T^{\text{ref}}(z, \vec{d}) - T(z, \vec{d})
$$

Projecting this error onto the envelope $\psi_j^{\mathbb{T}}$ with respect to the canonical Hilbert space (see *"CPSF: Core Terms — Projection Space Measure"*):

$$
  L^2(\mathbb{T}_\mathbb{C}^N \times \mathbb{S}_\mathbb{C}^{2N-1}; \mathbb{C}^S)
$$

yields the orthogonal projection (see *"CPSF: Core Terms — Semantic Error Projection"*):

$$
  \Delta \hat{T}_j := \frac{1}{\alpha_j} \cdot \frac{ \int_{\mathbb{T}_\mathbb{C}^N} \int_{\mathbb{S}_\mathbb{C}^{2N-1}} \overline{\psi_j^{\mathbb{T}}(z, \vec{d})} \cdot \Delta T(z, \vec{d}) \, d\sigma(\vec{d}) \, d\mu(z) }{ \int_{\mathbb{T}_\mathbb{C}^N} \int_{\mathbb{S}_\mathbb{C}^{2N-1}} |\psi_j^{\mathbb{T}}(z, \vec{d})|^2 \, d\sigma(\vec{d}) \, d\mu(z) }
$$

This projection yields the optimal semantic update $\Delta \hat{T}_j \in \mathbb{C}^S$ that minimizes the squared error weighted by the localization profile $\psi_j^{\mathbb{T}}$, which in turn is induced by $\Sigma_j$.

---

### 6. Summary

* $\Sigma_j$ defines a directionally aligned, anisotropic envelope centered at $\ell_j$;
* Its construction follows from a block-diagonal frame $\mathcal{R}(\vec{d}_j)$ aligned with $\vec{d}_j$, and a diagonal attenuation matrix $D_j$;
* The Gaussian envelope $\psi_j^{\mathbb{T}}$ provides localized spatial and directional weighting for both field synthesis and semantic projection;
* The projection $\Delta \hat{T}_j$ minimizes the localized semantic error with respect to $\psi_j^{\mathbb{T}}$, making $\Sigma_j$ central to the semantic learning process;
* All constructions are fully and rigorously consistent with the canonical structure of CPSF.
