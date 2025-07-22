## CPSF: Glossary of Core Terms

---

### Complex Torus

**$\mathbb{T}_\mathbb{C}^N := \mathbb{C}^N / \Lambda$** — *compact complex torus*, formed by quotienting complex space $\mathbb{C}^N$ by a full-rank lattice $\Lambda \subset \mathbb{C}^N$, typically $\Lambda = \mathbb{Z}^N + i \mathbb{Z}^N$. This generalizes the real torus and supports a global Fourier basis indexed by the dual lattice $\Lambda^* \subset \mathbb{C}^N$.

---

### Toroidal Spectral Coordinates

**$\mathbb{T}_\mathbb{C}^N := \mathbb{C}^N / \Lambda$** — *N-dimensional complex torus*, serving as the toroidal spatial domain for harmonic projection; a compact, orientable, boundaryless complex manifold.

All coordinates $z \in \mathbb{T}_\mathbb{C}^N$ are complex positions modulo the lattice $\Lambda \subset \mathbb{C}^N$. Arithmetic is periodic over $\mathbb{C}^N$ modulo $\Lambda$, enabling toroidal continuity and Fourier projection.

**Spectral Basis** — Global toroidal harmonics

$$
\phi_m(z) := e^{2\pi i \langle m, z \rangle}, \quad m \in \Lambda^*
$$

form a complete orthonormal basis in $L^2(\mathbb{T}_\mathbb{C}^N)$. They satisfy $\phi_m(z + \lambda) = \phi_m(z) \cdot e^{2\pi i \langle m, \lambda \rangle}$ and are covariant under toroidal shifts.

The phase factor $e^{2\pi i \langle m, \lambda \rangle}$ encodes the shift symmetry of the basis under toroidal translations, ensuring covariance of the harmonic structure with respect to spatial shifts in $z$.

The index $m \in \Lambda^*$ is a discrete spectral vector defining the frequency mode of the harmonic. Each component determines the dual frequency along a corresponding complex lattice direction.

---

### Projection Coordinates

**$\ell := (z, \vec{d}) \in \mathbb{T}_\mathbb{C}^N \times \mathbb{C}^N$** — *projection coordinate (ray)*.

Defines the geometric configuration of directional observation or interaction within CPSF.

* **Origin**: $z \in \mathbb{T}_\mathbb{C}^N$ — base point on the complex torus;
* **Direction**: $\vec{d} \in \mathbb{C}^N, \|\vec{d}\| = 1$ — unit complex direction vector;
* The pair $(z, \vec{d})$ defines a unique complex ray in the extended projection space:

$$
z(t) = z + t \cdot \vec{d} \mod \Lambda
$$

---

### Attenuation Parameters

**$\sigma^{\parallel} \in \mathbb{R}_{>0}$** — *longitudinal attenuation scalar*.

A single positive real value controlling the projection envelope's scale **along the ray direction** $\vec{d}$; determines the effective extent of contribution along the ray.

**$\sigma^{\perp} \in \mathbb{R}_{>0}$** — *transverse attenuation scalar*.

A single positive real value controlling isotropic decay **in all directions orthogonal to** $\vec{d}$ within the ambient space $\mathbb{C}^N$.

---

### Orthonormal Frame

**$R(\vec{d}) \in \mathrm{U}(N)$** — *unitary rotation matrix* associated with the direction vector $\vec{d} \in \mathbb{C}^N, \| \vec{d} \| = 1$.

When it exists, it satisfies:

* $R e_1 = \vec{d}$ — the first column aligns with the projection direction;
* $R^\dagger R = I_N$ — the matrix is unitary;
* The remaining $N - 1$ columns span the orthogonal complement of $\vec{d}$.

The construction of $R(\vec{d})$ is defined separately (see **TODO**).

---

### Extended Orthonormal Frame

**$\mathcal{R}(\vec{d}) \in \mathrm{U}(2N)$** — *block-diagonal unitary matrix* defined as:

$$
\mathcal{R}(\vec{d}) := \mathrm{diag}(R(\vec{d}), R(\vec{d}))
$$

It defines a unitary frame in $\mathbb{C}^{2N}$, aligned with the projection direction $\vec{d} \in \mathbb{C}^N$. Used in the construction of directionally aligned anisotropic structures, such as the geometric covariance matrix $\Sigma_j$.

---

### Geometric Covariance Matrix

**$\Sigma_j \in \mathbb{C}^{2N \times 2N}$** — *anisotropic localization matrix* associated with the projection coordinate $\ell_j = (z_j, \vec{d}_j)$ and attenuation parameters $\sigma_j^{\parallel}, \sigma_j^{\perp} \in \mathbb{R}_{>0}$.

Defined as:

$$
\Sigma_j := \mathcal{R}(\vec{d}_j)^\dagger \cdot \mathrm{diag}(\sigma_j^{\parallel}, \underbrace{\sigma_j^{\perp}, \dotsc, \sigma_j^{\perp}}_{N-1}, \sigma_j^{\parallel}, \underbrace{\sigma_j^{\perp}, \dotsc, \sigma_j^{\perp}}_{N-1}) \cdot \mathcal{R}(\vec{d}_j)
$$

where:

* $\mathcal{R}(\vec{d}_j) \in \mathrm{U}(2N)$ is the extended orthonormal frame aligned with direction $\vec{d}_j$,
* the diagonal encodes longitudinal scaling along $\vec{d}_j$ and isotropic transverse scaling in the orthogonal complement.

This matrix defines a local Gaussian envelope aligned with the projection ray, and determines the spatial footprint of the contribution in the complex toroidal domain $\mathbb{T}_\mathbb{C}^N \times \mathbb{C}^N$.

---

### Spectral Content Vector

**$\hat{T}_j \in \mathbb{C}^S$** — *semantic spectral vector* assigned to a contribution $C_j$.

The dimensionality $S$ is arbitrary and model-defined, representing the number of harmonics stored in the local semantic spectrum.

$\hat{T}_j$ encodes localized semantic content directly in the spectral domain. It is not computed from other values, but defined as an intrinsic, stored parameter of the contribution. While not fixed in the sense of being immutable, it is persistent and modifiable through field operations.

> The vector $\hat{T}_j \in \mathbb{C}^S$ resides outside the toroidal spectral domain and belongs to a higher semantic layer. It does not depend on the toroidal coordinates $z \in \mathbb{T}_\mathbb{C}^N$, but is coupled to it through localized geometric weighting.

---

### Field Contribution

**$C_j := (\ell_j, \hat{T}_j, \sigma_j^{\parallel}, \sigma_j^{\perp}, \alpha_j)$** — *elementary localized excitation in the field*, specified by:

* projection coordinate $\ell_j = (z_j, \vec{d}_j) \in \mathbb{T}_\mathbb{C}^N \times \mathbb{C}^N$,
* spectral content vector $\hat{T}_j \in \mathbb{C}^S$,
* attenuation scalars $\sigma_j^{\parallel}, \sigma_j^{\perp} \in \mathbb{R}_{>0}$,
* scalar weight $\alpha_j \in \mathbb{R}_{\ge 0}$.

This tuple defines a directionally localized generator of field structure.
