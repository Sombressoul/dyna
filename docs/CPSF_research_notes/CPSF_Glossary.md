## CPSF: Glossary of Core Terms

---

### Toroidal Spectral Coordinates

**$\mathbb{T}^N := (\mathbb{R} / \mathbb{Z})^N$** — *N-dimensional torus*, serving as the toroidal spatial domain for harmonic projection; a compact, orientable, boundaryless manifold.

All coordinates $x \in \mathbb{T}^N$ are real positions modulo 1 on the unit torus. Arithmetic is periodic over $\mathbb{R}^N$ modulo $\mathbb{Z}^N$, enabling toroidal continuity and Fourier projection.

**Spectral Basis** — Global toroidal harmonics

$$
\phi_k(x) := e^{2\pi i \langle k, x \rangle}, \quad k \in \mathbb{Z}^N
$$

form a complete orthonormal basis in $L^2(\mathbb{T}^N)$. They satisfy $\phi_k(x + a) = \phi_k(x) \cdot e^{2\pi i \langle k, a \rangle}$ and are covariant under toroidal shifts.

The phase factor $e^{2\pi i \langle k, a \rangle}$ encodes the shift symmetry of the basis under toroidal translations, ensuring covariance of the harmonic structure with respect to spatial shifts in $x$.

The index $k \in \mathbb{Z}^N$ is a discrete spectral vector defining the frequency mode of the harmonic. Each component $k_j$ determines the integer frequency along the $j$-th toroidal dimension.

---

### Projection Coordinates

**$\ell := (\vec{o}, \vec{d}) \in \mathbb{T}^N \times \mathbb{C}^N$** — *projection coordinate (ray)*.

Defines the geometric configuration of directional observation or interaction within CPSF.

* **Origin**: $\vec{o} \in \mathbb{T}^N$ — base point on the real torus;
* **Direction**: $\vec{d} \in \mathbb{C}^N, \|\vec{d}\| = 1$ — unit complex direction vector;
* The pair $(\vec{o}, \vec{d})$ defines a unique ray in the extended projection space.

---

### Attenuation Parameters

**$\sigma^{\parallel} \in \mathbb{R}_{>0}$** — *longitudinal attenuation scalar*.
A single positive real value controlling the projection envelope's scale **along the ray direction** $\vec{d}$; determines the effective extent of contribution along the ray.

**$\sigma^{\perp} \in \mathbb{R}_{>0}$** — *transverse attenuation scalar*.
A single positive real value controlling isotropic decay **in all directions orthogonal to** $\vec{d}$ within the ambient space $\mathbb{C}^N$.

---

### Orthonormal Frame

**$R(\vec{d}) \in \mathrm{U}(N)$** — *unitary rotation matrix* canonically associated with the direction vector $\vec{d} \in \mathbb{C}^N, \| \vec{d} \| = 1$.

It satisfies:

* $R e_1 = \vec{d}$ — the first column aligns with the projection direction (notation $R[:,1] := \vec{d}$ in Python-style indexing);
* $R^\dagger R = I_N$ — the matrix is unitary (orthonormal in complex space);
* The remaining $N - 1$ columns span the orthogonal complement of $\vec{d}$.

This matrix defines a canonical local frame in $\mathbb{C}^N$ and is used to construct anisotropic Gaussian envelopes aligned with $\vec{d}$.

---

### Extended Orthonormal Frame

**$\mathcal{R}(\vec{d}) \in \mathrm{U}(2N)$** — *block-diagonal unitary matrix* defined as:

$$
\mathcal{R}(\vec{d}) := \mathrm{diag}(R(\vec{d}), R(\vec{d}))
$$

It defines a canonical frame in $\mathbb{C}^{2N}$, aligned with the projection direction $\vec{d} \in \mathbb{C}^N$. Used in the construction of directionally aligned anisotropic structures, such as the geometric covariance matrix $\Sigma_j$.

---

### Geometric Covariance Matrix

**$\Sigma_j \in \mathbb{C}^{2N \times 2N}$** — *anisotropic localization matrix* associated with the projection coordinate $\ell_j = (\vec{o}_j, \vec{d}_j)$.

Constructed as:

$$
\Sigma_j := \mathcal{R}(\vec{d}_j)^\dagger \cdot
\mathrm{diag}\left( 
\mathrm{diag}(\sigma_j^{\parallel}, \sigma_j^{\perp} I_{N-1}),
\mathrm{diag}(\sigma_j^{\parallel}, \sigma_j^{\perp} I_{N-1})
\right) 
\cdot \mathcal{R}(\vec{d}_j)
$$

where:

* $\sigma_j^{\parallel}, \sigma_j^{\perp} \in \mathbb{R}_{>0}$ are longitudinal and transverse attenuation scalars,
* $\mathcal{R}(\vec{d}_j) \in \mathrm{U}(2N)$ is the extended orthonormal frame aligned with the direction $\vec{d}_j$.

This matrix defines the local anisotropic Gaussian envelope centered at $\ell_j$ and determines the spatial footprint in the toroidal projection coordinate space $\mathbb{T}^N \times \mathbb{C}^N$.

---

### Spectral Content Vector

**$\hat{T}_j \in \mathbb{C}^S$** — *semantic spectral vector* assigned to a contribution $C_j$.

The dimensionality $S$ is arbitrary and model-defined, representing the number of harmonics stored in the local semantic spectrum.

$\hat{T}_j$ encodes localized semantic content directly in the spectral domain. It is not computed from other values, but defined as an intrinsic, stored parameter of the contribution. While not fixed in the sense of being immutable, it is persistent and modifiable through field operations.

---

### Field Contribution

**$C_j := (\ell_j, \hat{T}_j, \sigma_j^{\parallel}, \sigma_j^{\perp}, \alpha_j)$** — *elementary localized excitation in the field*, specified by:

* projection coordinate $\ell_j = (\vec{o}_j, \vec{d}_j) \in \mathbb{T}^N \times \mathbb{C}^N$,
* spectral content vector $\hat{T}_j \in \mathbb{C}^S$,
* attenuation scalars $\sigma_j^{\parallel}, \sigma_j^{\perp} \in \mathbb{R}_{>0}$,
* scalar weight $\alpha_j \in \mathbb{R}_{\ge 0}$.

This tuple defines a directionally localized generator of field structure.
