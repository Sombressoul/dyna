## CPSF: Glossary of Core Terms

---

### Toroidal Spectral Coordinates

**$\mathbb{T}^N := (\mathbb{C} / \mathbb{Z})^N$** — *N-dimensional complex torus*, defining the spectral coordinate domain as a compact, orientable, boundaryless manifold.
All coordinates $x \in \mathbb{T}^N$ are phase-aligned complex positions modulo 1. Arithmetic is spectral over $\mathbb{C}^N$, enabling toroidal continuity and Fourier projection.

**Spectral Basis** — Global toroidal harmonics

$$
\phi_k(x) := e^{2\pi i \langle k, x \rangle}, \quad k \in \mathbb{Z}^N
$$

form a complete orthonormal basis in $L^2(\mathbb{T}^N)$. They satisfy $\phi_k(x + a) = \phi_k(x) \cdot \phi_k(a)$ and are equivariant under toroidal shifts.
The index $k \in \mathbb{Z}^N$ is a discrete spectral vector defining the frequency mode of the harmonic. Each component $k_j$ determines the integer frequency along the $j$-th toroidal dimension.

---

### Projection Coordinates

**$\ell := (\vec{o}, \vec{d}) \in \mathbb{T}^{2N} \subset \mathbb{C}^{2N}$** — *projection coordinate (ray)*.
Defines the geometric configuration of directional observation or memory access within CPSF.

* **Origin**: $\vec{o} \in \mathbb{T}^N$ — complex-valued base point on the torus;
* **Direction**: $\vec{d} \in \mathbb{C}^N$, $\|\vec{d}\| = 1$ — unit complex direction vector;
* The pair $(\vec{o}, \vec{d})$ defines a unique ray on the toroidal manifold, modulo 1 componentwise.

---

### Attenuation Parameters

**$\sigma^{\parallel} \in \mathbb{R}_{>0}$** — *longitudinal attenuation scalar*.
A single positive real value controlling the projection envelope's scale **along the ray direction** $\vec{d}$; determines the effective extent of contribution along the ray.

**$\sigma^{\perp} \in \mathbb{R}_{>0}$** — *transverse attenuation scalar*.
A single positive real value controlling isotropic decay **in all directions orthogonal to** $\vec{d}$ within the ambient space $\mathbb{C}^N$.

---

### Orthonormal Frame

**$R(\vec{d}) \in \mathrm{U}(N)$** — *unitary rotation matrix* canonically associated with the direction vector $\vec{d} \in \mathbb{C}^N$, $\| \vec{d} \| = 1$.

It satisfies:

* $R[:,1] := \vec{d}$ — the first column aligns with the projection direction;
* $R^\dagger R = I_N$ — the matrix is unitary (orthonormal in complex space);
* The remaining $N - 1$ columns span the orthogonal complement of $\vec{d}$.

This matrix defines a canonical local frame in $\mathbb{C}^N$ and is used to construct anisotropic Gaussian envelopes aligned with $\vec{d}$.

---

### Spectral Content Vector

**$\hat{T}_j \in \mathbb{C}^S$** — *semantic spectral vector* assigned to a memory contribution $C_j$.
The dimensionality $S$ is arbitrary and model-defined, representing the number of harmonics stored in the local semantic spectrum.

$\hat{T}_j$ encodes localized semantic content directly in the spectral domain. It is not computed from other values, but defined as an intrinsic, stored parameter of the contribution.
While not fixed in the sense of being immutable, it is persistent and modifiable through field operations.
