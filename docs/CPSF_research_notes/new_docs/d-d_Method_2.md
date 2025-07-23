## Directional Offset via Logarithmic Map on the Unit Sphere

This section introduces a strictly geometric and analytically minimal modification to the directional offset construction in CPSF. It replaces the linear surrogate $\vec{d} - \vec{d}_j$ with a geodesically aligned offset in the tangent space to the unit sphere $\mathbb{S}^{2N-1}_\text{unit} \subset \mathbb{C}^N$.

---

### Motivation and Context

In the canonical CPSF construction, the relative offset vector used in the Gaussian envelope is:

$$
w := \iota(\tilde{z} - \tilde{z}_j,\ \vec{d} - \vec{d}_j) \in \mathbb{C}^{2N},
$$

with $\iota(u, v) := \begin{bmatrix} u \\ v \end{bmatrix}$. While this surrogate is valid locally, it becomes geometrically inaccurate for moderate or large deviations on the sphere $\mathbb{S}^{2N-1}_\text{unit}$, misrepresenting angular proximity and interfering with unitarily invariant decay.

---

### Logarithmic Offset in Tangent Space (Regularized)

Let:

* $\vec{d}, \vec{d}_j \in \mathbb{S}^{2N-1}_\text{unit} \subset \mathbb{C}^N$: projection directions (unit-norm);
* $\tilde{z}, \tilde{z}_j \in \mathbb{C}^N$: lifted toroidal positions (with $\tilde{z} \equiv z \mod \Lambda$);
* $\iota : \mathbb{C}^N \times \mathbb{C}^N \to \mathbb{C}^{2N}$: canonical embedding $\iota(u, v) := \begin{bmatrix} u \\ v \end{bmatrix}$;
* $\varepsilon > 0$: regularization constant (e.g., $10^{-6}$).

Define the angular distance:

$$
\theta := \arccos \Re\langle \vec{d}, \vec{d}_j \rangle \in [0, \pi].
$$

Define the Hermitian-orthogonal projection:

$$
P^{\perp}_{\vec{d}_j}(\vec{d}) := \vec{d} - \langle \vec{d}, \vec{d}_j \rangle \cdot \vec{d}_j.
$$

Then the regularized directional offset is:

$$
\delta \vec{d} := \theta \cdot \frac{P^{\perp}_{\vec{d}_j}(\vec{d})}{\sqrt{\max\left(1 - |\langle \vec{d}, \vec{d}_j \rangle|^2, \varepsilon\right)}} \in \mathbb{C}^N.
$$

This expression belongs to the tangent space $T_{\vec{d}_j} \mathbb{S}^{2N-1}_\text{unit}$ and its norm smoothly approximates the geodesic angle $\theta$.

---

### Reformulated Unnormalized Gaussian Envelope

The full argument vector becomes:

$$
w := \iota(\tilde{z} - \tilde{z}_j,\ \delta \vec{d}) \in \mathbb{C}^{2N}.
$$

The envelope $\rho_j(w) \in \mathbb{R}_{>0}$ is:

$$
\rho_j(w) := \exp(-\pi \langle \Sigma_j^{-1} w, w \rangle),
$$

where:

* $\Sigma_j \in \mathbb{C}^{2N \times 2N}$: geometric covariance matrix,
* $\Sigma_j := \mathcal{R}(\vec{d}_j)^\dagger D_j \mathcal{R}(\vec{d}_j)$,
* $\mathcal{R}(\vec{d}_j) \in \mathrm{U}(2N)$: extended orthonormal frame,
* $D_j \in \mathbb{R}^{2N \times 2N}$: attenuation matrix.

---

### Periodized Envelope and Semantic Field

Toroidally periodized envelope:

$$
\psi_j^{\mathbb{T}}(z, \vec{d}) := \sum_{n \in \Lambda} \rho_j\left( \iota(\tilde{z} - \tilde{z}_j + n,\ \delta \vec{d}) \right).
$$

Semantic field response:

$$
T(z, \vec{d}) := \sum_{j \in \mathcal{J}} \alpha_j \cdot \psi_j^{\mathbb{T}}(z, \vec{d}) \cdot \hat{T}_j \in \mathbb{C}^S.
$$

All semantic update, projection, and aggregation structures remain unchanged and are valid under this definition.

---

### Consistency and Functional Properties

This construction satisfies all structural constraints of CPSF:

* $\delta \vec{d} \in T_{\vec{d}_j} \mathbb{S}^{2N-1}_\text{unit}$;
* $\|\delta \vec{d}\| \approx \angle(\vec{d}, \vec{d}_j)$;
* Embedding $w \in \mathbb{C}^{2N}$ remains compatible with $\Sigma_j$;
* $\psi_j^{\mathbb{T}} \in L^2$ and all integrals over the projection domain remain well-defined;
* Construction is smooth, rotation-invariant, and non-degenerate under the standard Hermitian structure.

---

### Implementation Notes

* As a consequence, contributions $C_j$ with orthogonal directions $\vec{d}_j \perp \vec{d}$ do not vanish strictly but retain an exponentially suppressed influence, governed by the decay:

  $$
  \rho_j(w) \sim \exp\left(-\pi \|\delta \vec{d}\|^2\right), \quad \text{with } \|\delta \vec{d}\| \sim \frac{\theta}{\sqrt{\varepsilon}}.
  $$

* This ensures smooth continuity in angular space while preserving effective localization.

* This form requires no modification to $\Sigma_j$, $\mathcal{R}(\vec{d}_j)$, or integration domains.

* Drop-in replacement for all usages of $\vec{d} - \vec{d}_j$ in the existing field architecture.
