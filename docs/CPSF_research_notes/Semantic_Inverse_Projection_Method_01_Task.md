### Directional Boundary Projection

Let $T^* \in \mathbb{C}^S \setminus {0}$ be the fixed semantic target, and let $\ell^*(T^*) = (z^*, \vec{d}^*)$ be the coordinate pair obtained via *Semantic Inverse Projection* (Step 1–2).

Let $\Psi(z)$ be the accumulated semantic influence envelope defined by:

$$
\Psi(z) := \sum_{j \in \mathcal{J}} \|\gamma_j\|^2 \cdot \psi_j^{\mathbb{T}}(z, \vec{d}^*), \quad z \in \mathbb{T}_\mathbb{C}^N
$$

Let the field response at $(z, \vec{d}^*)$ be:

$$
T(z, \vec{d}^*) := \sum_{j \in \mathcal{J}} \alpha_j \cdot \psi_j^{\mathbb{T}}(z, \vec{d}^*) \cdot \hat{T}_j \in \mathbb{C}^S
$$

---

### Definition: $z^{**}$ (Directional boundary projection)

Define $z^{**} \in \mathbb{T}_\mathbb{C}^N$ as the point satisfying the following:

$$
z^{**} := z^* - \lambda^* \cdot \vec{d}^*, \quad \lambda^* > 0
$$

such that:

$$
\|T(z^{**}, \vec{d}^*) - T^*\|^2 \leq \tau^2 \cdot \|T(z^*, \vec{d}^*) - T^*\|^2
$$

for a given tolerance level $\tau \in (0, 1)$.
In other words, $z^{**}$ lies on the boundary of the region where the local field response approximates $T^*$ within relative deviation $\tau$, and is reached from $z^*$ along the direction $-\vec{d}^*$.

---

### Rationale

* $\Psi(z)$ defines a smooth, locally convex accumulation of directed Gaussian envelopes $\psi_j^{\mathbb{T}}(z, \vec{d}^*)$, each aligned approximately along $\vec{d}_j \approx \vec{d}^*$.
* $z^*$ is the barycenter of $\Psi(z)$ and lies within the high-response region.
* Due to envelope elongation along $\vec{d}^*$, the maximal constructive interference is typically displaced along $-\vec{d}^*$ from $z^*$.
* The directional displacement from $z^*$ to $z^{**}$ thus projects $z^*$ onto the outer response boundary along the axis of semantic anisotropy.

---

### Justification: Field approximation error

The deviation

$$
\Delta T(z) := T^* - T(z, \vec{d}^*)
$$

is minimized more strongly at $z^{**}$ than at $z^*$ under the structural condition that:

$$
\frac{\partial \Psi(z)}{\partial z} \cdot \vec{d}^* < 0 \quad \text{at } z = z^*
$$

Thus:

$$
\|\Delta T(z^{**})\| < \|\Delta T(z^*)\|
$$

follows directly from the monotonic decay of $\Psi(z)$ along $\vec{d}^*$ and the linearity of $T(z, \vec{d}^*)$ in $\psi_j^{\mathbb{T}}$.

---

### Summary

The computation of $z^{**}$ reduces to a **geometric projection** of $z^*$ onto the boundary surface of the envelope $\Psi(z)$ along direction $-\vec{d}^*$, such that the projected point satisfies a field approximation constraint relative to $T^*$.
This operation is analytic, directionally constrained, and consistent with the differential geometry and signal composition model of CPSF.
