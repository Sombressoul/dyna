# Appendix A — Spectral Extension of Holographic Projection Memory

> *A structured formulation of memory fields with spectral embeddings as an optional extension of the core HPM architecture.*

---

## A.1 Motivation

The HPM framework defines a differentiable memory field $W(x)$ accessed via directional projections. While the mathematical core assumes no particular structure of $W(x)$ beyond local integrability and differentiability, practical implementations benefit from structured internal representations.

This appendix introduces one such representation: **spectral memory**, where each location $x$ stores a local frequency decomposition rather than a single scalar or feature vector. This enables projection to return structured signals (spectra) instead of raw activations, allowing semantic interpretation via interference, phase patterns, or frequency-based attention.

---

## A.2 Spectral Memory Field Representation

Let the memory field at each spatial location $x$ be represented as:

$$
W(x) = s_x = [\hat{w}_0(x), \hat{w}_1(x), \dots, \hat{w}_{K-1}(x)] \in \mathbb{C}^K,
$$

where each $\hat{w}_k(x)$ is a complex coefficient corresponding to a basis function $\phi_k(x)$. For example, in the Fourier basis:

$$
\phi_k(x) = e^{i 2 \pi f_k x},
$$

with $f_k$ being the $k$-th spatial frequency (generalized to multiple dimensions as needed).

This means that each memory voxel stores a small local spectrum rather than a scalar or latent embedding.

---

## A.3 Spectral Projection Operator

Given a projection ray $\ell_u$ and a spatial kernel $K(x, \ell_u)$, the projection response is computed for **each frequency component independently**:

$$
T_k(u) = \int \hat{w}_k(x) \cdot \phi_k(x) \cdot K(x, \ell_u) \, dx, \quad \text{for } k = 0 \dots K-1.
$$

The full projection $T(u)$ becomes a frequency-resolved signal:

$$
T(u) = [T_0(u), T_1(u), \dots, T_{K-1}(u)], \quad T_k(u) \in \mathbb{C}.
$$

This can be interpreted as the **spectral profile along ray $u$**, capturing how different frequencies respond to content in memory along that direction.

In practice, this yields a projection tensor with shape:

$$
T \in \mathbb{R}^{R_x \times R_y \times K \times 2},
$$

where $R_x$, $R_y$ are spatial resolution along the projection surface, $K$ is the number of frequencies, and the final dimension encodes $(\operatorname{Re}, \operatorname{Im})$ parts.

---

## A.4 Delta-Learning in Spectral Space

If a desired projection $T^*(u)$ is known (e.g., as a spectral target or latent embedding), memory can be updated per frequency via projection error:

$$
\delta_k(u) = T_k^*(u) - T_k(u)
$$

Update rule for each component:

$$
\Delta \hat{w}_k(x) = \alpha \cdot \delta_k(u) \cdot \phi_k^*(x) \cdot K(x, \ell_u),
$$

where $\phi_k^*(x)$ is the complex conjugate of the basis function.

Total memory update at point $x$ becomes:

$$
\Delta W(x) = \sum_k \Delta \hat{w}_k(x) \cdot \phi_k(x)
$$

This rule preserves spatial locality, differentiability, and alignment with the geometric projection structure of HPM.

---

## A.5 Interpretive Implications

Spectral HPM enables a number of additional behaviors:

* **Phase-driven semantics**: coherence across projections can induce constructive or destructive interference, enabling selective recall.
* **Directional tuning**: spatial frequencies implicitly capture orientation and repetition.
* **Compression and sparsity**: low-frequency dominance can be exploited to reduce memory size.
* **Semantic resonances**: meaningful structures may emerge as frequency-domain attractors.

The spectral projection $T(u)$ can also be interpreted downstream by a module such as **TensorComposerMobius** (a part of DyNA framework), which decomposes or composes structured spectral signals into abstract representations.

---

## A.6 Compatibility and Generalization

* This formulation is valid for any complete or overcomplete basis ${\phi_k(x)}$ (e.g., wavelets, learned filters).
* Memory and projection remain linear and differentiable.
* Standard HPM kernels $K(x, \ell_u)$ can be reused without change.

Importantly, the spectral extension does **not require any modification** of the HPM framework itself — it only changes the type of content stored at each point $x$.

---

## A.7 Summary

The spectral extension of HPM treats each ray projection as a **localized frequency scan** through the semantic field. This enables richer and more compact representations, allows natural modulation and interference, and opens new directions for symbolic interpretation, dynamic adaptation, and high-level semantic composition.

> *Projection becomes not only an act of perception, but an act of resonance.*
