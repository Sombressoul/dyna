## Chapter F - Practical Engineering Optimizations

**This chapter translates the theoretical geometry of Holographic Projection Memory (HPM) into high-performance, gradient-compatible engineering.**  

It outlines how core operations - projection, ray traversal, and memory updates - can be implemented efficiently without sacrificing differentiability, semantic alignment, or structural clarity.  

The goal is not to approximate HPM heuristically, but to execute it *faithfully* and *at scale*.  

---

## F.1 Overview

The Holographic Projection Memory (HPM) framework is mathematically elegant and fully differentiable, but its naive implementation is computationally intensive and ill-suited for real-time deployment. To bridge the gap between theory and practice, this chapter presents a collection of **engineering optimizations** that preserve the full expressive power and differentiability of HPM while making it tractable on modern hardware.

These optimizations are structured along four functional axes:

1. **Geometric Simplification** - Analytical inference of projection geometry to eliminate redundant storage and reduce computational overhead.
2. **Gradient-Compatible Rasterization** - Enabling gradient flow through discretized ray tracing by reconstructing differentiable surrogates for projection parameters.
3. **Runtime Efficiency Enhancements** - Performance-aware techniques for memory access, batching, and kernel evaluation to enable high-throughput projection in large memory fields.
4. **Flexible Architectural Configuration** - Support for modular projection interfaces, hybrid execution regimes, and tunable trade-offs between expressiveness and speed.

All methods are designed to maintain strict alignment with the geometric foundations of HPM, as formulated in Chapters D and E. The memory field $W(x)$ remains a continuous, differentiable function discretized over a regular lattice, and projection rays $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$ remain the core access mechanism.

### Design Philosophy

These optimizations follow three core principles:

* **Preserve Differentiability:** Every transformation or approximation must admit exact or surrogate gradient flow. Where native gradients are lost (e.g., in rasterized traversal), alternative pathways (e.g., via entry/exit point reconstruction) must be provided.

* **Exploit Locality:** The projection kernel $K(x, \ell_u)$ decays exponentially with spatial and angular distance. Efficient approximation is possible by restricting computation to a bounded support region around each ray.

* **Decouple Geometry from Implementation:** HPM geometry is continuous and analytical. Discrete realizations (e.g., voxel stepping) are viewed as numerical approximations, not fundamental alterations of the model. The geometry should dictate the implementation - not the reverse.

### Scope and Relationship to Theory

This appendix does not alter any theoretical definition of the HPM model. Instead, it specifies how to implement the projection operator

$$
T(u) = \int W(x) \cdot K(x, \ell_u) \, dx
$$

and its corresponding update rule

$$
W(x) \leftarrow W(x) + \alpha \cdot \delta(u) \cdot K(x, \ell_u)
$$

in a way that is consistent with the analytical structure but computationally feasible.

The strategies presented herein are fully compatible with the adaptive memory dynamics (Chapters A–C), directional protocol (Chapter D), field discretization and traversal (Chapter E), hierarchical scanning (Chapter G), and delta-based learning (Chapter H). They also reflect and incorporate clarifications from Chapter Q regarding gradient flow, kernel behavior, and projection semantics.

> *Engineering should not overwrite geometry. It should reveal it, at speed.*

---

## F.2 Geometric Inference Instead of Storage

In the core formulation of Holographic Projection Memory (HPM), each projection ray $\ell_u(t)$ is defined by a viewpoint surface $\Phi(u)$ and a direction field $\mathbf{v}_u$. For practical implementation, the ray is discretized into a finite sequence of voxel centers $x_i$ along the ray path. However, naively storing auxiliary variables such as step index $i$, time-step $t_i$, or directional gradients for each projection incurs substantial overhead.

This section describes how to reconstruct all such quantities **analytically and differentiably** from geometric primitives, eliminating the need to cache or backpropagate through ray-tracing state. The result is a stateless, gradient-compatible implementation of HPM projection.

---

### F.2.1 Scalar Projection Time from Geometry

For any voxel center $x_i$ on the ray $\ell_u$, its longitudinal coordinate (relative to the surface $\Phi(u)$) can be computed via scalar projection:

$$
t_i = (x_i - \Phi(u)) \cdot \mathbf{v}_u
$$

This gives the relative distance along the ray direction, assuming $|\mathbf{v}_u| = 1$. The quantity $t_i$ is required for longitudinal decay in projection kernels:

$$
K(x_i, \ell_u) = \exp\left( -\frac{t_i}{\tau} \right)
$$

Importantly, this formula is fully differentiable with respect to both $\Phi(u)$ and $\mathbf{v}_u$.

---

### F.2.2 Inferring Ray Direction from Entry and Exit Points

In rasterized ray traversal (e.g., Bresenham-style), the input may not provide a differentiable $\mathbf{v}_u$, especially when directions are selected from a discrete codebook.

To recover a smooth surrogate direction, use the **entry and exit points** of the ray through the memory volume:

$$
A = \text{entry point}, \quad B = \text{exit point}
$$

Then define the effective direction as:

$$
\mathbf{v}_{\text{eff}} = \frac{B - A}{\|B - A\|}
$$

This approximation enables gradient backpropagation through ray alignment, even when the direction was originally non-differentiable.

---

### F.2.3 Eliminating Step Index Dependence

In standard ray marching, a voxel $x_i$ is indexed by an integer $i$ indicating its distance from the ray origin. However, $i$ is an implicit counter - not a continuous or differentiable variable.

Instead, all required quantities can be recovered from geometry:

* $t_i$ as in Section F.2.1
* $x_i = \Phi(u) + t_i \cdot \mathbf{v}_u$ (optional forward generation)

Thus, the step index $i$ **never needs to be stored or exposed**. The system remains stateless and differentiable.

---

### Summary

By using scalar projection and geometric reconstruction, HPM implementations avoid the need for explicit ray traversal metadata. The entire ray context - step distance, kernel decay, effective direction - can be recovered on the fly from $\Phi(u)$, $x_i$, and $\mathbf{v}_u$.

These optimizations preserve:

* **Gradient flow** through all projection parameters
* **Modularity** of projection implementation (analytic or discrete)
* **Performance** by reducing memory bandwidth and state dependencies

> *The geometry remembers everything. You only need to ask it correctly.*

---

## F.3 Discrete Rasterization-Compatible Backpropagation

While HPM is defined over continuous fields and projections, practical implementations frequently rely on **discrete rasterization** to traverse the voxel grid along projection rays. Algorithms such as 3D Bresenham or DDA (Digital Differential Analyzer) offer efficient enumeration of intersected voxels, but they introduce **non-differentiable operations** that can block gradient flow.

This section introduces a set of **gradient-preserving techniques** that enable HPM to remain trainable under discrete traversal. These techniques reconstruct differentiable surrogates for all projection parameters and apply analytical gradients where possible.

---

### F.3.1 Gradient of Projection with Respect to Attenuation Parameter $\tau$

Even when the ray path $x_i$ is produced by discrete rasterization, the projection output

$$
T(u) = \sum_i W[x_i] \cdot \exp\left( -\frac{t_i}{\tau} \right)
$$

is differentiable with respect to the attenuation parameter $\tau$, provided the scalar distances $t_i$ are recovered geometrically as in Section F.2.1. The gradient is:

$$
\frac{\partial T(u)}{\partial \tau} = -\frac{1}{\tau^2} \sum_i t_i \cdot W[x_i] \cdot \exp\left( -\frac{t_i}{\tau} \right)
$$

This derivative allows learning of optimal kernel decay rates even under non-differentiable ray paths.

---

### F.3.2 Surrogate Gradients via Entry and Exit Geometry

Discrete direction selection (e.g., index-based routing from a codebook) results in zero gradients through $\mathbf{v}_u$. To recover a differentiable approximation, use the geometry of the voxelized ray:

1. Let $A$ and $B$ be the entry and exit points of the ray through the memory volume.
2. Define a smooth surrogate direction:

$$
\mathbf{v}_{\text{eff}} = \frac{B - A}{\|B - A\|}
$$

This direction is differentiable with respect to $A$ and $B$, which in turn depend on the surface $\Phi(u)$ and the memory volume bounds. This enables gradient flow into viewpoint configuration and implicit direction selection.

---

### F.3.3 Differentiable Codebook Direction Selection

To allow learnable selection of projection directions from a discrete codebook ${\mathbf{v}_k}$, we define a soft mixture:

$$
\mathbf{v}_u = \sum_k a_k(u) \cdot \mathbf{v}_k, \quad \text{where } a_k(u) = \text{Softmax}_k (s_k(u))
$$

Here $s_k(u)$ is a learnable scoring function (e.g., small MLP over $u$). The resulting direction $\mathbf{v}_u$ remains differentiable with respect to all scores $s_k(u)$ and parameters of the scoring model.

Alternative routing strategies include:

* **Gumbel-Softmax** for discrete but differentiable sampling
* **Top-$k$ mixture** with learnable sparsity or entropic penalty

These approaches allow discrete-like projection behavior while preserving differentiability throughout the system.

---

### Summary

Rasterization-based HPM can be trained via:

* **Analytical gradients** for attenuation and decay parameters
* **Geometric surrogates** for direction vectors and projection timing
* **Learnable codebook mixtures** for directional modulation

These techniques ensure that HPM remains end-to-end trainable, even when its forward pass relies on discrete voxel traversal.

> *Differentiability is not a property of algorithms. It is a contract with geometry.*

---

## F.4 Kernel Design and Width Handling

The projection kernel $K(x, \ell_u)$ in HPM governs how memory values along a ray contribute to the projected output $T(u)$. Its spatial profile determines both the **field of influence** and the **attenuation characteristics** of each projection. In practical settings, carefully managing the shape, width, and normalization of the kernel is essential for numerical stability, performance, and interpretability.

This section presents engineering strategies for kernel design that preserve mathematical consistency while enabling flexible, efficient implementations.

---

### F.4.1 Surface-Based Beam Widening via Convolution

When multiple projections share a common direction $\mathbf{v}_u$, the field $T(u)$ can be interpreted as a grid of directional percepts over the projection surface $u \in \mathbb{R}^M$. To simulate a broader beam or lateral blur, we convolve this surface-level map:

$$
\widetilde{T}(u) = \sum_s \omega_s \cdot T(u + s)
$$

where $\omega_s$ are normalized Gaussian weights over a small offset neighborhood ${s}$. This technique avoids the need to dilate the projection kernel volumetrically, which is computationally expensive and harder to differentiate.

This strategy is especially useful when:

* Projections are arranged on a regular grid
* Beam overlap is desired without modifying core geometry
* Direction $\mathbf{v}_u$ is shared across $u$ or selected from a discrete codebook

---

### F.4.2 Separable Kernels for Transverse and Longitudinal Decay

To enable modular control over the kernel shape, we define $K(x, \ell_u)$ as a **separable function** in longitudinal and transverse directions:

$$
K(x, \ell_u) = K_{\parallel}(t) \cdot K_{\perp}(r)
$$

where:

* $t = (x - \Phi(u)) \cdot \mathbf{v}_u$ is the longitudinal distance along the ray
* $r = |x - (\Phi(u) + t \cdot \mathbf{v}_u)|$ is the transverse (orthogonal) deviation

Typical choices include:

* $K_{\parallel}(t) = \exp(-t/\tau)$ or $\exp(-t^2 / (2\sigma_{\parallel}^2))$
* $K_{\perp}(r) = \exp(-r^2 / (2\sigma_{\perp}^2))$

This separation allows fine-grained tuning of projection sharpness and spread in each dimension independently.

---

### F.4.3 Normalization-Free Kernels

In the theoretical formulation, $T(u)$ is an integral weighted by $K$:

$$
T(u) = \int W(x) \cdot K(x, \ell_u) \, dx
$$

In practice, some implementations normalize this by the integral of $K$ to yield an average rather than a sum. However, **normalization introduces unwanted coupling** between spatial support and numerical scale, especially when kernel width varies.

For efficiency and stability, we recommend using **unnormalized kernels**, where:

$$
T(u) \propto \text{total memory mass intersected by the ray}
$$

This preserves linearity and locality. If normalization is required (e.g., for bounded range outputs), it can be applied downstream via an adaptive scaling layer or explicit renormalization.

---

### Summary

Proper kernel handling in HPM balances precision, efficiency, and flexibility. Key strategies include:

* Surface-level convolution to simulate beam widening
* Separable control of longitudinal and transverse decay
* Avoiding normalization inside the projection integral for linear behavior

---

TODO: F5-F8
F.5 Forward-Pass Performance Optimizations
F.6 Gradient Survivability Map
F.7 Implementation Guidelines and Tradeoffs
F.8 Summary of Key Strategies
