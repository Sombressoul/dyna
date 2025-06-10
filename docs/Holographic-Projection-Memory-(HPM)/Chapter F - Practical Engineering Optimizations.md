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

## F.5 Forward-Pass Memory and Performance Optimizations

HPM projection relies on volumetric traversal and kernel accumulation over a spatial memory field $W(x)$. In large-scale or high-resolution systems, naive implementation of these operations can lead to memory bottlenecks, redundant computation, and excessive branching.

This section presents engineering strategies to accelerate the forward pass without compromising the geometric semantics of HPM.

---

### F.5.1 Stateless Ray Traversal

Traditional ray marching accumulates state at each step: step index $i$, depth $t_i$, current position $x_i$, and accumulated output. To reduce memory pressure and branching, we eliminate explicit ray state.

Using scalar projection (see F.2.1), the relative depth $t_i$ for each voxel $x_i$ is computed on-the-fly:

$$
t_i = (x_i - \Phi(u)) \cdot \mathbf{v}_u
$$

Thus, projection accumulation is implemented as a pure map-reduce over voxels:

$$
T(u) = \sum_{x_i \in \ell_u} W[x_i] \cdot K(t_i)
$$

No additional per-ray metadata is required. This enables batched, SIMD-compatible evaluation.

---

### F.5.2 Hybrid Tracing: Analytical Boundaries + Discrete Marching

To identify the voxel segment of a ray efficiently:

1. Use **analytical ray-box intersection** to compute entry and exit points $A$, $B$ of the ray within the memory volume.
2. Perform **discrete traversal** (e.g., 3D Bresenham) between $A$ and $B$ to enumerate $x_i$.

This hybrid strategy reduces traversal overhead and ensures rays are clipped precisely to the bounds of $W(x)$, improving locality and reducing unnecessary kernel evaluations.

It also enables surrogate gradient flow through $\mathbf{v}_u$ via $A$, $B$ (see F.3.2).

---

### F.5.3 Direction Caching and Lookup

When projection directions $\mathbf{v}_u$ are drawn from a finite codebook ${\mathbf{v}_k}$, it is beneficial to **cache traversal patterns per direction**:

* For each $\mathbf{v}_k$, precompute voxel offsets ${\Delta x_i}$ within a support window.
* At runtime, instantiate ray $\ell_u$ using center point $\Phi(u)$ and offset list indexed by $k$.

This avoids redundant DDA computation and allows high-throughput batching for all $u$ sharing the same $\mathbf{v}_k$.

For dynamic routing (F.3.3), interpolation over multiple cached rays can approximate smooth mixtures.

---

### F.5.4 Direction-Based Batching

When memory projection is applied over a grid of $u$ values (e.g., during scanning or canonical readout), performance can be improved by **sorting or grouping** queries by their associated $\mathbf{v}_k$.

* Within each batch, use the same cached offset trace.
* Kernel evaluations and memory fetches become contiguous.

This exploits memory coherence and reduces GPU thread divergence. On modern hardware, such batching can yield significant speedups.

---

### Summary

Efficient forward execution of HPM is possible through stateless traversal, hybrid tracing, and directional reuse. These methods:

* Eliminate redundant ray-state storage
* Support high-performance batching and SIMD computation
* Maintain full geometric interpretability and projection semantics

---

## F.6 Stateless vs. Stateful Execution

The execution model of Holographic Projection Memory (HPM) determines how rays $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$ are represented, traversed, and consumed during projection. In practical implementations, this leads to a fundamental design question:

> Should the projection engine operate in a **stateless** or **stateful** mode?

This section formalizes both paradigms and establishes a clear interface contract for projection operations.

**Note:** In rasterized or low-level tracing scenarios, there is no fundamental barrier to forwarding the original ray parameters - namely $\Phi(u)$ and $\mathbf{v}_u$ - into the projection kernel. Doing so eliminates the need for surrogate reconstruction and preserves full gradient coverage. The omission of these quantities is a matter of engineering convenience, not theoretical necessity.

---

### F.6.1 Stateless Execution

In **stateless mode**, the projection system receives explicit, fully specified inputs:

* Viewpoint $\Phi(u) \in \mathbb{R}^N$ (starting point of the ray)
* Direction vector $\mathbf{v}_u \in \mathbb{R}^N$ (unit or normalized)
* Projection kernel parameters (e.g., $\tau$, $\sigma$)

Given these inputs, all derived quantities can be computed analytically:

* Scalar depth: $t_i = (x_i - \Phi(u)) \cdot \mathbf{v}_u$
* Kernel decay: $K(t_i) = e^{-t_i / \tau}$ or Gaussian
* Voxel center: $x_i = \Phi(u) + t_i \cdot \mathbf{v}_u$

Stateless execution is functionally pure:

* No internal state or traversal history is required
* Supports vectorized, batched processing
* Enables gradient flow through all inputs

This is the preferred mode when $\Phi(u)$ and $\mathbf{v}_u$ are readily available, either from surface parameterization, codebooks, or neural generators.

---

### F.6.2 Stateful Execution

In **stateful mode**, the projection system manages internal ray traversal and records auxiliary geometric quantities:

* Voxel sequence ${x_i}$ produced via rasterized tracing (e.g., 3D Bresenham)
* Step index $i$, scalar depths $t_i$ (if computed)
* Entry and exit points $A$, $B$ of the ray through the memory volume

In this mode, $\Phi(u)$ and $\mathbf{v}_u$ may be unknown or unavailable during backward pass. Reconstruction techniques (see F.2, F.3) become necessary to recover differentiability:

* $\mathbf{v}_{\text{eff}} = \frac{B - A}{|B - A|}$
* $t_i$ via geometry: $t_i = (x_i - A) \cdot \mathbf{v}_{\text{eff}}$

Stateful mode is often used when tracing is handled by low-level GPU kernels that output only rasterized voxel lists.

---

### F.6.3 Interface Contract for Projection Kernels

To support modular, interchangeable projection backends, we define a minimal interface:

**Required inputs (stateless):**

* $\Phi(u)$: origin point
* $\mathbf{v}_u$: direction
* Kernel hyperparameters ($\tau$, $\sigma$)
* Memory field $W(x)$ (discretized over a lattice)

**Optional internal state (stateful):**

* Rasterized voxel indices ${x_i}$
* Scalar steps $t_i$
* Entry/exit points $A$, $B$

If $\Phi(u)$ and $\mathbf{v}_u$ are provided, no reconstruction is needed. If not, the kernel must either:

* Reconstruct missing quantities from geometric data, or
* Signal inability to propagate gradients (e.g., freeze directional learning)

---

### F.6.4 Recommendations

* Use stateless execution whenever possible. The overhead is minimal (e.g., 24 bytes per ray in FP32) and fully justified by analytical clarity and gradient support.
* When interfacing with rasterized or hardware-accelerated tracing, retain $A$, $B$ to enable surrogate direction recovery.
* Always expose $\Phi(u)$ and $\mathbf{v}_u$ to the projection kernel in training modes that require learning of surface geometry or view adaptation.

---

### Summary

Stateless projection is geometrically explicit, differentiable, and efficient. Stateful projection is occasionally necessary for low-level traversal, but requires careful management of ray metadata. With clear interface boundaries, both modes can coexist within a unified, modular HPM system.

---

## F.7 Gradient Survivability Map

A core requirement of HPM as a differentiable memory system is the ability to propagate gradients through all projection parameters: position, direction, decay, and structure. However, depending on the execution mode (stateless vs. stateful), not all quantities are equally accessible during backpropagation.

This section defines the **gradient survivability** of key parameters and enumerates recovery strategies when native gradients are blocked or unavailable.

---

### F.7.1 Definition: Gradient Survivability

Let $\theta$ be a projection parameter. Its **gradient survivability** is defined as the condition under which

$$
\frac{\partial T(u)}{\partial \theta} \in \mathbb{R}
$$

can be computed:

* **Always preserved** - gradient flows naturally via autodiff
* **Recoverable** - requires geometric reconstruction or surrogate computation
* **Unavailable** - cannot be computed without explicit input or auxiliary state

---

### F.7.2 Survivability Table (Stateless Mode)

In stateless execution (see F.6), all required quantities are explicitly passed. Gradient coverage is thus maximal:

| Parameter             | Survivability | Recovery Needed?  | Notes                                                   |
| --------------------- | ------------- | ----------------- | ------------------------------------------------------- |
| $W[x]$                | Always        | No                | Linear in projection                                    |
| $\tau$, $\sigma$      | Always        | No                | Appears in kernel, fully differentiable                 |
| $\Phi(u)$             | Always        | No                | Passed directly, affects $t_i$                          |
| $\mathbf{v}_u$        | Always        | No                | Passed directly, affects $t_i$                          |
| $t_i$                 | Implicit      | Derived           | Computed from $\Phi(u)$, $\mathbf{v}_u$, $x_i$          |

All projection components are differentiable by design. No reconstruction is needed.

---

### F.7.3 Survivability Table (Stateful Mode)

When tracing is handled via discrete rasterization without explicit $\Phi(u)$ or $\mathbf{v}_u$, gradient access becomes limited:

| Parameter             | Survivability | Recovery Required | Recovery Strategy                                             |
| --------------------- | ------------- | ----------------- | ------------------------------------------------------------- |
| $W[x]$                | Always        | No                | Linear in projection                                          |
| $\tau$, $\sigma$      | Always        | No                | Analytic gradient via $t_i$ (see F.3.1)                       |
| $\Phi(u)$             | Blocked       | Partially         | Reconstruct via $t_i$ and $x_i$                               |
| $\mathbf{v}_u$        | Blocked       | Yes               | Surrogate: $\mathbf{v}_{\text{eff}} = \frac{B-A}{\|B-A\|}$    |
| $t_i$                 | Blocked       | Yes               | $t_i = (x_i - A) \cdot \mathbf{v}_{\text{eff}}$               |

Gradient survivability in stateful execution depends not on the tracing method itself, but on whether the entry and exit points $A$, $B$ of the ray are retained. Without access to this geometric context, gradients with respect to direction or position cannot be recovered - at least within the scope of the reconstruction techniques considered here.

---

### F.7.4 Recommendations

* **Use stateless execution** when training or when $\Phi(u)$ and $\mathbf{v}_u$ are learnable.
* **In stateful mode**, retain $A$ and $B$ per ray to enable surrogate gradients.
* **Avoid hard index selection** (e.g., for codebook directions) without soft routing (see F.3.3).
* **Cache $t_i$** only if $\Phi(u)$ and $\mathbf{v}_u$ are unavailable; otherwise compute on-the-fly.

---

### Summary

Gradient flow in HPM is guaranteed under stateless execution, where all projection-defining variables are explicit.  

In rasterized or stateful scenarios, **there is no fundamental barrier to passing $\Phi(u)$ and $\mathbf{v}_u$ directly into the projection kernel**. If this information is retained or reconstructed, all required gradients can propagate without approximation. The choice to omit such data is a performance heuristic - not a theoretical necessity.

---

## F.8 Implementation Guidelines and Tradeoffs

Engineering an efficient and trainable HPM system requires balancing multiple constraints: memory access patterns, gradient propagation, execution throughput, and architectural flexibility. This section summarizes practical recommendations and design tradeoffs that arise when implementing projection and update mechanisms.

---

### F.8.1 Preferred Data Contract

The minimal, fully differentiable interface for projection consists of:

* $\Phi(u) \in \mathbb{R}^N$: ray origin (viewpoint surface)
* $\mathbf{v}_u \in \mathbb{R}^N$: ray direction (unit or normalized)
* $W(x)$: memory field (discretized)
* Kernel parameters ($\tau$, $\sigma$, etc.)

This stateless contract ensures that all geometric quantities ($t_i$, $x_i$, direction vectors) can be computed on-the-fly, preserving gradient flow and enabling pure-function ray evaluation.

Tradeoff: passing full geometric data per ray slightly increases memory bandwidth (e.g., 24 bytes/ray in FP32), but eliminates the need for reconstruction and surrogate gradients.

---

### F.8.2 Codebook Direction Handling

If $\mathbf{v}_u$ is selected from a discrete codebook ${\mathbf{v}_k}$:

* Use soft routing: $\mathbf{v}_u = \sum_k a_k(u) \cdot \mathbf{v}_k$
* Gradients flow through weights $a_k(u)$ (learnable, e.g., via softmax or Gumbel-Softmax)
* Optional: cache precomputed ray traces for each $\mathbf{v}_k$

Tradeoff: increases flexibility and learnability, but introduces soft interpolation error and runtime mixture cost.

---

### F.8.3 Tracing Backend Selection

| Mode                | Characteristics                                     | Gradient Support          |
| ------------------- | --------------------------------------------------- | ------------------------- |
| Stateless           | Requires $\Phi(u)$ and $\mathbf{v}_u$ per ray       | Full                      |
| Stateful + $A,B$    | Entry/exit points cached                            | With surrogate recovery   |
| Stateful only       | Voxel list only, no geometry                        | Limited or blocked        |

Recommendation: Prefer stateless tracing with geometry passed explicitly. Only use pure stateful tracing when hardware constraints dominate.

---

### F.8.4 Memory and Batch Efficiency

* Group rays by shared $\mathbf{v}_k$ to exploit memory coherence
* Use precomputed offset lists for codebook directions
* Cache projection kernels in LUTs for repeated decay profiles
* Evaluate $K(t_i)$ in-place to avoid memory allocation per ray

Tradeoff: these optimizations improve runtime speed and batching, but may reduce flexibility in learnable direction scenarios.

---

### F.8.5 Update Modes and Streaming Writes

In update mode (e.g., Delta-Learning):

* Streaming writes to $W(x)$ can be performed incrementally
* Each ray contributes $\delta(u) \cdot K(x_i, \ell_u)$ at its visited voxels
* Use atomic operations or scatter-add to merge contributions

Tradeoff: atomicity ensures correctness, but may slow down parallelism; buffer-based accumulation may improve throughput but requires more memory.

---

### Summary

Efficient HPM implementation depends on carefully managing projection interface, memory access, and execution context. Stateless geometry contracts and modular direction selection maximize differentiability and compatibility. Tradeoffs between flexibility, performance, and learnability should be evaluated based on task and deployment environment.

> *Optimization is not about removing complexity. It is about placing it where it belongs.*

---

## F.9 Summary of Key Strategies

This section consolidates the core engineering strategies for implementing Holographic Projection Memory (HPM) systems in a way that is efficient, scalable, and fully differentiable. Each method below is grounded in the geometry of projection and aligned with the analytical formulation of HPM.

---

### F.9.1 Projection Interface Design

| Component         | Recommendation                                      | Rationale                                   |
| ----------------- | --------------------------------------------------- | ------------------------------------------- |
| $\Phi(u)$         | Pass explicitly per ray                             | Enables stateless execution, full gradients |
| $\mathbf{v}_u$    | Pass explicitly or as a codebook mixture            | Differentiable direction control            |
| $t_i$             | Recompute from geometry                             | Avoids state accumulation                   |
| $x_i$             | Generate via $\Phi(u) + t_i \cdot \mathbf{v}_u$     | Stateless ray evaluation                    |

---

### F.9.2 Gradient Preservation

| Scenario                        | Strategy                                          | Gradient Status |
| ------------------------------- | ------------------------------------------------- | --------------- |
| Stateless execution             | Pass $\Phi(u)$ and $\mathbf{v}_u$                 | Full            |
| Stateful + entry/exit ($A,B$)   | Recover $\mathbf{v}_{\text{eff}}$ from $A,B$      | Surrogate       |
| Rasterized only (no $A,B$)      | Avoid or approximate                              | Blocked         |

---

### F.9.3 Kernel Handling

| Design Choice | Strategy                                | Benefit              |
| ------------- | --------------------------------------- | -------------------- |
| Beam widening | Surface-level convolution               | Fast, differentiable |
| Shape control | Separable longitudinal/transverse decay | Tunable locality     |
| Normalization | Avoid in-kernel; apply post-projection  | Preserves linearity  |

---

### F.9.4 Performance Optimization

| Technique                     | Application Context                 | Effect                                |
| ----------------------------- | ----------------------------------- | ------------------------------------- |
| Directional batching          | Codebook-based rays                 | Improves cache coherence              |
| Ray trace caching             | Shared $\mathbf{v}_k$ directions    | Eliminates redundant tracing          |
| LUT kernel evaluation         | Repeated decay profiles             | Accelerates kernel application        |
| Scatter-add with accumulation | Update phase                        | Ensures correctness under parallelism |

---

### F.9.5 Execution Mode Selection

| Mode              | Preferred When                          | Limitations             |
| ----------------- | --------------------------------------- | ----------------------- |
| Stateless         | Training, learnable geometry            | Slight memory overhead  |
| Hybrid ($A,B$)    | Low-level GPU kernels with ray clipping | Requires extra caching  |
| Fully rasterized  | Inference-only with fixed geometry      | No direction gradients  |

---

### Final Observation

The core insight of this chapter is that **efficiency and differentiability are not at odds** - provided that geometric structure is made explicit and preserved throughout execution. Stateless ray contracts, analytical reconstruction, and modular control over projection dynamics allow HPM systems to scale without sacrificing precision or gradient reach.

> *Every projection carries structure. Optimization is the art of preserving it.*
