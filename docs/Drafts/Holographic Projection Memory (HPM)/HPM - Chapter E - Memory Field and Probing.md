# Chapter E — Memory Field and Probing

> *This chapter introduces the internal structure of the Holographic Projection Memory (HPM) field and presents a rasterized formulation of projection that is both mathematically principled and computationally tractable. It enables high-throughput memory access via parallelizable, discrete geometric traversal, generalizing naturally to arbitrary memory dimensionality.*

---

Artificial memory systems often rely on discrete keys, flat embeddings, or address-based retrieval. HPM, in contrast, is based on a geometric principle: information is stored across a continuous spatial field, and accessed via directional integration along semantically meaningful rays. In this formulation, memory becomes not an indexable container, but a **reflective medium** — one in which meaning is encountered through structured probing.

Formally, the HPM memory field is defined as a differentiable function $W : \mathbb{R}^N \to \mathbb{R}^C$, mapping coordinates in continuous space to semantic vectors. Each point $x \in \mathbb{R}^N$ encodes latent content — a high-dimensional vector $W(x) \in \mathbb{R}^C$ — that contributes to a projected response when intersected by an external probe. These probes take the form of directed rays $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}$, where $\Phi(u) \in \mathbb{R}^N$ is the origin of the ray (sampled from a projection surface), and $\mathbf{v} \in \mathbb{R}^N$ is a unit direction vector.

The projected signal at point $u \in \mathbb{R}^{N-1}$ is computed by integrating contributions from the field along the ray $\ell_u$, typically via a Gaussian kernel centered on the ray trajectory. This defines a directional projection operator:

$$
T(u) = \int_{\mathbb{R}^N} W(x) \cdot K(x, \ell_u) \, dx
$$

where $K(x, \ell_u)$ is a continuous, localized kernel that weights memory points based on perpendicular distance to the ray and longitudinal attenuation.

While this integral formulation is elegant and differentiable, its direct implementation is computationally expensive: evaluating $K(x, \ell_u)$ for all $x \in \mathbb{R}^N$ is infeasible on real hardware, particularly when $W$ is represented as a large, discrete tensor. Therefore, an efficient approximation strategy is required.

This chapter develops a **rasterized projection framework**, in which rays are discretized, clipped to the memory volume, and traversed using voxel-efficient line-drawing algorithms (e.g., N-dimensional Bresenham). The resulting discrete paths allow projection values to be computed from a sparse neighborhood of relevant memory points. Furthermore, lateral influence — required to simulate beam width — is implemented either via local convolution in memory space or by smoothing over adjacent rays in the projection domain.

This rasterized strategy offers multiple benefits:

* It eliminates the need for ray–voxel intersection tests at runtime
* It permits fully batched and parallelized execution on GPU hardware
* It generalizes naturally to memory fields of dimension $N \geq 2$
* It preserves the geometric structure and locality properties of HPM

> *Although we speak intuitively of shadows or projections, HPM operates via active semantic reflection: rays illuminate latent content, and the response is what returns from the distributed field.*

In the following sections, we formalize the memory field, define its discretization and structure, and develop the complete rasterized projection procedure, including entry–exit clipping, ray traversal, smoothing, and generalization to arbitrary dimensions.

---

### E.1 The Memory Field as a Semantic Medium

The Holographic Projection Memory (HPM) system is built upon a continuous, high-dimensional memory field, denoted as:

$$
W : \mathbb{R}^N \rightarrow \mathbb{R}^C
$$

Here, $x \in \mathbb{R}^N$ represents a spatial coordinate in an $N$-dimensional ambient memory space, and $W(x) \in \mathbb{R}^C$ is a $C$-dimensional latent vector associated with that position. The field $W(x)$ can be interpreted as encoding local semantic content distributed over a differentiable geometric substrate.

---

#### Discrete Representation

In practice, the continuous field $W(x)$ is realized as a discretized tensor:

$$
W[x] \in \mathbb{R}^{D_1 \times D_2 \times \dots \times D_N \times C},
$$

where $x \in \mathbb{Z}^N$ indexes the center of a voxel in a regular grid. The spatial resolution $(D_1, \dots, D_N)$ is typically uniform, and the value $W[x]$ represents the semantic vector (or feature embedding) associated with the corresponding voxel.

This formulation maintains full compatibility with the continuous model in the limit as grid resolution increases:

$$
W(x) \approx W[\lfloor x / \delta \rfloor], \quad \text{for small voxel size } \delta > 0.
$$

Interpolation techniques (e.g., trilinear or kernel-based) may be used to recover smooth responses between voxel centers.

---

#### Role of the Channel Space $\mathbb{R}^C$

Each memory value $W(x) \in \mathbb{R}^C$ is a vector of latent components or channels. The dimensionality $C$ is not fixed by geometry, and is instead chosen based on the expressiveness required by the application:

* $C = 1$: scalar density field (e.g., for volumetric masking)
* $C \gg 1$: high-dimensional latent codes (e.g., for semantic memory or reconstruction tasks)

The interpretation of channels is model-dependent, but generally assumed to be differentiable and continuous. Projection operations defined in later sections (e.g., $T(u)$) are computed independently for each channel and later aggregated or convolved as needed.

---

#### Spatial Reference and Alignment

The memory field $W$ is embedded in a shared ambient coordinate system $\mathbb{R}^N$, which is also used to define projection rays $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$. All geometric constructs (projection surfaces, rays, attenuation kernels) operate within this space, ensuring full differentiability and compatibility with kernel-based integration schemes.

The origin, orientation, and extent of $W$ are not constrained by the model. Memory may occupy a fixed bounding box or dynamically evolve in form, as long as the probing operators maintain geometric consistency.

---

#### Trainability and Dynamic Update

The memory tensor $W[x]$ is generally **trainable**, and may be updated:

* **Via standard backpropagation** through projection layers $T(u)$, with gradients $\partial T / \partial W$ given by the projection kernel $K(x, \ell_u)$
* **Locally during inference**, using error-based correction:

$$
W(x) \leftarrow W(x) + \alpha \cdot \delta(u) \cdot K(x, \ell_u),
$$

as introduced in Chapter A.3 and formalized in Chapter D.5. This enables **inference-time plasticity** and selective memory adaptation.

The locality of the projection kernel $K$ ensures that updates only affect regions semantically aligned with the probing ray, thereby preventing widespread corruption of unrelated memory content.

---

#### Interpretability and Modularity

Because $W$ is defined over a spatial lattice with fixed indexing and structured semantics, its internal organization can be visualized, monitored, and analyzed. This interpretability is essential for:

* Debugging and visualization of memory responses
* Understanding the structure of projection shadows $T(u)$
* Supporting modular and compositional learning paradigms

In summary, the memory field $W(x)$ in HPM acts as a structured, high-capacity, differentiable medium — enabling coherent interaction between spatial geometry and latent semantics. Its design supports both stability under projection and adaptability under learning, forming the backbone of the HPM architecture.

---

### E.2 Projection Rays and Probing Protocol

Information stored in the HPM memory field is not accessed through discrete indexing or content-based similarity, but rather via structured geometric **probing** — the process of integrating semantic contributions along **directed rays** that traverse the memory volume. These rays originate from a projection surface embedded in the ambient space and are parameterized by coordinates $u \in \mathbb{R}^{N-1}$.

Each such coordinate defines a **ray** $\ell_u(t) \in \mathbb{R}^N$, constructed from a surface mapping $\Phi(u)$ and a direction vector $\mathbf{v}_u$, forming the basis of directional memory access. The response of the memory field to probing is given by the integral of local content $W(x)$ modulated by a kernel $K(x, \ell_u)$ that localizes interaction both **laterally** (via distance to the ray path) and **longitudinally** (via depth attenuation).

This section formalizes the construction of projection rays, introduces conventions for directionality and attenuation, and outlines the operational modes for defining and modulating memory access paths. The probing protocol described here provides the geometric foundation for all downstream projection and update operations in HPM.

---

#### E.2.1 Ray Construction and Notation

In the Holographic Projection Memory (HPM) framework, access to the memory field $W(x)$ is mediated by directed probing rays that originate from a projection hypersurface and traverse the ambient memory space $\mathbb{R}^N$. Each ray is defined in terms of a projection coordinate $u \in \mathbb{R}^{N-1}$, which indexes a location on the $(N-1)$-dimensional projection surface $\mathcal{P}$.

---

**Ray Parameterization**

Given a differentiable mapping $\Phi : \mathbb{R}^{N-1} \rightarrow \mathbb{R}^N$, the ray corresponding to coordinate $u$ is defined as:

$$
\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u, \quad t \in \mathbb{R},
$$

where:

* $\Phi(u) \in \mathbb{R}^N$ is the origin point of the ray on the projection surface,
* $\mathbf{v}_u \in \mathbb{R}^N$ is a unit-length direction vector associated with $u$,
* $t \in \mathbb{R}$ is the scalar tracing parameter that defines the position along the ray.

The direction vector $\mathbf{v}_u$ may be shared across all $u$ (global direction) or vary locally (per-ray orientation). In either case, normalization is enforced to ensure well-posedness:

$$
\| \mathbf{v}_u \|_2 = 1.
$$

---

**Geometric Role**

Each ray $\ell_u(t)$ defines a line in $\mathbb{R}^N$ along which contributions from the memory field are aggregated. The ray acts as a **semantic probe**, aligned along direction $\mathbf{v}_u$ and rooted at surface point $\Phi(u)$. Only points in the vicinity of this ray contribute meaningfully to the projection, due to the localized nature of the projection kernel $K(x, \ell_u)$.

---

**Kernel Decomposition**

The contribution of a memory point $x \in \mathbb{R}^N$ to the projection associated with $\ell_u$ is modulated by a kernel function $K(x, \ell_u)$, defined as a product of two components:

$$
K(x, \ell_u) = K_\perp(d_\perp(x)) \cdot A(t(x)),
$$

where:

* $d_\perp(x)$ is the perpendicular distance from point $x$ to the ray $\ell_u$,

* $K_\perp$ is a transverse weighting kernel, typically Gaussian:

  $$
  K_\perp(d) = \exp\left(-\frac{d^2}{2\sigma^2}\right),
  $$

* $t(x)$ is the axial projection of $x$ onto the ray, i.e., the value of $t$ that minimizes $\| x - \ell_u(t) \|^2$, given by:

  $$
  t(x) = (x - \Phi(u)) \cdot \mathbf{v}_u,
  $$

* $A(t)$ is a longitudinal attenuation function that suppresses distant contributions along the ray.

---

**Exponential Attenuation**

The default attenuation model is exponential decay with a ray-specific attenuation parameter $\tau_u > 0$:

$$
A(t) = \exp\left(-\frac{t}{\tau_u}\right), \quad t \geq 0.
$$

This form ensures that the kernel $K(x, \ell_u)$ is directionally causal: points further along the ray have progressively reduced influence. The decay rate $\tau_u$ controls the effective receptive depth of the ray and may be:

* Fixed globally,
* Learned per ray coordinate $u$, or
* Adaptively modulated based on external context.

This directional and attenuated formulation of projection rays enables localized, anisotropic sampling of the memory field and forms the basis for the differentiable projection operator $T(u)$ described in subsequent sections.

---

#### E.2.2 Generalizations and Operational Conventions

The ray construction framework introduced in Section E.2.1 supports a variety of generalizations and implementation conventions that allow for flexible trade-offs between expressiveness, efficiency, and differentiability. These choices influence how rays interact with the memory field and are critical for designing practical HPM systems.

---

**Bidirectional Emission**

By default, each projection coordinate $u \in \mathbb{R}^{N-1}$ emits a **single ray** $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$, with $t \geq 0$. However, the system can optionally be configured for **bidirectional emission**, where both forward and backward rays are evaluated:

$$
\ell_u^{(+)}(t) = \Phi(u) + t \cdot \mathbf{v}_u, \quad
\ell_u^{(-)}(t) = \Phi(u) - t \cdot \mathbf{v}_u, \quad t \geq 0.
$$

Each ray is processed independently with identical attenuation and kernel structure. The resulting pair of projection values $T^{(+)}(u), T^{(-)}(u)$ can be:

* Summed or averaged: $T(u) = T^{(+)}(u) + T^{(-)}(u)$
* Concatenated: $T(u) = [T^{(+)}(u), T^{(-)}(u)] \in \mathbb{R}^{2C}$
* Used selectively, based on gating or learned preference

**Bidirectional emission is particularly useful when the projection surface lies *inside* the boundary of the memory field.**  

In this configuration, the projection acts as a symmetric bidirectional probe that softly illuminates the memory field from within. The exponential attenuation applied in both directions ensures that the contribution from nearby regions is emphasized, while distant points fade smoothly, avoiding sharp cutoff artifacts.

This results in:

* **Smooth gradient flow** during backpropagation, as there are no abrupt edges or truncation in the integration path.
* **Compact and centered projections** that highlight a local region of the memory field — ideal for introspective decoding or context-dependent memory focusing.
* A geometrically coherent way to implement **soft attention over a spatial neighborhood**, without requiring explicit masks or sampling windows.

This approach is especially powerful when the projection surface is narrow or spatially compressed — it effectively “resonates” within a bounded volume, providing a **focused semantic snapshot** of a localized memory region.


---

**Fixed vs. Learnable Parameters**

The ray definition depends on two key quantities:

* **Direction vector** $\mathbf{v}_u$
* **Attenuation constant** $\tau_u$

These may be treated as **fixed hyperparameters** or **learnable functions**.

**Fixed configuration:**

* $\mathbf{v}_u = \mathbf{v}$ for all $u$, with $\mathbf{v} \in \mathbb{R}^N, \|\mathbf{v}\|_2 = 1$
* $\tau_u = \tau \in \mathbb{R}_{>0}$

This mode supports high-performance implementations (e.g., ray caching, rasterized traversal) and simpler batching.

**Learnable configuration:**

* $\mathbf{v}_u = \mathbf{v}(u)$ or sampled from a learnable codebook: $\mathbf{v}_u = \sum_k a_k(u) \cdot \mathbf{v}_k$
* $\tau_u = \tau(u)$, either directly predicted or derived from local features

Learnable variants enhance flexibility and allow the system to adapt its field-of-view and depth perception based on task demands. See Chapters D.2, D.7, and Q3–Q4 for implementation and gradient recovery strategies.

---

**Convention Flags**

To simplify experimentation and implementation, a set of **convention flags** is defined to control ray behavior:

* `fixed_v_convention`: Assume all rays share a fixed global direction $\mathbf{v}$
* `fixed_tau_convention`: Use a constant attenuation value $\tau$ for all rays
* `axial_alignment_convention`: Align $\mathbf{v}$ with one of the principal axes (e.g., $x$, $y$, $z$) for memory-efficient traversal

These flags are **not semantic constraints**, but **engineering heuristics** that guide implementation without altering the core geometry.

---

**Modularity vs. Trainability**

The choice between fixed and adaptive ray parameters reflects a deeper trade-off:

* **Fixed configurations** prioritize efficiency, reproducibility, and clarity. They are well-suited for rasterized projection (see Chapter F) and deployment in constrained environments.
* **Trainable configurations** support task-specific specialization, dynamic focus, and contextual behavior. They align more naturally with gradient-based learning systems.

In practice, hybrid schemes are most likely to be used, where:

* Direction vectors $\mathbf{v}_u$ are drawn from a learned dictionary
* Attenuation $\tau_u$ is predicted from the projection surface $\Phi(u)$
* Only a subset of rays are fully dynamic, while others remain fixed

This modular design ensures that HPM can operate across a spectrum of regimes — from static geometric probing to fully adaptive perceptual routing.

---

### E.3 Entry–Exit Clipping and Ray Activation

* Use of ray–AABB intersection to determine ray bounds $t_{\text{entry}}, t_{\text{exit}}$
* Conditional ray validity
* Optional support for clipped attenuation (see Q10)

---

### E.4 Discrete Rasterization of Rays

* Bresenham-style voxel traversal as efficient approximation
* Description of discrete voxel path $\{x_i\}$
* Connection to geometric path reconstruction (cf. Q14)

---

### E.5 Beam Width Strategies (Unified Section)

**E.5.1 Projection-Surface-Based Convolution (Preferred)**

* Beam width modeled as post-projection convolution:

  $$
  \widetilde{T}(u) = \sum_{s} \omega_s \cdot T(u + s)
  $$
* Advantages in parallelization and gradient propagation

**E.5.2 Memory-Volume-Based Sampling (Alternative)**  

* Useful in case of **independent $\mathbf{v}_u$** per each ray
* Beam widening via **convolution over neighboring voxels** in the memory grid
* More precise for non-parallel or adaptive ray fields
* Requires **more memory and interpolation**, less efficient than surface-based method
* Recommended only when lateral coherence across rays cannot be assumed

---

### E.6 Support for Arbitrary Dimensions

* All operations valid for $N \geq 2$
* Projection plane: $\mathbb{Z}^{N-1} \to \mathbb{R}^N$
* Memory grid: $\mathbb{Z}^N \to \mathbb{R}^C$

---

### E.7 Practical Implementation Modules

* API-style description of reusable functions:

  * `trace_ray(phi_u, v_u)` — voxel index sequence
  * `entry_exit_clip(...)` — compute A, B
  * `recover_t(x_i, phi_u, v_u)` — see Q14
  * `project_ray(W, ray)` — forward pass
* Modular structure for forward and backward operations
* Explicit note on optional buffering

---

### E.8 Engineering Optimizations

* Reference to Chapter F for optimization logic
* Summary of optional vs. required components
* Explicit performance/memory tradeoffs
