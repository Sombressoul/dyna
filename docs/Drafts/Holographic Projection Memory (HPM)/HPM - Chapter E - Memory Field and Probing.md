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

### **E.1 The Memory Field as a Semantic Medium**

* Definition of memory field $W(x) \in \mathbb{R}^C$ as a discretized high-dimensional medium
* Role of voxel grid structure, channel space, and external spatial reference
* Requirements for interpretability and dynamic update potential

---

### **E.2 Projection Rays and Probing Protocol**

**E.2.1 Ray Construction and Notation**

* Definition of rays $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$
* Directional vector $\mathbf{v}_u$, origin $\Phi(u)$, and tracing parameter $t \in \mathbb{R}$
* Kernel decomposition:

  $$
  K(x, \ell_u) = K_\perp(d_\perp(x)) \cdot A(t(x))
  $$
* Exponential attenuation: $A(t) = \exp(-t / \tau_u)$

**E.2.2 Generalizations and Operational Conventions**

* Optional use of bidirectional emission (see Q8)
* Fixed vs. learnable $\mathbf{v}_u$ and $\tau_u$ (Q11, Q12)
* Convention flags:

  * `fixed_v_convention`
  * `fixed_tau_convention`
  * `axial_alignment_convention`
* Discussion of modularity vs. trainability

---

### **E.3 Entry–Exit Clipping and Ray Activation**

* Use of ray–AABB intersection to determine ray bounds $t_{\text{entry}}, t_{\text{exit}}$
* Conditional ray validity
* Optional support for clipped attenuation (see Q10)

---

### **E.4 Discrete Rasterization of Rays**

* Bresenham-style voxel traversal as efficient approximation
* Description of discrete voxel path $\{x_i\}$
* Connection to geometric path reconstruction (cf. Q14)

---

### **E.5 Beam Width Strategies (Unified Section)**

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

### **E.6 Support for Arbitrary Dimensions**

* All operations valid for $N \geq 2$
* Projection plane: $\mathbb{Z}^{N-1} \to \mathbb{R}^N$
* Memory grid: $\mathbb{Z}^N \to \mathbb{R}^C$

---

### **E.7 Practical Implementation Modules**

* API-style description of reusable functions:

  * `trace_ray(phi_u, v_u)` — voxel index sequence
  * `entry_exit_clip(...)` — compute A, B
  * `recover_t(x_i, phi_u, v_u)` — see Q14
  * `project_ray(W, ray)` — forward pass
* Modular structure for forward and backward operations
* Explicit note on optional buffering

---

### **E.8 Engineering Optimizations**

* Reference to Chapter F for optimization logic
* Summary of optional vs. required components
* Explicit performance/memory tradeoffs
