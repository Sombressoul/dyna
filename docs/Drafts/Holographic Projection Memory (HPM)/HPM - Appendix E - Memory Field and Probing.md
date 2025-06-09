# Appendix E — Memory Field and Probing

> *This appendix introduces the internal structure of the Holographic Projection Memory (HPM) field and presents a rasterized formulation of projection that is both mathematically principled and computationally tractable. It enables high-throughput memory access via parallelizable, discrete geometric traversal, generalizing naturally to arbitrary memory dimensionality.*

---

Artificial memory systems often rely on discrete keys, flat embeddings, or address-based retrieval. HPM, in contrast, is based on a geometric principle: information is stored across a continuous spatial field, and accessed via directional integration along semantically meaningful rays. In this formulation, memory becomes not an indexable container, but a **reflective medium** — one in which meaning is encountered through structured probing.

Formally, the HPM memory field is defined as a differentiable function $W : \mathbb{R}^N \to \mathbb{R}^C$, mapping coordinates in continuous space to semantic vectors. Each point $x \in \mathbb{R}^N$ encodes latent content — a high-dimensional vector $W(x) \in \mathbb{R}^C$ — that contributes to a projected response when intersected by an external probe. These probes take the form of directed rays $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}$, where $\Phi(u) \in \mathbb{R}^N$ is the origin of the ray (sampled from a projection surface), and $\mathbf{v} \in \mathbb{R}^N$ is a unit direction vector.

The projected signal at point $u \in \mathbb{R}^{N-1}$ is computed by integrating contributions from the field along the ray $\ell_u$, typically via a Gaussian kernel centered on the ray trajectory. This defines a directional projection operator:

$$
T(u) = \int_{\mathbb{R}^N} W(x) \cdot K(x, \ell_u) \, dx
$$

where $K(x, \ell_u)$ is a continuous, localized kernel that weights memory points based on perpendicular distance to the ray and longitudinal attenuation.

While this integral formulation is elegant and differentiable, its direct implementation is computationally expensive: evaluating $K(x, \ell_u)$ for all $x \in \mathbb{R}^N$ is infeasible on real hardware, particularly when $W$ is represented as a large, discrete tensor. Therefore, an efficient approximation strategy is required.

This appendix develops a **rasterized projection framework**, in which rays are discretized, clipped to the memory volume, and traversed using voxel-efficient line-drawing algorithms (e.g., N-dimensional Bresenham). The resulting discrete paths allow projection values to be computed from a sparse neighborhood of relevant memory points. Furthermore, lateral influence — required to simulate beam width — is implemented either via local convolution in memory space or by smoothing over adjacent rays in the projection domain.

This rasterized strategy offers multiple benefits:

* It eliminates the need for ray–voxel intersection tests at runtime
* It permits fully batched and parallelized execution on GPU hardware
* It generalizes naturally to memory fields of dimension $N \geq 2$
* It preserves the geometric structure and locality properties of HPM

> *Although we speak intuitively of shadows or projections, HPM operates via active semantic reflection: rays illuminate latent content, and the response is what returns from the distributed field.*

In the following sections, we formalize the memory field, define its discretization and structure, and develop the complete rasterized projection procedure, including entry–exit clipping, ray traversal, smoothing, and generalization to arbitrary dimensions.

---
