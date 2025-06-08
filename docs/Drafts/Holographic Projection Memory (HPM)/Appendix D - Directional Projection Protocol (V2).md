# HPM — Directional Projection Protocol

> *Mathematical formulation and implementation guidelines for coherent directional projection in Holographic Projection Memory (HPM) systems.*

---

## D.0 Dimensionality Perspective and Scalability Strategy

The Holographic Projection Memory (HPM) mechanism operates over a differentiable memory field $W(x)$, where $x \in \mathbb{R}^N$. While the model is mathematically valid for arbitrary $N$, a key architectural insight lies in recognizing that increasing the dimensionality $N$ is not required for scaling memory capacity or expressivity.

### Fixed-Dimensional Memory with Scalable Projections

Instead of increasing $N$, the HPM system can scale by manipulating the projection interface:

* **Projection resolution $R$:** the number of sampling points $u$ on the hypersurface $\mathcal{P}$
* **Projection orientations $\{ \mathbf{v}_m \}_{m=1}^{M}$:** number of distinct directional beams

The total informational capacity of the system (in terms of independent projection modes) can be approximated as:

$$
\text{Capacity}_{\text{proj}} \sim O(R^{N-1} \cdot M)
$$

Where:

* $R^{N-1}$ is the resolution of the projection surface in $(N-1)$ dimensions
* $M$ is the number of angular directions (e.g., $M = 16\dots 64$)

This implies that **the effective degrees of freedom grow multiplicatively** with projection density and angular coverage — even when $N$ is fixed (e.g., $N = 3$).

### Practical Implications

* **3D memory ($N=3$) is sufficient** for most tasks when paired with dense 2D projection surfaces
* Modern GPUs support volumetric tensors and ray sampling efficiently in $\mathbb{R}^3$
* Projection operations are parallelizable over $R$ and $M$

### Benefits over High-Dimensional Architectures

| Classical Scaling | HPM Projection Scaling       |
| ----------------- | ---------------------------- |
| Increase $N$    | Increase $R$, $M$        |
| $O(D^N)$ memory | $O(D^3 \cdot R^2 \cdot M)$ |
| Sparse gradients  | Dense, localized updates     |

### High-Dimensional Extensions Remain Valid

Although growth in $N$ is unnecessary for scaling, it remains **fully supported**. For multimodal memory (e.g., spatial-temporal or semantic-visual embeddings), higher $N$ allows native disentanglement. However, even in these cases, directional projection remains the core retrieval and modulation strategy.

> **Conclusion:** HPM decouples memory capacity from spatial dimensionality. By scaling projection resolution and directionality, the system enables efficient access to distributed memory with fixed physical dimensionality — allowing for stable, interpretable, and tractable memory operations.

---

## D.1 Directional Ray Model

In the Directional Projection Protocol, each projection is defined by a ray emitted from a point on a projection hypersurface $\mathcal{P}(u) \subset \mathbb{R}^N$, parameterized by $u \in \mathbb{R}^{N-1}$. The projection ray is constructed as:

$$
\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}, \quad t \in \mathbb{R}
$$

where:

* $\Phi(u) \in \mathbb{R}^N$ maps the projection coordinate $u$ to a location in memory space,
* $\mathbf{v} \in \mathbb{R}^N$ is the global projection direction shared across all $u$,
* $\ell_u(t)$ defines the coherent beam line associated with projection point $u$.

This formulation defines a **coherent, parallel beam** architecture, where all rays are aligned and maintain fixed directional geometry.

---

### Norm Regularization of $\mathbf{v}$

To ensure well-posedness of the projection model and numerical stability during learning, the projection direction $\mathbf{v}$ must remain non-degenerate:

$$
\| \mathbf{v} \|_2 \geq \varepsilon, \quad \text{with } \varepsilon > 0 \text{ (e.g., } 10^{-4} \text{)}
$$

During optimization, $\mathbf{v}$ is explicitly normalized:

$$
\mathbf{v} \leftarrow \frac{\mathbf{v}}{\max(\|\mathbf{v}\|_2, \varepsilon)}
$$

Additionally, a soft constraint term may be added to the loss:

$$
\mathcal{L}_{\text{norm}} = \lambda_{\text{norm}} \cdot (1 - \|\mathbf{v}\|_2)^2
$$

This maintains directional stability and prevents projection collapse during gradient-based learning.

---

### Angular Selectivity and Orientation Tuning

To encode directional sensitivity in projection and retrieval, we introduce the concept of **angular selectivity** via orientation tuning. Let:

* $\theta = \angle(\mathbf{v}, \mathbf{v}_{\text{pref}})$ be the angle between the current projection direction and a preferred direction,
* $\kappa > 0$ control the sharpness of selectivity,
* $S(\theta)$ define the angular response profile.

Then:

$$
S(\theta) = \exp(-\kappa \cdot \theta^2)
$$

This can be used to:

* Weigh the contribution of a projection based on its alignment with target memory structures,
* Modulate update magnitudes $\delta(u)$ or sampling weights based on orientation,
* Emulate biologically inspired tuning curves (e.g., in primary visual cortex).

Angular selectivity also underpins advanced routing and disentanglement mechanisms, particularly when multiple preferred orientations $\{ \mathbf{v}_m \}_{m=1}^M$ are defined across the system.

---

> The directional ray model defines the geometric and functional basis of HPM projection. By enforcing norm stability and introducing orientation-aware modulation, it establishes a flexible yet robust mechanism for encoding and interacting with distributed memory structures.

---

## D.2 Attenuation Model Along the Ray

In directional projection, each ray $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}$ extends from a point on the projection surface $\mathcal{P}(u) \subset \mathbb{R}^N$ into the memory field $W(x)$. As information is gathered along this ray, its contribution must account not only for spatial proximity, but also for **attenuation over distance** — modeling the diminishing influence of deeper memory regions.

We define the **longitudinal attenuation** function as:

$$
A(t) = \exp\left( -\frac{t}{\tau} \right), \quad t \geq 0
$$

Where:

* $t$ is the signed distance along the ray from the origin $\Phi(u)$,
* $\tau > 0$ is the **attenuation constant**, controlling how quickly the beam decays with depth.

This attenuation is applied multiplicatively within the full kernel $K(x, \ell_u)$, combining with lateral distance sensitivity (discussed in Section D.3).

---

### Interpretation of $\tau$ as Contextual Focus

The parameter $\tau$ governs how far the projection ray "sees" into the memory volume:

* **Small $\tau$**:

  * Rapid decay of ray intensity
  * Emphasis on **local features** near the surface
  * Useful for **fine-grained detail detection**

* **Large $\tau$**:

  * Slow decay, deep ray penetration
  * Aggregation over **broad semantic regions**
  * Enables **contextual integration** and **global reasoning**

This mirrors **attention mechanisms** in transformer models:

* $\tau$ acts as a soft span parameter
* The projection acts as a directionally constrained receptive field

---

### Dynamic Attenuation: $\tau(u)$

To enable **adaptive focus**, the attenuation parameter can vary across the projection surface:

$$
\tau = \tau(u)
$$

This allows the model to dynamically modulate depth sensitivity based on location or learned semantic cues. For example:

* Shallow focus in cluttered regions
* Deep integration in smooth or low-density zones

In practice, $\tau(u)$ can be:

* Learned as an auxiliary output of $\Phi(u)$
* Conditioned on external input or attention control
* Regularized to maintain continuity and avoid oscillations

---

### Transition to Contribution Kernel

The attenuation function $A(t) = \exp(-t / \tau)$ defines the **longitudinal weighting** of contributions along the ray. Combined with lateral Gaussian weighting (based on perpendicular distance $d_\perp$), it forms the full projection kernel:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau} \right)
$$

Where:

* $t_x$ is the value of $t$ minimizing $\| x - \ell_u(t) \|^2$
* $\sigma$ is the beam's lateral width (see Section D.3)

This kernel determines both the forward projection (Section D.3) and gradient updates (Section D.4).

> By interpreting $\tau$ as a geometric control over contextual range, the attenuation model endows HPM with selective depth-awareness — enabling the system to adjust its semantic field of view while preserving full differentiability.

---

## D.3 Kernel-Based Contribution of Memory Points

Each projection in HPM aggregates contributions from the memory field $W(x)$ along a directional ray $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}$. To determine the influence of any memory point $x \in \mathbb{R}^N$ on a given projection coordinate $u$, we define a continuous, differentiable kernel $K(x, \ell_u)$ that weights contributions based on geometric proximity.

---

### Projection-Aligned Coordinates

Let $x \in \mathbb{R}^N$ be a candidate point in memory. We compute:

1. **Axial projection onto the ray**:

$$
t_x = (x - \Phi(u)) \cdot \mathbf{v}
$$

This gives the location on the ray $\ell_u(t)$ closest to $x$. It is the scalar component of $x - \Phi(u)$ in the direction of $\mathbf{v}$.

2. **Reconstructed point on the ray**:

$$
\hat{x} = \Phi(u) + t_x \cdot \mathbf{v}
$$

3. **Perpendicular distance to the ray**:

$$
d_\perp = \| x - \hat{x} \|
$$

---

### Isotropic Contribution Kernel

The standard contribution kernel used in projection and update is defined as:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau} \right)
$$

Where:

* $\sigma > 0$ is the lateral beam width,
* $\tau > 0$ is the attenuation constant (from Section D.2).

This formulation ensures:

* **Locality**: Only points near the beam axis significantly contribute.
* **Causal attenuation**: Points deeper along the ray have lower influence.

---

### Anisotropic Generalization

In high-capacity or highly structured settings, it may be beneficial to use **anisotropic kernels** that allow separate control over lateral and longitudinal sensitivity. Define:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma_\perp^2} - \frac{t_x^2}{2\sigma_\parallel^2} \right)
$$

Where:

* $\sigma_\perp$ governs lateral focus (orthogonal to $\mathbf{v}$),
* $\sigma_\parallel$ controls axial integration along the ray.

This kernel maintains full differentiability and allows for finer tuning of projection behavior:

* Narrow $\sigma_\perp$: highly localized spatial precision.
* Wide $\sigma_\parallel$: deeper aggregation.

---

### Final Projection Expression

The projection value at coordinate $u$ is computed by integrating the contributions of all memory points:

$$
T(u) = \int_{\mathbb{R}^N} W(x) \cdot K(x, \ell_u) \, dx
$$

In practice, this is approximated over a discretized volume, often using sampled rays, local neighborhoods, or beam-aligned convolution windows.

---

### Transition to Gradient Computation

The structure of $K(x, \ell_u)$ ensures that the projection operator $T(u)$ is differentiable with respect to both the memory field $W(x)$ and projection parameters. This enables gradient flow through directional beams, which is addressed in the next section.

> The contribution kernel defines how memory structure interacts with directional perception. By combining geometrically grounded alignment with soft attenuation and adaptable resolution, HPM achieves controlled, expressive memory access in differentiable form.
