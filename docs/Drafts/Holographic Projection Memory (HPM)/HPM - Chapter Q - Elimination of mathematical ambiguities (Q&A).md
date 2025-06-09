# Chapter Q - Elimination of mathematical ambiguities (Q&A)


## Q1. Do we compute the response from all memory points for each projection ray?  

**In theory:** Yes — the projection integral $T(u) = \int W(x) \cdot K(x, \ell_u) \, dx$ spans the entire memory field.

**In practice:** Absolutely not.
The kernel $K(x, \ell_u)$ decays rapidly with distance from the ray. Therefore, we compute contributions **only from a local neighborhood** around the ray — typically within a few multiples of the kernel width $\sigma$, e.g., $d_\perp < 3\sigma$.

**Implementation Strategy:**

* Define a bounding region (cylinder or box) around the ray path.
* Select only memory points $x$ whose centers fall within this region.
* Compute $K(x, \ell_u)$ and sum the weighted contributions.

This reduces complexity from $O(N_{\text{voxels}})$ to a small constant per ray (e.g., 300–600 points), with negligible loss of accuracy — since distant points contribute almost nothing.

> **Conclusion:**  
> The locality of the projection kernel is not a trick — it's a **core design principle**. It ensures efficient, differentiable, and semantically focused memory access.

---

## Q2. If the projection surface is positioned far outside the memory field, do the rays still produce valid responses?  

**Yes — by design.**  
Practical implementations of HPM could use a **bidirectional probing convention**, where each projection coordinate $u$ on the surface emits **two symmetric rays**: one in the direction $\mathbf{v}$, and another in the opposite direction $-\mathbf{v}$.
These rays are evaluated independently and identically, using the same attenuation kernel:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{t}{\tau} \right), \quad t \ge 0
$$

This ensures that even if the surface is outside or parallel to the memory volume, one of the two rays will typically intersect the field — preserving projection fidelity without requiring special handling or directional flipping.

> **Conclusion:**  
> The projection operator is geometrically symmetric but computationally one-sided. Bidirectional emission allows consistent probing from any surface placement while maintaining simple forward-only integration logic.

---

## Q3. Can each projection ray in HPM have its own direction vector, or must all rays share the same direction?  

**Yes, each ray can have its own direction vector.**  
While most theoretical implementations define a global direction $\mathbf{v}$ shared across the entire projection surface for simplicity and efficiency, the mathematical formulation of HPM imposes no such restriction. Each ray $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$ may independently define its own orientation $\mathbf{v}_u \in \mathbb{R}^N$, as long as the direction is differentiable with respect to system parameters and satisfies norm constraints (e.g., $\|\mathbf{v}_u\| \varepsilon$).

The projection kernel remains well-defined:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2(x, \ell_u)}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau} \right)
$$

where both $d_\perp$ and $t_x$ are computed using the local ray $\ell_u$ defined by $\mathbf{v}_u$.

**Benefits of per-ray direction flexibility:**

* Supports angularly selective projection bundles
* Enables adaptive view-dependent memory access
* Allows use of learnable directional codebooks per $u$

**Implementation note:**
Using per-ray directions may increase computational complexity and buffer size, but does not alter the correctness of projection or gradient flow.

> **Conclusion:**  
> HPM supports both globally aligned and ray-specific direction vectors. This generality preserves the full differentiability and semantic structure of projection, and may be exploited for modular, multi-view memory access strategies.

---

## Q4. Can each projection ray have its own attenuation parameter $\tau$, or must all rays share the same value? Does this break the projection model?  

**Yes, each ray can have its own attenuation constant $\tau_u$, and no, it does not break the model.**  
In the HPM projection kernel:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2(x, \ell_u)}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau_u} \right)
$$

the parameter $\tau$ controls **longitudinal attenuation** — i.e., how quickly information fades along the direction of the ray. By default, $\tau$ is assumed to be shared across all rays in a bundle. However, Chapter D.2 explicitly permits this parameter to vary:

“To enable adaptive focus, the attenuation parameter can vary across the projection surface:
$\tau = \tau(u)$”

When $\tau$ becomes a function of $u$, each ray can independently define its attention span or context depth. This is particularly useful for:

* **Learnable depth-of-focus** for different regions
* **Context-aware probing** (e.g., finer resolution near boundaries)
* **Task-specific specialization** (e.g., attention narrowing in dense regions)

**Mathematical correctness:**
All expressions in the projection and gradient computation remain valid as long as:

* $\tau_u 0$
* $\tau_u$ is differentiable (if learned or optimized)

The only affected term is the longitudinal attenuation $e^{-t_x / \tau_u}$, which naturally accommodates variation per ray. Gradients flow cleanly through both $\tau_u$ and the memory field $W(x)$.

> **Conclusion:**  
> The HPM projection model supports ray-specific attenuation parameters without loss of generality, correctness, or differentiability. Adaptive $\tau_u$ provides a principled mechanism for selective depth control and can be used to implement detail-sensitive memory access.

---

## Q5. What is the dimensional relationship between the memory field, the projection hypersurface, and the ambient space in which they coexist?  

The dimensional structure of Holographic Projection Memory (HPM) is fully internal to a single ambient Euclidean space — denoted $\mathbb{R}^N$ — and all components of the system, including memory, projection rays, and projection surfaces, are defined within it. This dimensional alignment is both mathematically grounded and practically essential for differentiable implementation.

---

### 5.1 Memory Field

The memory is defined as a differentiable field:

$$
W : \mathbb{R}^N \to \mathbb{R}^C
$$

Here:

* $N$ is the **intrinsic spatial dimensionality** of the memory field.
* $C$ is the number of **semantic channels** (e.g., features per voxel).
* The memory field exists as a continuous or discretized tensor in $\mathbb{R}^N$.

This space $\mathbb{R}^N$ serves as the ambient coordinate system for all geometric constructs in HPM.

---

### 5.2 Projection Hypersurface

The projection surface (also called the "probe manifold") is defined via a mapping:

$$
\Phi : \mathbb{R}^{N-1} \to \mathbb{R}^N
$$

This defines a $(N-1)$-dimensional **differentiable submanifold** $\mathcal{P}$ embedded in $\mathbb{R}^N$:

$$
\mathcal{P}(u) = \{ \Phi(u) \mid u \in \mathbb{R}^{N-1} \} \subset \mathbb{R}^N
$$

This surface emits projection rays into the memory volume.

The dimensional reduction — from $N$ to $N - 1$ — is deliberate and grounded in the **holographic principle**, where information in a volume is represented on a lower-dimensional surface.

---

### 5.3 Projection Rays

Each coordinate $u \in \mathbb{R}^{N-1}$ defines a ray:

$$
\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u, \quad t \in \mathbb{R}
$$

* $\Phi(u) \in \mathbb{R}^N$ is the origin point on the projection surface.
* $\mathbf{v}_u \in \mathbb{R}^N$ is a unit direction vector (which may vary per ray).
* The full ray $\ell_u$ is entirely contained in $\mathbb{R}^N$.

Thus, **the rays, memory field, and projection surface all coexist in the same space** — $\mathbb{R}^N$.

---

### 5.4 Dimensional Summary Table

| Component                             | Domain                     | Dimension   | Contained In                    |
| ------------------------------------- | -------------------------- | ----------- | ------------------------------- |
| Memory Field $W(x)$                 | $x \in \mathbb{R}^N$     | $N$       | $\mathbb{R}^N$                |
| Projection Surface $\mathcal{P}(u)$ | $u \in \mathbb{R}^{N-1}$ | $N - 1$   | $\mathbb{R}^N$                |
| Ray $\ell_u(t)$                    | $t \in \mathbb{R}$       | 1 (per ray) | $\mathbb{R}^N$                |
| Output Projection $T(u)$            | $u \in \mathbb{R}^{N-1}$ | $N - 1$   | Implicit ($\mathbb{R}^{N-1}$) |

---

### 5.5 Practical Confirmation

This design is consistently used throughout the formal documentation:

* **Chapter D.1:**

  “Each projection is defined by a ray emitted from a point on a projection hypersurface $\mathcal{P}(u) \subset \mathbb{R}^N$, parameterized by $u \in \mathbb{R}^{N-1}$.”

* **Chapter E:**

  “The projected signal at point $u \in \mathbb{R}^{N-1}$ is computed by integrating contributions from the field along the ray $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}$, where $\Phi(u) \in \mathbb{R}^N$.”

* **Chapter Q (Q3 & Q4):**
  Confirm that both $\Phi(u)$ and $\mathbf{v}_u$ live in $\mathbb{R}^N$, without requiring higher-dimensional embedding.

---

### Conclusion:

The projection surface in HPM is **always an $(N - 1)$-dimensional hypersurface** embedded in the **same ambient space $\mathbb{R}^N$ as the memory field**. This is a direct instantiation of the holographic principle and ensures that projection, gradient flow, and update dynamics all remain fully differentiable and geometrically coherent.

> *In HPM, the memory and its shadows live side-by-side — in the same space, at different scales of meaning.*

---

## Q6. What is the relationship between the continuous memory field $W(x)$ and the discrete implementation $W[x]$?

The Holographic Projection Memory (HPM) is formulated in terms of continuous geometric structures to ensure mathematical elegance, differentiability, and physical interpretability. However, practical implementations necessarily operate over discretized domains, such as tensors stored in GPU memory. This question addresses the relationship between the theoretical continuous memory field $W(x)$ and its discrete counterpart $W[x]$ used in computation.

---

### 6.1 Continuous Memory Field — Theoretical Foundation

The memory is defined as a continuous field:

$$
W : \mathbb{R}^N \to \mathbb{R}^C
$$

Here:

* $N$ is the spatial dimension of the memory domain.
* $C$ is the number of semantic channels (e.g., feature dimensions).

The field $W(x)$ is treated as a differentiable function over $\mathbb{R}^N$, supporting:

* Integration along continuous paths (e.g., rays $\ell_u(t)$)
* Continuous optimization with respect to spatial derivatives $\nabla_x W(x)$
* Formal reasoning via calculus and variational principles

This perspective allows HPM to be defined in terms of smooth projection operators:

$$
T(u) = \int_{\mathbb{R}^N} W(x) \cdot K(x, \ell_u) \, dx
$$

and its corresponding gradient flow:

$$
\frac{\partial T(u)}{\partial W(x)} = K(x, \ell_u)
$$

---

### 6.2 Discrete Tensor Field — Practical Realization

In practical machine learning systems, the memory field is implemented as a discretized tensor:

$$
W[x] \in \mathbb{R}^{D_1 \times D_2 \times \dots \times D_N \times C}
$$

Here:

* $x \in \mathbb{Z}^N$ indexes voxel locations in a uniform grid.
* Each voxel $W[x]$ stores a feature vector of dimension $C$.
* The field is typically stored in GPU memory as a high-dimensional array.

The continuous projection integral is approximated by a weighted sum over voxels:

$$
T(u) \approx \sum_{x \in \Omega_u} W[x] \cdot K(x, \ell_u)
$$

where:

* $\Omega_u$ is a small neighborhood of voxels near the ray $\ell_u$ (as determined by kernel support)
* $K(x, \ell_u)$ is computed based on geometric distance from voxel center $x$ to ray $\ell_u$

This discretization is exact in the limit where voxel spacing $\delta \to 0$ and the kernel is sufficiently smooth.

---

### 6.3 Interpretation: Continuum as Limit of Discretization

The continuous model serves as a conceptual and mathematical foundation. The discrete implementation is an approximation of this ideal, where the integral becomes a Riemann sum:

$$
\int_{\mathbb{R}^N} f(x) \, dx \quad \longrightarrow \quad \delta^N \sum_{x \in \mathbb{Z}^N} f(x)
$$

This approximation is valid under standard assumptions:

* $f(x)$ (here, $W(x) \cdot K(x, \ell_u)$) is smooth
* The voxel size $\delta$ is small relative to the kernel width $\sigma$
* The kernel has compact support or decays rapidly outside a bounded region

In this light, the discrete implementation is not a compromise but a **numerical realization** of a continuous model.

---

### 6.4 Implications for Differentiability and Learning

Despite discretization, the projection operator $T(u)$ remains differentiable in all learnable parameters:

* $W[x]$ (memory content)
* $\Phi(u)$ (projection surface)
* $\mathbf{v}_u$, $\tau_u$ (direction, attenuation)

The kernel $K(x, \ell_u)$ is computed analytically and differentiably w\.r.t. all these quantities, and the resulting weighted sum retains gradient flow.

Therefore, the continuous formulation is preserved in spirit and function — enabling backpropagation, active updates, and integration with gradient-based optimization frameworks.

---

### Conclusion:

The continuous memory field $W(x)$ defines the **idealized geometric behavior** of HPM, while the discrete tensor $W[x]$ realizes this behavior in practice. The integral projection becomes a finite sum over spatially indexed voxels, and all learning dynamics remain fully compatible. This dual perspective — continuous for theory, discrete for implementation — is foundational to HPM's design.

> *In HPM, discreteness is not a limitation, but a lens through which continuous geometry becomes computable.*

---

## Q7. Does precomputing ray paths conflict with learnable direction vectors or attenuation?  
  
**Yes — when direction vectors $\mathbf{v}_u$ or attenuation parameters $\tau_u$ are learnable or dynamic, precomputing ray paths becomes partially incompatible with some assumptions of HPM’s rasterization-based optimization strategy. However, this is not a contradiction in the model itself, but a trade-off between runtime flexibility and implementation efficiency.**  

---

### 7.1 Rasterized Projection as Optimization Strategy

In Chapter E, a high-performance implementation strategy is proposed based on rasterizing projection rays over a fixed grid:

* Rays $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}$ are assumed to be **aligned** with fixed directions (e.g., constant $\mathbf{v}$)
* Discrete voxel paths are precomputed using line traversal algorithms (e.g., Amanatides–Woo)
* This permits efficient, GPU-parallel evaluation of contributions

Such optimization assumes **static geometry** of rays — direction and attenuation are fixed during traversal and do not change across iterations or training steps.

---

### 7.2 Learnable Direction Vectors $\mathbf{v}_u$

In more expressive models, the direction of each ray may depend on learnable parameters:

$$
\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u
$$

Here, $\mathbf{v}_u$ may be:

* Learned as a differentiable function of $u$
* Sampled from a directional codebook
* Conditioned on external context or latent states

Since the direction is no longer fixed, the voxel path traversed by $\ell_u$ cannot be computed ahead of time. It must be computed **at runtime**, as part of the forward pass.

---

### 7.3 Learnable or Spatially Varying Attenuation $\tau_u$

Similarly, if attenuation is spatially varying or learned:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau_u} \right)
$$

Then contributions along the ray depend on $\tau_u$, which may differ per projection point $u$. Precomputed paths do not account for this variation, and kernel weights must be recomputed online.

---

### 7.4 Implementation Implications

**Precomputed ray paths are valid only if:**

* $\mathbf{v}$ is shared globally across all $u$
* $\tau$ is constant or piecewise static

**If either parameter is dynamic or learned:**

* Paths must be traced at runtime
* Kernel weights must be recomputed online
* Data structures (buffers, masks) must be constructed dynamically

This does not invalidate the projection model — it simply increases computational complexity and necessitates dynamic batching and traversal.

---

### 7.5 Hybrid and Modular Strategies

To reconcile performance and flexibility, HPM supports hybrid schemes:

* **Fixed directions + learnable attention**:

  * Use a fixed set of $M$ precomputed directions ${\mathbf{v}_m}$
  * Each ray $u$ selects or interpolates among them

* **Blockwise precomputation**:

  * Group rays into local blocks where $\mathbf{v}_u$ is approximately constant
  * Use cached paths for each block

* **Low-resolution probing + refinement**:

  * Use precomputed coarse ray bundles to estimate regions of interest
  * Trace fine-resolution rays dynamically within selected subregions

These approaches preserve efficiency while enabling expressive, differentiable geometry.

---

### Conclusion:

Precomputed ray paths are an effective optimization for static directional projection in HPM, but they are fundamentally incompatible with learnable per-ray direction vectors $\mathbf{v}_u$ or attenuation parameters $\tau_u$. When dynamic control is needed, ray traversal and kernel evaluation must occur at runtime. This trade-off reflects the broader design philosophy of HPM: optimization strategies are modular — the geometric foundation remains intact.

> *When rays can learn where to look, they must also learn how to travel.*

---

## Q8. Bidirectional emission adds an extra channel — what to do with it?

In the bidirectional probing convention, each projection point $u \in \mathbb{R}^{N-1}$ emits two rays:

$$
\ell_u^{(+)}(t) = \Phi(u) + t \cdot \mathbf{v}_u, \quad
\ell_u^{(-)}(t) = \Phi(u) - t \cdot \mathbf{v}_u, \quad t \ge 0
$$

Each ray yields an independent projection value:

$$
T^{(+)}(u) = \int W(x) \cdot K(x, \ell_u^{(+)}) \, dx, \quad
T^{(-)}(u) = \int W(x) \cdot K(x, \ell_u^{(-)}) \, dx
$$

This results in two projected values per location $u$, effectively doubling the channel dimension of $T(u)$.

---

### Interpretation Options

1. **Sum / Average:**

$$
T(u) = T^{(+)}(u) + T^{(-)}(u) \quad \text{or} \quad \frac{1}{2}(T^{(+)} + T^{(-)})
$$

* Suitable when symmetry is assumed
* Ensures invariance to direction sign

2. **Concatenation:**

$$
T(u) = [T^{(+)}(u),\ T^{(-)}(u)] \in \mathbb{R}^{2C}
$$

* Preserves directional separation
* Enables downstream models to learn asymmetries

3. **Selective use:**

* Use only one ray based on prior, learned gate, or task.  

---

### Conclusion:

Bidirectional emission yields two responses per projection point. Whether to sum, concatenate, or select them depends on the symmetry assumptions of the task and architecture. All choices are mathematically valid and implementation-compatible.

> *Bidirectional rays illuminate both sides of meaning — how to interpret them is up to the observer.*

---

## Q9. Should the projection be kernel-normalized?

**Answer:**  
Normalization is optional and depends on the intended semantics:

* **Unnormalized projection:**
  $T(u) = \sum_x W[x] \cdot K(x, \ell_u)$
  — sensitive to local sampling density and ray depth.

* **Normalized projection:**
  $T(u) = \frac{\sum_x W[x] \cdot K(x, \ell_u)}{\sum_x K(x, \ell_u)}$
  — invariant to sampling artifacts; interpretable as a local weighted average.

Both forms are valid. The choice can be exposed as a configurable switch (e.g., `normalize=True`), depending on whether absolute mass or relative structure is more important in the application.

---

## Q10. Is the attenuation function $A(t)$ defined for $t < 0$?

**No — in the current formulation of HPM, attenuation is explicitly defined only for $t \ge 0$.** This is a design convention adopted for simplicity, efficiency, and consistency across bidirectional ray processing. However, the mathematical model itself does not prohibit extension to $t < 0$; such generalizations remain theoretically valid and could be implemented if needed.

---

### 10.1 Current Definition

In Chapter D.2, the longitudinal attenuation function is defined as:

$$
A(t) = \exp\left( -\frac{t}{\tau} \right), \quad t \ge 0
$$

The kernel used in projection is:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau} \right), \quad t_x \ge 0
$$

This definition ensures that only points *in the forward direction* along the ray contribute to the projection.

---

### 10.2 Justification

This one-sided attenuation has several practical advantages:

* Eliminates ambiguity near $t = 0$
* Enables efficient, unidirectional traversal
* Aligns with ray-box clipping and rasterized integration schemes
* Simplifies bidirectional projection by treating each direction independently

In Chapter Q (Q2), bidirectional probing is implemented by evaluating two rays separately:

$$
\ell_u^{(+)}(t), \quad \ell_u^{(-)}(t), \quad \text{with } t \ge 0 \text{ in both cases}
$$

Each direction has its own kernel, evaluated from $t = 0$ forward. No integration is performed over negative $t$.

---

### 10.3 Theoretical Extensions (Optional)

While the current kernel omits $t < 0$, several plausible extensions are theoretically valid:

1. **Symmetric attenuation**:

$$
A(t) = \exp\left( -\frac{|t|}{\tau} \right)
$$

2. **Asymmetric attenuation**:

$$
A(t) = \begin{cases}
\exp(-t/\tau_+) & t \ge 0 \\
\exp(t/\tau_-) & t < 0
\end{cases}
$$

3. **Soft gating**:

$$
A(t) = \sigma(\beta t) \cdot \exp\left( -\frac{|t|}{\tau} \right)
$$

These alternatives allow bidirectional integration along a single ray, or gradient-based learning of asymmetric focus. However, they require kernel and traversal logic changes.

---

### Conclusion:

The attenuation function $A(t)$ is currently defined only for $t \ge 0$, consistent with HPM’s unidirectional ray traversal design. Bidirectional probing is handled via two independent forward-only rays. Extensions to $t < 0$ are mathematically possible, but not included in the default formulation.

> *Attenuation in HPM is a directional lens — defined forward by default, but extendable in theory.*


---

## Q11. What happens to gradients when using rasterized ray traversal? Can they be recovered?

**Answer:** Rasterized ray traversal, such as Bresenham-style grid stepping, introduces discontinuities in ray paths and thus breaks the standard differentiability with respect to geometric parameters like direction vectors and projection origins. However, if intermediate geometric information (e.g., ray entry and exit points) is retained, then surrogate gradients can be reconstructed. This makes the use of discrete ray casting compatible with gradient-based learning.

---

### 11.1 Theoretical Background

In the ideal continuous formulation, the projection is defined by:

$$
T(u) = \int W(x) \cdot K(x, \ell_u) \, dx
$$

where:

* $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$ is the ray emitted from point $\Phi(u)$ in direction $\mathbf{v}_u$.
* $K(x, \ell_u)$ is a smooth kernel centered along the ray.

In practice, this is replaced by a sum over rasterized voxel indices:

$$
T(u) \approx \sum_{x \in \text{Ray}(u)} W[x] \cdot K(x, \ell_u)
$$

Here, `Ray(u)` is computed discretely using Bresenham’s or other grid-based algorithms.

---

### 11.2 Parameters Affected by Rasterization

| Parameter                   | Gradient Loss (Default) | Recoverable? | Notes                                                                                     |
| --------------------------- | ----------------------- | ------------ | ----------------------------------------------------------------------------------------- |
| Memory content $W[x]$    | No loss                 | Yes          | Directly used in sum. Gradients fully preserved.                                          |
| Kernel width $\sigma$     | No loss                 | Yes          | Appears in smooth kernel function.                                                        |
| Attenuation $\tau_u$     | Possibly lost           | Yes          | If $t$ is discretized, gradients may vanish. Recoverable via surrogate time encoding.   |
| Origin $\Phi(u)$          | Lost                    | Partial      | Discrete voxel routing is insensitive to small shifts. Recoverable via entry point $A$. |
| Direction $\mathbf{v}_u$ | Lost                    | Yes          | Recoverable via entry/exit point analysis.                                                |

---

### 11.3 Recovery via Geometric Surrogates

Let $A$ and $B$ be the entry and exit points of the ray inside the memory field:

$$
A = \Phi(u) + t_\text{entry} \cdot \mathbf{v}_u, \quad B = \Phi(u) + t_\text{exit} \cdot \mathbf{v}_u
$$

Then the **effective direction vector** can be reconstructed as:

$$
\vec{v}_{\text{eff}} = \frac{B - A}{\|B - A\|}
$$

This allows a surrogate gradient with respect to $\mathbf{v}_u$ to be computed:

$$
\frac{\partial T(u)}{\partial \mathbf{v}_u} \approx \left( \frac{\partial T(u)}{\partial B} - \frac{\partial T(u)}{\partial A} \right) \cdot \frac{\partial B, A}{\partial \mathbf{v}_u}
$$

This approximation is valid if $A$ and $B$ are computed analytically (not snapped to grid).

---

### 11.4 Fixed Codebook Directions

In codebook-based ray projection:

* The direction $\mathbf{v}_u$ is chosen from a fixed set ${\mathbf{v}_k}$
* Index $k(u)$ is selected either by soft-attention or hard routing

In this case:

* Gradients do not propagate through the discrete voxel path
* But they do propagate through:

  * Codebook vectors $\mathbf{v}_k$
  * Selection weights $a_k(u)$ if attention-based

This makes routing differentiable via meta-parameters:

$$
\mathbf{v}_u = \sum_k a_k(u) \cdot \mathbf{v}_k
$$

---

### 11.5 Time-Attenuation Gradients

If kernel attenuation depends on the longitudinal distance $t$:

$$
K(x, \ell_u) = \exp\left( -\frac{\text{dist}^2_\perp(x)}{2\sigma^2} \right) \cdot \exp\left( -\frac{t(x)}{\tau_u} \right)
$$

Then discretization affects gradients if:

* $t(x)$ is approximated via discrete index (e.g., step number)

Recoverable solution:

* Store continuous $t_x$ values per voxel during traversal
* Compute gradients of $K$ w\.r.t. $\tau_u$ analytically

---

### 11.6 Practical Recommendations

1. **Always cache entry ($A$) and exit ($B$) points per ray.**
2. **Treat direction vectors as meta-parameters and pass gradients through $A, B$ or $k(u)$**.
3. **If using dynamic directions, maintain float-precision during traversal.**
4. **Use differentiable interpolation instead of hard voxel assignment if needed (e.g., via trilinear weights).**

---

### Conclusion

Discrete rasterization breaks differentiability in the forward trace, but if sufficient geometric metadata is retained (entry/exit points, continuous time indices, direction IDs), then gradients can be reconstructed with high fidelity. This enables efficient, high-performance projection with full backward compatibility.

> *A ray need not be smooth to carry a gradient — only transparent enough to let it pass through.*

---

## Q12. Can longitudinal attenuation gradients be recovered analytically under rasterized ray traversal?

**Yes.** When ray traversal is rasterized (e.g., using N-dimensional Bresenham) and attenuation along the ray is defined as an exponential decay with respect to distance, it is possible to recover both the exact contribution and the gradient of the attenuation term. This is feasible under the condition that we retain access to the full voxel trace, ray entry/exit points, and per-step longitudinal positions $t_i$ during forward traversal.

---

### 12.1 Geometric Setup and Assumptions

We consider a coherent, parallel bundle of rays. Each ray is defined by:

* A fixed direction vector $\mathbf{v} \in \mathbb{R}^N$, normalized such that $|\mathbf{v}| = 1$
* An origin point on the projection hyperplane $\Phi(u)$, where $u \in \mathbb{Z}^{N-1}$ is a discrete projection coordinate

The ray enters the memory volume (assumed to be an axis-aligned hypercube) at point $A$ and exits at point $B$, both computed using exact **Ray–AABB intersection**. The physical length of the ray inside the volume is:

$$
L = \|B - A\| = t_{\text{exit}} - t_{\text{entry}}
$$

The ray is rasterized using an N-dimensional Bresenham algorithm, producing a discrete sequence of visited voxels $\{x_i\}_{i=1}^N$, along with corresponding longitudinal positions $\{t_i\} \subset [t_{\text{entry}}, t_{\text{exit}}]$.

---

### 12.2 Projection and Attenuation Definitions

We define the longitudinal attenuation kernel as:

$$
K_\parallel(t_i; \, \tau) = \exp\left( - \frac{t_i}{\tau} \right)
$$

The total projected signal for ray $u$ is:

$$
T(u) = \sum_{i=1}^{N} W[x_i] \cdot K_\parallel(t_i; \, \tau)
\tag{1}
$$

For comparison, we define the "unattenuated" projection (i.e., assuming infinite $\tau$) as:

$$
T_{\text{flat}}(u) = \sum_{i=1}^{N} W[x_i]
\tag{2}
$$

The difference between them reflects the cumulative suppression caused by attenuation:

$$
\Delta T(u) = T_{\text{flat}}(u) - T(u) = \sum_{i=1}^N W[x_i] \cdot \left(1 - \exp\left( -\frac{t_i}{\tau} \right)\right)
\tag{3}
$$

All quantities in this equation are known during forward traversal. Thus, the attenuation contribution can be exactly separated.

---

### 12.3 Gradient with Respect to $\tau$

The projection $T(u)$ is differentiable with respect to $\tau$:

$$
\frac{\partial T(u)}{\partial \tau} = \sum_{i=1}^N W[x_i] \cdot \frac{\partial}{\partial \tau} \exp\left( -\frac{t_i}{\tau} \right) = -\frac{1}{\tau^2} \sum_{i=1}^N t_i \cdot W[x_i] \cdot \exp\left( -\frac{t_i}{\tau} \right)
\tag{4}
$$

This gradient can be computed analytically and efficiently using the cached trace $\{x_i, t_i\}$.

---

### 12.4 Modeling Beam Width via Surface Convolution

Instead of expanding the ray path laterally within the volume (e.g., using a cylinder or Gaussian shell), we implement beam width through **convolution across neighboring projection rays** at the surface level. After computing $T(u)$ via Equation (1), we apply an N-1-dimensional Gaussian convolution over neighboring projection points $u$:

$$
\widetilde{T}(u) = \sum_{s \in \mathcal{N}(0)} \omega_s \cdot T(u + s), \quad \text{where } \omega_s = \exp\left( - \frac{\|s\|^2}{2\sigma_\perp^2} \right)
\tag{5}
$$

This results in a wide-beam effect while preserving coherence across parallel rays. The width parameter $\sigma_\perp$ controls lateral spread and can be optimized.

---

### 12.5 Recovery Scenarios

1. If the direction $\mathbf{v}$ is fixed and $\tau$ is learnable, then Equation (4) gives exact gradients.
2. If $\mathbf{v}$ is dynamically generated, then entry and exit points $A$ and $B$ allow us to recover $\mathbf{v}_{\text{eff}} = (B - A)/|B - A|$ for surrogate differentiation.
3. If $\mathbf{v}$ is selected from a codebook, gradients flow through the selection weights and vector basis.
4. If both $\tau$ and $\mathbf{v}$ are learnable, we combine (2) and (4) via a custom backward rule.

---

### 12.6 Conclusion

In rasterized projection with exact geometric metadata (entry/exit points, voxel sequence, longitudinal distances), longitudinal attenuation contributes a precisely quantifiable and analytically differentiable term to the projection. Even under coarse voxel stepping, attenuation remains smooth and fully trainable — provided that tracing information is retained and leveraged.

> *Even if the path is discrete, the decay along it need not be — and its gradient flows precisely, if we let it.*

---

## Q13. Which gradients are lost under aggressive rasterization, and which can be recovered?

**Answer:** Under extreme projection optimizations—such as Bresenham-based voxel stepping, fixed ray dictionaries, discrete-only forward computation, and no float-level tracing—it is possible to lose several gradients essential for training. However, most can be recovered analytically or through surrogate pathways if the right geometric metadata is retained.

---

### 13.1 Always Retained Gradients

1. **Memory Content $W[x]$** — Used directly in projection sum; gradients always preserved.
2. **Transverse Kernel Parameters (e.g., $\sigma$)** — If Gaussian or similar kernel is applied across voxels, its gradients are preserved, assuming $\text{dist}_\perp(x)$ is computed in float.

---

### 13.2 Lost by Default but Recoverable

3. **Longitudinal Attenuation $\tau_u$** — If $t_i$ values are stored, then:

$$
\frac{\partial T(u)}{\partial \tau} = -\frac{1}{\tau^2} \sum_i t_i \cdot W[x_i] \cdot e^{-t_i/\tau}
$$

This provides a fully analytical surrogate gradient.

4. **Ray Direction $\mathbf{v}_u$** — Can be recovered through entry/exit points $A, B$:

$$
\mathbf{v}_{\text{eff}} = \frac{B - A}{\|B - A\|}
$$

Backpropagation can proceed via surrogate Jacobians.

5. **Projection Origin $\Phi(u)$** — If the ray start point $A$ is retained and not hard-snapped, its contribution to the path can be reintroduced.

---

### 13.3 Recoverable If Architecture Permits

6. **Direction Codebook Index $k$** — Gradients can flow through soft-attention or Gumbel-Softmax layers if the direction is selected from a dictionary.

7. **Kernel Normalization Terms** — If $T(u)$ uses normalization by $\sum K(x)$, gradient still flows but becomes more coupled; if unnormalized, it remains purely linear.

---

### 13.4 Fully Lost Without Special Measures

8. **Time Step $t_i$ Recovery** — If not stored during rasterization (e.g., steps are implicit integers), the attenuation gradient is unrecoverable.

9. **$\Phi(u)$ Influence** — If the ray entry point is hard-snapped to voxel grid and float offset is discarded, the gradient through $\Phi(u)$ is permanently lost.

10. **Dynamic $\mathbf{v}_u$ without Trace** — If rays are not traced at all and a precomputed path is reused for a "nearby" direction, the gradient with respect to the actual $\mathbf{v}_u$ is lost.

---

### 13.5 Summary

Gradient flow under rasterization depends not only on which operations are used in the forward pass, but **what geometric information is preserved and made available to backpropagation**. The key quantities enabling recovery are:

* Entry/exit points ($A$, $B$)
* Longitudinal distances $t_i$
* Voxel visitation list ${x_i}$

> *Rasterization doesn't destroy gradients — but lack of memory does.*

By caching just a few geometric variables, full training signal can be preserved even under highly optimized projection traversal.

---

## Q14. Do we need to store $t_i$ at all? Or can it be reconstructed analytically?

**Answer:** No, we do not need to store $t_i$ during ray traversal. The longitudinal position $t_i$ of each voxel relative to the origin of the ray can be reconstructed geometrically with full precision using only the known spatial coordinates of voxel centers and the origin/direction of the ray.

This insight eliminates the need to explicitly accumulate or cache $t_i$ values during traversal, even in discrete rasterized settings such as Bresenham's algorithm. All that is needed is the geometric configuration of the projection surface, the ray direction, and the memory grid.

---

### 14.1 Geometric Framework

Assume the following:

* $\Phi(u) \in \mathbb{R}^N$ — the origin of the ray for projection coordinate $u$
* $\mathbf{v}_u \in \mathbb{R}^N$ — the (normalized) direction of the ray
* $x_i \in \mathbb{R}^N$ — the center of the $i$-th visited voxel along the ray

Then the longitudinal distance from the ray origin to voxel $x_i$ is given by **scalar projection**:

$$
t_i = \langle x_i - \Phi(u), \, \mathbf{v}_u \rangle
\tag{1}
$$

This is the orthogonal projection of $x_i$ onto the line defined by the ray.

---

### 14.2 Derivation and Justification

Recall that the ray is defined parametrically as:

$$
\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u
$$

For any point $x_i$, the projection onto this line is obtained by solving:

$$
x_i = \Phi(u) + t_i \cdot \mathbf{v}_u + r_i
$$

where $r_i$ is the residual vector orthogonal to $\mathbf{v}_u$. Taking the dot product with $\mathbf{v}_u$ eliminates $r_i$:

$$
\langle x_i - \Phi(u), \, \mathbf{v}_u \rangle = t_i \cdot \langle \mathbf{v}_u, \mathbf{v}_u \rangle = t_i
$$

because $|\mathbf{v}_u| = 1$. Hence Equation (1) is exact.

---

### 14.3 Practical Implementation

In practice, voxel centers $x_i$ are known (or recoverable) from their integer indices and the grid's transform-to-world matrix. Projection origin $\Phi(u)$ is constructed from the surface grid. Thus:

```python
# Given:
# x_i: center of voxel i (float vector in R^N)
# phi_u: origin of ray for projection point u
# v: normalized direction vector

# Reconstruct t_i
t_i = (x_i - phi_u).dot(v)
```

This operation is differentiable, efficient, and does not depend on traversal logic.

---

### 14.4 Benefits

* **No accumulation needed** — works even with fixed-step voxel traversal
* **Fully differentiable** — all components are float-valued and autograd-compatible
* **Compatible with optimization** — $t_i$ is reconstructed only when needed
* **Eliminates numerical drift** — avoids accumulation errors from raster indices

---

### 14.5 When Does This Fail?

This method assumes that the voxel center $x_i$ is accessible. Therefore, it fails only if:

* The traversal does not retain the voxel index
* The voxel layout is irregular or non-grid

In HPM, both conditions are avoided by design. Memory is a regular N-dimensional grid, and voxel IDs are always known.

---

### Conclusion

$\boxed{\text{There is no need to store } t_i.}$

It can always be reconstructed analytically and differentiably from the geometry of the projection surface and memory grid. Even under discrete rasterization, the underlying geometry remains smooth — and so do the gradients.

> *The voxel knows where it lies. The ray knows where it began. The rest is simple geometry.*
