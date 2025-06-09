# Appendix F - Elimination of mathematical ambiguities (Q&A)


## **Q1. Do we compute the response from all memory points for each projection ray?**  

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

## **Q2. If the projection surface is positioned far outside the memory field, do the rays still produce valid responses?**

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

## **Q3. Can each projection ray in HPM have its own direction vector, or must all rays share the same direction?**

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

## **Q4. Can each projection ray have its own attenuation parameter $\tau$, or must all rays share the same value? Does this break the projection model?**

**Yes, each ray can have its own attenuation constant $\tau_u$, and no, it does not break the model.**  
In the HPM projection kernel:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2(x, \ell_u)}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau_u} \right)
$$

the parameter $\tau$ controls **longitudinal attenuation** — i.e., how quickly information fades along the direction of the ray. By default, $\tau$ is assumed to be shared across all rays in a bundle. However, Appendix D.2 explicitly permits this parameter to vary:

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

## **Q5. What is the dimensional relationship between the memory field, the projection hypersurface, and the ambient space in which they coexist?**

The dimensional structure of Holographic Projection Memory (HPM) is fully internal to a single ambient Euclidean space — denoted $\mathbb{R}^N$ — and all components of the system, including memory, projection rays, and projection surfaces, are defined within it. This dimensional alignment is both mathematically grounded and practically essential for differentiable implementation.

---

### **5.1 Memory Field**

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

### **5.2 Projection Hypersurface**

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

### **5.3 Projection Rays**

Each coordinate $u \in \mathbb{R}^{N-1}$ defines a ray:

$$
\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u, \quad t \in \mathbb{R}
$$

* $\Phi(u) \in \mathbb{R}^N$ is the origin point on the projection surface.
* $\mathbf{v}_u \in \mathbb{R}^N$ is a unit direction vector (which may vary per ray).
* The full ray $\ell_u$ is entirely contained in $\mathbb{R}^N$.

Thus, **the rays, memory field, and projection surface all coexist in the same space** — $\mathbb{R}^N$.

---

### **5.4 Dimensional Summary Table**

| Component                             | Domain                     | Dimension   | Contained In                    |
| ------------------------------------- | -------------------------- | ----------- | ------------------------------- |
| Memory Field $W(x)$                 | $x \in \mathbb{R}^N$     | $N$       | $\mathbb{R}^N$                |
| Projection Surface $\mathcal{P}(u)$ | $u \in \mathbb{R}^{N-1}$ | $N - 1$   | $\mathbb{R}^N$                |
| Ray $\ell_u(t)$                    | $t \in \mathbb{R}$       | 1 (per ray) | $\mathbb{R}^N$                |
| Output Projection $T(u)$            | $u \in \mathbb{R}^{N-1}$ | $N - 1$   | Implicit ($\mathbb{R}^{N-1}$) |

---

### **5.5 Practical Confirmation**

This design is consistently used throughout the formal documentation:

* **Appendix D.1:**

  “Each projection is defined by a ray emitted from a point on a projection hypersurface $\mathcal{P}(u) \subset \mathbb{R}^N$, parameterized by $u \in \mathbb{R}^{N-1}$.”

* **Appendix E:**

  “The projected signal at point $u \in \mathbb{R}^{N-1}$ is computed by integrating contributions from the field along the ray $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}$, where $\Phi(u) \in \mathbb{R}^N$.”

* **Appendix F (Q3 & Q4):**
  Confirm that both $\Phi(u)$ and $\mathbf{v}_u$ live in $\mathbb{R}^N$, without requiring higher-dimensional embedding.

---

### **Conclusion:**

The projection surface in HPM is **always an $(N - 1)$-dimensional hypersurface** embedded in the **same ambient space $\mathbb{R}^N$ as the memory field**. This is a direct instantiation of the holographic principle and ensures that projection, gradient flow, and update dynamics all remain fully differentiable and geometrically coherent.

> *In HPM, the memory and its shadows live side-by-side — in the same space, at different scales of meaning.*
