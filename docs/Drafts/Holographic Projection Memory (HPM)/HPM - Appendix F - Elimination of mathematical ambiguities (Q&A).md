# Appendix F - Elimination of mathematical ambiguities (Q&A)


**Q1. Do we compute the response from all memory points for each projection ray?**  

> **In theory:** Yes — the projection integral $T(u) = \int W(x) \cdot K(x, \ell_u) \, dx$ spans the entire memory field.
>
> **In practice:** Absolutely not.
> The kernel $K(x, \ell_u)$ decays rapidly with distance from the ray. Therefore, we compute contributions **only from a local neighborhood** around the ray — typically within a few multiples of the kernel width $\sigma$, e.g., $d_\perp < 3\sigma$.
>
> **Implementation Strategy:**
>
> * Define a bounding region (cylinder or box) around the ray path.
> * Select only memory points $x$ whose centers fall within this region.
> * Compute $K(x, \ell_u)$ and sum the weighted contributions.
>
> This reduces complexity from $O(N_{\text{voxels}})$ to a small constant per ray (e.g., 300–600 points), with negligible loss of accuracy — since distant points contribute almost nothing.
>
> **Conclusion:**
> The locality of the projection kernel is not a trick — it's a **core design principle**. It ensures efficient, differentiable, and semantically focused memory access.

---

**Q2. If the projection surface is positioned far outside the memory field, do the rays still produce valid responses?**

> **Yes — by design.**
> HPM uses a **bidirectional probing convention**, where each projection coordinate $u$ on the surface emits **two symmetric rays**: one in the direction $\mathbf{v}$, and another in the opposite direction $-\mathbf{v}$.
> These rays are evaluated independently and identically, using the same attenuation kernel:
>
> $$
> K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{t}{\tau} \right), \quad t \ge 0
> $$
>
> This ensures that even if the surface is outside or parallel to the memory volume, one of the two rays will typically intersect the field — preserving projection fidelity without requiring special handling or directional flipping.
>
> **Conclusion:**
> The projection operator is geometrically symmetric but computationally one-sided. Bidirectional emission allows consistent probing from any surface placement while maintaining simple forward-only integration logic.

---

**Q3. Can each projection ray in HPM have its own direction vector, or must all rays share the same direction?**

> **Yes, each ray can have its own direction vector.**
> While most implementations define a global direction $\mathbf{v}$ shared across the entire projection surface for simplicity and efficiency, the mathematical formulation of HPM imposes no such restriction. Each ray $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$ may independently define its own orientation $\mathbf{v}_u \in \mathbb{R}^N$, as long as the direction is differentiable with respect to system parameters and satisfies norm constraints (e.g., $\|\mathbf{v}_u\| > \varepsilon$).
>
> The projection kernel remains well-defined:
>
> $$
> K(x, \ell_u) = \exp\left( -\frac{d_\perp^2(x, \ell_u)}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau} \right)
> $$
>
> where both $d_\perp$ and $t_x$ are computed using the local ray $\ell_u$ defined by $\mathbf{v}_u$.
>
> **Benefits of per-ray direction flexibility:**
>
> * Supports angularly selective projection bundles
> * Enables adaptive view-dependent memory access
> * Allows use of learnable directional codebooks per $u$
>
> **Implementation note:**
> Using per-ray directions may increase computational complexity and buffer size, but does not alter the correctness of projection or gradient flow.
>
> **Conclusion:**
> HPM supports both globally aligned and ray-specific direction vectors. This generality preserves the full differentiability and semantic structure of projection, and may be exploited for modular, multi-view memory access strategies.

---

**Q4. Can each projection ray have its own attenuation parameter $\tau$, or must all rays share the same value? Does this break the projection model?**

> **Yes, each ray can have its own attenuation constant $\tau_u$, and no, it does not break the model.**
> In the HPM projection kernel:
>
> $$
> K(x, \ell_u) = \exp\left( -\frac{d_\perp^2(x, \ell_u)}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau_u} \right)
> $$
>
> the parameter $\tau$ controls **longitudinal attenuation** — i.e., how quickly information fades along the direction of the ray. By default, $\tau$ is assumed to be shared across all rays in a bundle. However, Appendix D.2 explicitly permits this parameter to vary:
>
> > “To enable adaptive focus, the attenuation parameter can vary across the projection surface:
> > $\tau = \tau(u)$”
>
> When $\tau$ becomes a function of $u$, each ray can independently define its attention span or context depth. This is particularly useful for:
>
> * **Learnable depth-of-focus** for different regions
> * **Context-aware probing** (e.g., finer resolution near boundaries)
> * **Task-specific specialization** (e.g., attention narrowing in dense regions)
>
> **Mathematical correctness:**
> All expressions in the projection and gradient computation remain valid as long as:
>
> * $\tau_u > 0$
> * $\tau_u$ is differentiable (if learned or optimized)
>
> The only affected term is the longitudinal attenuation $e^{-t_x / \tau_u}$, which naturally accommodates variation per ray. Gradients flow cleanly through both $\tau_u$ and the memory field $W(x)$.
>
> **Conclusion:**
> The HPM projection model supports ray-specific attenuation parameters without loss of generality, correctness, or differentiability. Adaptive $\tau_u$ provides a principled mechanism for selective depth control and can be used to implement detail-sensitive memory access.
