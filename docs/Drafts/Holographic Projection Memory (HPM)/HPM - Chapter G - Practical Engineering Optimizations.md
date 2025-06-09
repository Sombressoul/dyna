## Chapter F — Practical Engineering Optimizations

---

### G.1 Overview

* Purpose of engineering optimizations in HPM
* Balancing efficiency with full differentiability
* Scope of this appendix: geometry, traversal, kernels, memory, gradients

---

### G.2 Geometric Inference Instead of Storage

**G.2.1 Reconstructing $t_i$ via Scalar Projection**

* Formula: $t_i = (x_i - \Phi(u)) \cdot \mathbf{v}_u$
* Eliminates need to store longitudinal time steps
* Fully differentiable and analytic

**G.2.2 Inferring Ray Direction from Entry and Exit Points**

* Use $A = \text{entry point},\; B = \text{exit point}$
* Compute: $\mathbf{v}_\text{eff} = \frac{B - A}{\|B - A\|}$
* Enables surrogate gradients even when $\mathbf{v}_u$ is non-differentiable

**G.2.3 Avoiding Step Index Accumulation**

* Step indices $i$ unnecessary when voxel centers and $\Phi(u)$ are known
* Geometry provides all needed time-distance information

---

### G.3 Discrete Rasterization-Compatible Backpropagation

**G.3.1 Recovering Gradients for $\tau$ Analytically**

* Gradient formula: $\partial T / \partial \tau = -\sum t_i W[x_i] e^{-t_i / \tau} / \tau^2$
* Valid even under discrete traversal (e.g., Bresenham)

**G.3.2 Surrogate Gradients for $\mathbf{v}_u$ and $\Phi(u)$**

* Use $A, B$ to backpropagate through effective geometry
* No need for differentiable traversal logic

**G.3.3 Differentiable Codebook Direction Selection**

* Direction from learned basis: $\mathbf{v}_u = \sum_k a_k \cdot \mathbf{v}_k$
* Soft attention or Gumbel-Softmax allows backpropagation through routing

---

### G.4 Kernel Design and Width Handling

**G.4.1 Surface-Based Beam Width via Convolution**

* Instead of volumetric dilation, apply convolution over $u$-grid:
  $\widetilde{T}(u) = \sum_{s} \omega_s \cdot T(u + s)$
* $\omega_s$: Gaussian weights for lateral spread

**G.4.2 Separable Kernels**

* Use separate kernels for transverse and longitudinal decay
* Enables modular parameter tuning and independent optimization

**G.4.3 Normalization-Free Kernels**

* Remove $\sum K$ normalization for linearity and efficiency
* Output becomes proportional to raw memory-field mass along ray

---

### G.5 Forward-Pass Memory and Performance Optimizations

**G.5.1 Stateless Ray Traversal**

* No need to cache $t_i$ or ray state
* On-demand geometry reconstruction from voxel center and $\Phi(u)$

**G.5.2 Hybrid Tracing: Analytical vs. Discrete**

* Use analytical ray–AABB intersection for $A, B$
* Use Bresenham for voxel path, reconstruct $t_i$ geometrically

**G.5.3 Direction Caching and Lookup**

* Fixed dictionary of directions with precomputed indices
* Reduced compute and easier batching

---

### G.6 Gradient Survivability Map (Ref: Chapter Q Q13)

* Summarizes which gradients are:

  * Always preserved (e.g., $W[x]$)
  * Lost by default but recoverable (e.g., $\tau, \mathbf{v}_u$)
  * Fully lost without geometry (e.g., $t_i$ with hard raster)
* Outlines minimal metadata required to recover each one

---

### G.7 Implementation Guidelines and Tradeoffs

* When to use codebooks vs. dynamic rays
* When to sacrifice exactness for runtime performance
* Mixed-mode support for training and inference

---

### G.8 Summary of Key Strategies

* Tabular or bulleted review of all optimization levers
* Highlight high-impact choices for real-world deployments
