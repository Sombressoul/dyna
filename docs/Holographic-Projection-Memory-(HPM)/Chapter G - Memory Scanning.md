# Chapter G - Memory Scanning: Hierarchical Localization via Projection Bundles

*A coarse-to-fine procedure for localizing meaningful memory regions using directional projections across levels of abstraction.*

---

## Introduction

In high-dimensional memory systems such as Holographic Projection Memory (HPM), unconstrained projection from arbitrary surfaces into dense memory fields is computationally inefficient and semantically unstructured. To address this, HPM introduces a hierarchical scanning mechanism based on Level-of-Detail (LOD) decomposition and structured projection bundles. This enables the system to localize regions of semantic relevance prior to initiating high-resolution projection or Delta-Learning updates.

The scanning mechanism is organized as a multi-resolution inference process:

1. **Begin with a coarse-resolution memory field $W^{(n)}(x)$**, where $n$ denotes the highest (coarsest) LOD level.
2. **Emit structured projection bundles** (typically orthogonal) across a low-resolution surface $\Phi^{(n)}$.
3. **Analyze the distribution of projection responses $T^{(n)}(u)$** to detect semantically active regions.
4. **Recursively refine projection surfaces**, focusing computational attention toward increasingly localized subregions.
5. **Transition to full-resolution projection** at $\text{LOD}_0$ only when localization is complete.

This procedure significantly reduces memory access overhead, supports interpretability, and establishes a foundation for modular, composable memory interaction in HPM-based architectures.

---

## G.1 Motivation

Let $W(x)$ denote a semantic memory field defined over a spatial domain $x \in \mathbb{R}^N$. In the HPM framework, perceptual access to this field occurs via projection:

$$
T(u) = \int W(x) \cdot K(x, \ell_u) \, dx,
$$

where $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$ is the ray emitted from projection coordinate $u$, and $K(x, \ell_u)$ is a spatially localized kernel.

While this formulation is elegant and differentiable, **brute-force projection over the entire field** $W(x)$ is computationally infeasible for realistic dimensions - especially when the volume spans $D_1 \times D_2 \times D_3$ voxels, and thousands of rays are emitted in parallel.

Moreover, the semantic content of $W(x)$ is typically **sparse and topologically segregated**: only a small fraction of memory regions are relevant to any given task, input, or internal state. Projecting indiscriminately across all regions introduces not only inefficiency, but semantic noise.

### Active Attention via Structured Projection

To resolve this, we treat memory access not as a flat scan, but as a **targeted localization problem**. Specifically, we define a recursive, multi-level procedure:

1. **At coarse levels**, emit uniformly distributed rays across $\Phi^{(n)}$, with fixed direction $\mathbf{v}^{(n)}$.
2. **Compute projection responses $T^{(n)}(u)$** and evaluate their spatial distribution.
3. **Select subregions of interest** $\mathcal{R}^{(n)} \subset \Phi^{(n)}$ based on magnitude, entropy, or learned heuristics.
4. **Refine** to $\Phi^{(n-1)}$ over $\mathcal{R}^{(n)}$, and repeat.

This strategy mirrors hierarchical visual attention mechanisms found in biological perception - where coarse foveation identifies candidate regions, and fine-scale mechanisms extract details.

### Beyond Efficiency: Semantic Modularity

Crucially, scanning is not merely an optimization. It enables:

* **Semantic disentanglement**: Disjoint memory regions can encode unrelated modalities or stages of computation.
* **Dynamic specialization**: The same memory field $W(x)$ can be repurposed by changing the projection surface $\Phi(u)$ or ray configuration $\mathbf{v}_u$.
* **Topological modularity**: Large models can share a single memory field across depths and components, with access gated by projection geometry.

> **Note:** This supports (at least theoretically) scalable depth and parameter reuse, providing an architectural alternative to expert-splitting paradigms such as Mixture-of-Experts (MoE).

In summary, scanning introduces a structure-aware, cognitively aligned approach to memory access, enabling both computational tractability and semantic focus - essential for deploying HPM in large-scale systems.

---

## G.2 Rectilinear Volumetric Scanning

> *An efficient projection strategy based on orthogonal ray bundles emitted from the faces of a bounding hypercube.*

---

To localize semantically active regions within the memory field $W(x)$, HPM employs a hierarchical scanning protocol. The first and most efficient stage of this protocol is **Rectilinear Volumetric Scanning**, which operates at the coarsest resolution level $\text{LOD}_n$ and relies on structured ray bundles aligned with the global coordinate axes.

This section formalizes the procedure and its computational advantages.

---

### G.2.1 Projection Geometry

Let the memory volume $W^{(n)}(x)$ be defined on a regular $N$-dimensional grid at level $n$ (e.g., $8^3$ or $16^3$ resolution). Define a bounding hypercube $\mathcal{H}_n \subset \mathbb{R}^N$ enclosing the domain of $W^{(n)}$.

We define six projection surfaces ${\Phi_k}_{k=1}^6$ corresponding to the axis-aligned faces of $\mathcal{H}_n$. Each face $\Phi_k$ emits a **rectangular bundle** of parallel rays into the volume:

$$
\ell_{u}^{(k)}(t) = u + t \cdot \mathbf{v}_k, \quad u \in \Phi_k, \quad t \in [0, L_k],
$$

where:

* $\mathbf{v}_k$ is the inward normal vector of face $\Phi_k$
* $L_k$ is the traversal depth (typically equal to the edge length of $\mathcal{H}_n$)
* $u$ is sampled uniformly across $\Phi_k$ (e.g., an $8 \times 8$ grid)

Each projection bundle thus defines a scan plane through the memory volume, analogous to a tomographic slice.

---

### G.2.2 Projection Evaluation

For each ray $\ell_u^{(k)}$, compute the projection response:

$$
T^{(n)}_k(u) = \int W^{(n)}(x) \cdot K(x, \ell_u^{(k)}) \, dx,
$$

where $K(x, \ell)$ is a ray-aligned spatial kernel, typically a soft Gaussian falloff centered along the ray path. The kernel ensures that only memory regions in the immediate vicinity of the ray contribute to the integral.

This yields six tensor-valued projections:

$$
T^{(n)} = \{T^{(n)}_k\}_{k=1}^6, \quad T^{(n)}_k : \Phi_k \to \mathbb{C}^K \text{ or } \mathbb{R}^C,
$$

depending on whether spectral memory is used.

---

### G.2.3 Localization from Projections

The full projection set $T^{(n)}$ is interpreted as a coarse scan of semantic activity along multiple axes. To infer candidate memory regions for further refinement, we analyze:

* **Magnitude map** $|T_k^{(n)}(u)|$ to detect strong activations
* **Spectral coherence** (phase alignment or frequency density)
* **Entropy of projection responses** to identify structure-rich regions
* **Any arbitrary** function that allows to evaluate the relevance of a region to a search criteria

Subregions $\mathcal{R}_k \subset \Phi_k$ satisfying threshold or learned criteria are backprojected into volume coordinates. The union of these backprojected cones defines a **localization mask** $\Omega^{(n)} \subset \mathcal{H}_n$.

---

### G.2.4 Advantages

Rectilinear volumetric scanning provides:

* **Maximal coverage** with minimal projection complexity
* **Parallelization over axes and rays**
* **Efficient memory reuse** via aligned kernel sampling
* **Simple geometric interpretation**

It serves as the default initial phase of HPM-based inference and allows adaptive refinement in subsequent stages (see Section G.3).

---

## G.3 Semantic Zoom and Recursive Local Projection

> *Progressive refinement of memory access through hierarchical candidate evaluation and localized projection refinement.*

The rectilinear volumetric scan (Section G.2) provides a coarse projection field $T^{(n)}$ capturing directional semantic responses throughout the memory field $W^{(n)}(x)$. However, such global responses lack precision and may conflate unrelated signals along the ray paths.

To achieve high-resolution, context-sensitive access, HPM employs a recursive semantic zoom strategy. This process begins by selecting candidate subregions based on projection activity and then refines their spatial scope and resolution level through recursive projection cube construction.

At each stage, only the globally most relevant subregions are retained (top-$k$ by score), ensuring resource efficiency and semantically focused inference.

---

### G.3.1 From Global Scan to Candidate Regions

Let $T^{(n)} = {T_k^{(n)}(u)}_{k=1}^6$ be the set of coarse projection responses generated by rectilinear scanning at LOD level $n$.

Each face $\Phi_k$ of the hypercube $\mathcal{H}_n$ defines a coordinate grid $u \in \Phi_k$, with each ray $\ell_u^{(k)}$ contributing a spectral or scalar projection value $T_k^{(n)}(u)$.

To extract semantically meaningful memory regions, we define a scoring function:

$$
S: u \mapsto \mathbb{R}, \quad S(u) = f\left(T_k^{(n)}(u)\right),
$$

where $f(\cdot)$ is typically based on:

* Magnitude $|T_k^{(n)}(u)|$
* Spectral entropy
* Coherence across $k$
* Learned or task-specific heuristics

From the set of all ray endpoints across faces $\bigcup_k \Phi_k$, we select a global top-$k$ subset:

$$
\mathcal{U}^{(n)} = \text{top}_k \left\{ u \in \bigcup_k \Phi_k : S(u) \text{ is maximal} \right\}
$$

Each selected $u \in \mathcal{U}^{(n)}$ is mapped to a corresponding volumetric neighborhood in $\mathcal{H}_n$, forming a candidate region:

$$
\mathcal{R}_u^{(n)} = \left\{ x \in \mathcal{H}_n : \|x - \ell_u^{(k)}(t)\| < \varepsilon \right\},
$$

with $\varepsilon$ controlling the local support radius.

The resulting set of candidate regions is:

$$
\mathcal{C}^{(n)} = \{ \mathcal{R}_u^{(n)} : u \in \mathcal{U}^{(n)} \}, \quad |\mathcal{C}^{(n)}| = k
$$

These $k$ regions serve as independent seeds for refinement in the next phase of the hierarchy.

---

### G.3.2 Recursive Refinement of Local Cubes

Each candidate region $\mathcal{R}^{(n)}_j \in \mathcal{C}^{(n)}$ identified in the coarse projection scan defines a localized volume of semantic interest. To refine these regions, HPM constructs a set of higher-resolution subvolumes and repeats the projection process, recursively advancing from level $\text{LOD}_n$ toward $\text{LOD}_0$.

---

#### Construction of Higher-Resolution Cubes

For each $\mathcal{R}^{(n)}_j$, a local projection cube $\mathcal{H}_j^{(n-1)}$ is defined within the memory field $W^{(n-1)}(x)$:

* Centered at the centroid of $\mathcal{R}^{(n)}_j$
* Axis-aligned with the global coordinate system
* Scaled to maintain fixed spatial context at higher resolution

Let $s$ denote the refinement scale factor between LOD levels. Then:

$$
\operatorname{diam}(\mathcal{H}_j^{(n-1)}) = s \cdot \operatorname{diam}(\mathcal{R}_j^{(n)}), \quad s > 1
$$

From each face $\Phi_k^{(j)}$ of $\mathcal{H}_j^{(n-1)}$, we emit a bundle of rays:

$$
\ell_{u}^{(j,k)}(t) = u + t \cdot \mathbf{v}_k, \quad u \in \Phi_k^{(j)}, \quad t \in [0, L_k],
$$

yielding a new set of projections:

$$
T_{j}^{(n-1)} = \{T_{j,k}^{(n-1)}(u)\}_{k=1}^6, \quad T_{j,k}^{(n-1)}(u) = \int W^{(n-1)}(x) \cdot K(x, \ell_u^{(j,k)}) \, dx
$$

---

#### Aggregation and Candidate Selection

Let $\mathcal{P}^{(n-1)}$ denote the set of all projection values across all subregions:

$$
\mathcal{P}^{(n-1)} = \bigcup_{j=1}^k \bigcup_{k=1}^6 \left\{ T_{j,k}^{(n-1)}(u) : u \in \Phi_k^{(j)} \right\}
$$

Define a scoring function $S: u \mapsto \mathbb{R}$ as in Section G.3.1. The top-$k$ ray endpoints are selected globally:

$$
\mathcal{U}^{(n-1)} = \text{top}_k \left\{ u \in \mathcal{P}^{(n-1)} : S(u) \text{ is maximal} \right\}
$$

Each $u \in \mathcal{U}^{(n-1)}$ is mapped to a new candidate region $\mathcal{R}_u^{(n-1)}$ in the same way as before, yielding:

$$
\mathcal{C}^{(n-1)} = \{ \mathcal{R}_u^{(n-1)} : u \in \mathcal{U}^{(n-1)} \}
$$

These become the input for the next recursive iteration.

---

#### Termination Criteria

The recursive semantic zoom proceeds until one of the following conditions is met:

* Minimum LOD is reached: $n = 0$
* Semantic stability: top-$k$ scores converge or saturate
* Resource budget is exceeded (e.g., max depth, time constraint)
* **Any external learned or heuristic function**, including a trained MLP classifier on projection statistics, indicates termination

Upon termination, the current candidate set $\mathcal{C}^{(0)} = { \mathcal{R}_1^*, \dots, \mathcal{R}_k^* }$ defines the final regions to be used in full-resolution HPM projection.

> *Refinement is not a matter of size, but of precision in meaning.*

---

### G.3.3 Adaptive Projection and Final Focus

The terminal step of semantic zoom transitions from hierarchical localization to full-resolution memory interaction. After the recursive refinement procedure (Section G.3.2), the system arrives at a set of final candidate regions:

$$
\mathcal{C}^{(0)} = \{ \mathcal{R}_1^*, \dots, \mathcal{R}_k^* \} \subset W^{(0)}(x),
$$

where each $\mathcal{R}_i^*$ is a tightly localized region of high semantic relevance in the finest memory level.

Within each region $\mathcal{R}_i^*$, a dedicated projection surface $\Phi_i^{(0)}$ is instantiated, serving as the base of high-resolution HPM projection.

---

#### Local Projection Surface Construction

Each projection surface $\Phi_i^{(0)}$ is embedded inside $\mathcal{R}_i^*$ and defines the origin points $u$ of high-resolution rays $\ell_u$.

The geometry of $\Phi_i^{(0)}$ may vary:

* **Planar** (aligned with principal axes)
* **Curved** (informed by local curvature of prior $T(u)$ responses)
* **Learned or task-encoded** shape

Let $u \in \Phi_i^{(0)}$ and $\mathbf{v}_u$ be the per-ray direction vector. Then the projection ray is:

$$
\ell_u(t) = u + t \cdot \mathbf{v}_u, \quad t \in [0, L],
$$

and the corresponding high-resolution HPM projection is:

$$
T_i(u) = \int W^{(0)}(x) \cdot K(x, \ell_u) \, dx
$$

with optional spectral expansion as in Appendix A.

---

#### Adaptive Ray Parameterization

To maximize semantic fidelity, each ray may carry individualized projection parameters:

* $\mathbf{v}_u$ — direction vector
* $\tau_u$ — kernel decay parameter
* $\alpha_u$ — weighting or attention factor

These can be:

* Fixed per layer
* Learned via gradient descent
* Dynamically adjusted based on local features of $\mathcal{R}_i^*$

The full projection for region $\mathcal{R}_i^*$ is therefore a map:

$$
T_i: u \in \Phi_i^{(0)} \mapsto \mathbb{C}^K \text{ or } \mathbb{R}^C
$$

and the total output of HPM scanning becomes a set:

$$
\mathcal{T} = \{ T_1(u), T_2(u), \dots, T_k(u) \}
$$

---

#### Transition to Downstream Processing

Once projection maps $\mathcal{T}$ are computed, HPM scanning is complete. The output $\mathcal{T}$ may be consumed by:

* Delta-Learning modules (see Chapter H)
* Semantic decoders or symbolic heads
* External agents (e.g., controllers, interpreters)

Importantly, HPM does not impose post-projection logic: interpretation of $\mathcal{T}$ is the responsibility of the receiving architecture.

> *HPM ends not with a selection, but with a constellation of structured semantic rays.*
