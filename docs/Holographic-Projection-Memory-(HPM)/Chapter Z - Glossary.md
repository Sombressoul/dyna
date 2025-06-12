# Chapter Z - Glossary


**Holographic Projection Memory (HPM)**  
A geometric memory architecture in which access and modulation of memory content is performed via directional projection rather than indexed lookup. Memory is represented as a continuous, differentiable field $W(x)$, and information retrieval corresponds to integrals over geometrically defined paths.

**Memory Field $W(x)$**  
A differentiable mapping $W : \mathbb{R}^N \rightarrow \mathbb{R}^C$, where $x \in \mathbb{R}^N$ denotes a spatial coordinate in the memory domain, and $C$ is the dimensionality of the latent semantic embedding. This field forms the substrate of storage in HPM.

**Projection Ray $\ell_u(t)$**  
A parametrized line in memory space given by $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$, where $\Phi(u) \in \mathbb{R}^N$ is a point on the projection surface and $\mathbf{v}_u \in \mathbb{R}^N$ is a direction vector. The ray defines the path along which information is integrated from the memory field.

**Projection Surface $\Phi(u)$**  
A differentiable mapping $\Phi : \mathbb{R}^{N-1} \rightarrow \mathbb{R}^N$ that defines a hypersurface in memory space. This surface emits projection rays and serves as the geometric interface for accessing memory.

**Directional Projection $T(u)$**  
A function defined as an integral of memory content along a projection ray:

$$
T(u) = \int W(x) \cdot K(x, \ell_u) \, dx
$$

This quantity represents the system's perceptual response when observing memory along direction $u$.

**Projection Kernel $K(x, \ell_u)$**  
A weighting function determining the contribution of a memory point $x$ to the projection $T(u)$. Typically defined as a product of transverse and longitudinal decay terms:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{t}{\tau} \right)
$$

where $d_\perp$ is the perpendicular distance from $x$ to the ray, and $t$ is the scalar projection onto the ray.

**Projection Error $\delta(u)$**  
The difference between a target projection $T^*(u)$ and the current projection $T(u)$:

$$
\delta(u) = T^*(u) - T(u)
$$

Used as a signal to induce local or structured memory updates.

**Local Plasticity**  
The capacity of the memory field $W(x)$ to be updated in a spatially localized manner in response to projection errors, without requiring global backpropagation. This enables real-time semantic imprinting during inference.

**Topological Divergence**  
A mechanism by which conflicting updates to the memory field cause spatial reorganization rather than destructive interference. Competing semantic patterns separate into distinct regions, preserving both representational integrity and memory capacity.

**Contextual Modulation**  
The ability to access and interpret memory differently by altering the projection surface $\Phi(u)$ or the direction field $\mathbf{v}_u$. Enables flexible, perspective-dependent retrieval without modifying the memory field.

**Interpretability**  
A structural property of HPM wherein memory contents and projection outputs are geometrically organized and thus amenable to direct visualization, semantic analysis, and modular control.

**Semantic Field**  
An interpretation of $W(x)$ as encoding meaningful latent content distributed continuously across space. The spatial organization reflects associative structure and dynamic memory interactions.

**Geometric Access**  
A form of memory access governed by spatial coordinates and direction vectors, as opposed to index- or key-based lookup. Defines a structured and differentiable interface between computation and memory.

**Adaptive Memory**  
A memory system capable of continuous adaptation through local updates during inference, supporting lifelong learning, context sensitivity, and interactive cognition.

**Active Inference Memory**  
A speculative extension of HPM in which memory is modulated during inference via online feedback, treating the memory field $W(x)$ as a dynamic medium that adapts based on contextual projection error. This supports runtime semantic adjustment outside of standard training loops.

**Inference-Time Update Rule**  
A direct modification of the memory field during evaluation:

$$
W(x) \leftarrow W(x) + \alpha \cdot \delta(u) \cdot K(x, \ell_u)
$$

This rule relies on local projection error $\delta(u)$, step size $\alpha$, and the projection kernel. It is used to incrementally shape memory structure based on observed discrepancy.

**Projection-Driven Plasticity**  
The property whereby directional projection not only extracts semantic content but also guides memory revision, creating a bidirectional coupling between perception and adaptation.

**Runtime Memory Imprinting**  
A mechanism by which high-priority perceptual inputs can be encoded directly into memory during inference via the projection error pathway. It allows emergent restructuring without backpropagation.

**Stability vs. Drift**  
A conceptual tension in active memory systems where repeated local updates can yield either convergent adaptation or uncontrolled semantic drift. HPM's update kernel bounds spatial influence but does not guarantee long-term equilibrium.

**Perceptual Target $T^*(u)$**  
An externally or internally defined desired output of a projection, used to drive memory alignment during active inference. It may represent a prediction, label, or self-supervised signal.

**Associative Update**  
A memory update mechanism in which multiple projection errors $\delta(u)$ contribute to the same region of the memory field, leading to interference, reinforcement, or cancellation depending on spatial overlap and kernel shape.

**Additive Superposition**  
The principle that multiple memory updates combine linearly in the field $W(x)$, such that:

$$
W(x) \leftarrow W(x) + \sum_{u} \alpha_u \cdot \delta(u) \cdot K(x, \ell_u)
$$

This supports distributed representation and constructive interference.

**Interference Region**  
The spatial subset of the memory field where the projection kernels from multiple distinct rays $\ell_u$ overlap. Within this region, updates interact nonlinearly, enabling both memory reinforcement and destructive interference.

**Topological Realignment**  
A reconfiguration of memory geometry driven by accumulated update dynamics, where memory content self-organizes spatially to reduce conflict and improve projection consistency.

**Local Consistency Condition**  
A compatibility criterion for updates in shared regions: if two projections agree semantically in an overlapping area, their combined effect reinforces memory structure; if they diverge, conflict arises.

**Gradient Field Flow**  
The interpretation of memory evolution as a continuous flow field in $x$-space, induced by the local accumulation of gradient-like update vectors from multiple projections.

**Vectorial Error Field**  
The spatial distribution of update directionality in $W(x)$, defined by the vector sum of contributions from all active projections. Serves as a differential signal guiding semantic motion of memory.

**Interference Geometry**  
The spatial configuration induced by overlapping projection kernels, determining the nature (constructive, destructive, neutral) of interactions between multiple updates in the memory field.

**Conflict-Induced Reorganization**  
A dynamic reallocation of semantic content in the memory field resulting from incompatible projection-induced updates, driving structural separation to reduce interference.

**Attractor-Like Update Flow**  
A recurrent pattern in memory updates where conflicting semantic regions gradually polarize into stable configurations, resembling dynamical attractors in vector fields.

**Update Compatibility**  
A property of a set of projections such that their superposed updates reinforce one another without introducing contradictory gradients in shared regions.

**Semantic Cancellation**  
A destructive interference phenomenon where projection errors of opposite sign or incompatible content negate each other in overlapping kernel support, reducing update magnitude.

**Semantic Bifurcation**  
The structural splitting of a previously unified semantic representation into distinct spatial regions within the memory field, triggered by persistent disagreement across projection-induced updates.

**Conflict Gradient Axis**  
The principal direction of semantic tension within an interference region, along which the memory field exhibits maximum opposing gradient activity due to incompatible updates.

**Divergence Basin**  
A localized spatial zone within the memory field where early signs of semantic separation emerge, often serving as a nucleus for topological divergence under sustained conflict.

**Longitudinal Distance $t$**  
The scalar projection of a memory point $x$ onto the direction vector $\mathbf{v}_u$, measured from the projection origin $\Phi(u)$:

$$
t = (x - \Phi(u)) \cdot \mathbf{v}_u
$$

Used to control the axial decay of the projection kernel along the ray path.

**Transverse Distance $r$**  
The perpendicular distance from point $x$ to the projection ray $\ell_u(t)$, defined as:

$$
r = \left\| x - (\Phi(u) + t \cdot \mathbf{v}_u) \right\|
$$

Determines radial decay of the projection kernel in directions orthogonal to the ray.

**Separable Projection Kernel**  
A formulation of the kernel $K(x, \ell_u)$ as a product of longitudinal and transverse components:

$$
K(x, \ell_u) = K_\parallel(t) \cdot K_\perp(r)
$$

Enables modular control over the shape of projection influence along and across the ray direction.

**Viewpoint Configuration**  
The parameter pair $(\Phi(u), \mathbf{v}_u)$ defining the origin and direction of a projection ray. It encodes how memory is accessed and perceived geometrically.

**Directional Field $\mathbf{v}_u$**  
A unit or normalized vector associated with each projection point $u$, defining the traversal direction of the ray. It governs how memory is interrogated through space.

**Memory Probing**  
The act of evaluating a directional projection $T(u)$ through kernel-weighted integration over the memory field $W(x)$. Defines localized readout operations.

**Spatial Sampling Window**  
The effective subset of the memory field within which the projection kernel $K(x, \ell_u)$ has non-negligible support. Delimits the region influenced by a single projection.

**Perceptual Readout**  
The interpreted value $T(u)$ resulting from directional projection, treated as the system’s local perception of the memory field at viewpoint $u$.

**Field Resolution**  
The discretization granularity of the memory field $W(x)$, defining how finely memory is spatially sampled and updated.

**Kernel Support Region**  
The region of memory space $x \in \mathbb{R}^N$ where $K(x, \ell_u) > \epsilon$, effectively determining the spatial influence of a given projection ray.

**Ray Marching**  
A discrete procedure for approximating directional projection by sampling along ray paths $\ell_u(t)$ at intervals $\Delta t$. Used for evaluating the integral in $T(u)$ efficiently.

**Sparse Ray Sampling**  
A computational optimization technique where ray traversal skips inactive or zero-contributing regions, reducing unnecessary computations in memory projection.

**Kernel Truncation**  
The process of ignoring the contribution of memory points beyond a cutoff radius (e.g. $3\sigma$) in the projection kernel $K(x, \ell_u)$, used to enforce local support.

**FFT Projection**  
A fast projection technique utilizing Fast Fourier Transforms to approximate convolution-based directional integration under isotropic kernels.

**Passive Projection Mode**  
A read-only configuration of the HPM system in which $T(u)$ is computed without applying updates to the memory field $W(x)$, used in inference or analysis scenarios.

**Memory Scanning**  
A systematic process of evaluating the memory field $W(x)$ by sweeping projection parameters $u$ across a defined domain. Used to explore and extract localized information.

**Scanning Grid**  
A structured arrangement of projection indices $\{u_i\}$ covering the viewpoint domain, used for exhaustive or coarse-to-fine memory traversal.

**Adaptive Scanning**  
A feedback-driven scanning strategy where future projection parameters are selected based on previous perceptual outputs, such as $T(u)$ or $\delta(u)$.

**Stateful Projector**  
A dynamic controller that maintains internal state to regulate sequential projection behavior, enabling context-sensitive and temporally coherent scanning patterns.

**Perceptual Loop**  
A recurrent cognitive-like mechanism wherein projection outputs inform subsequent query configurations, forming a closed cycle of sensing and response.

**Delta-Learning**  
A local, projection-driven learning paradigm in HPM where updates to the memory field $W(x)$ are based on directional projection errors $\delta(u)$ without reliance on global optimization objectives.

**Self-Supervised Target**  
A perceptual target $T^*(u)$ generated internally, often through recurrence or intrinsic evaluation, enabling memory refinement without external labels.

**Attractor Memory State**  
A stable configuration of the memory field $W(x)$ toward which repeated delta updates converge under consistent input patterns.

**Plasticity Kernel**  
The projection kernel $K(x, \ell_u)$ interpreted as a shaping function for memory updates, defining the spatial locality and decay of delta-learning.

**Local Delta Field**  
The aggregate vector field over memory space produced by multiple projection errors $\delta(u)$, used to guide distributed updates to $W(x)$.

**Spectral Projection**  
A form of directional projection interpreted in the frequency domain, where the result $T(u)$ corresponds to a frequency-weighted sampling of the memory field.

**Kernel Frequency Profile**  
The spectral representation of the projection kernel $K(x, \ell_u)$, determining which frequency components of the memory field are emphasized or attenuated.

**Directional Convolution**  
An approximation of the projection operator as a convolution along a ray direction $\mathbf{v}_u$, under the assumption of translation-invariant kernels.

**Spectral Filtering**  
The attenuation or amplification of frequency components in the memory field $W(x)$ induced by the shape of the projection kernel.

**Frequency-Selective Access**  
The ability of HPM to extract specific spatial frequency bands from memory via the geometry and shape of projection kernels.
