## CPSF: Definition of a Memory Contribution $C_j$

Let $D, N, C \in \mathbb{N}$, $\mathbb{T}^N := (\mathbb{C} / \mathbb{Z})^N$ denote the $N$-dimensional complex torus.

---

### Canonical Memory Contribution (Spectral Form)

A memory contribution $C_j$ in CPSF is fully defined in spectral form by its localized projection configuration and semantic content.

Formally, define:

$$
C_j := (\hat{o}_j, \hat{d}_j, \hat{T}_j, \sigma_j^{\parallel}, \sigma_j^{\perp})
$$

where:

* $\hat{o}_j \in \mathbb{C}^N$ — **spectral origin**: complex-valued spectral representation of the geometric origin in frequency domain;
* $\hat{d}_j \in \mathbb{C}^N$, $|\hat{d}_j| = 1$ — **spectral direction**: complex-valued unit direction vector in spectral coordinates;
* $\hat{T}_j \in \mathbb{C}^C$ — **semantic spectrum**: localized spectral semantic content;
* $\sigma_j^{\parallel} \in \mathbb{R}_{>0}$ — **longitudinal attenuation**: determines the scale of projection along the spectral ray direction;
* $\sigma_j^{\perp} \in \mathbb{R}_{>0}$ — **transverse attenuation**: determines the spread orthogonal to the ray.

---

### Interpretation

* The tuple $C_j$ fully defines the spectral position, orientation, and localized semantic excitation of a single memory contribution.
* All derived quantities, including projection kernels, spatial envelopes, geometric covariances, and energy coefficients ($\Sigma_j$, $\Gamma_j$, $\Lambda_j$, $\alpha_j$), are strictly **computed** from this representation.
* All components are defined over complex-valued spectral domains and respect the toroidal topology of the CPSF coordinate space.
