## CPSF: Definition and Structure of the Memory Field

---

### 1. Non-Existence of the Global Spectrum

Let $\mathcal{M} := \{ C_j \}$ be the set of memory contributions, where each contribution is defined as

$$
C_j := (\ell_j, x_j, \hat{T}_j, \Lambda_j, k_j, \Gamma_j)
$$

with:

* $\ell_j := (\vec{o}_j, \vec{d}_j) \in \mathbb{T}^{2D} \subset \mathbb{C}^{2D}$: projection coordinate,
* $x_j \in \mathbb{T}^N \subset \mathbb{C}^N$: semantic center,
* $\hat{T}_j \in \mathbb{C}^C$: semantic spectral vector,
* $\Lambda_j \in \mathbb{S}_{++}^N$: semantic localization covariance,
* $k_j \in \mathbb{R}^N$: spectral center,
* $\Gamma_j \in \mathbb{S}_{++}^N$: spectral covariance.

Define the global Fourier basis:

$$
\phi_k(x) := e^{2\pi i \langle k, x \rangle}, \quad k \in \mathbb{Z}^N, \quad x \in \mathbb{T}^N := (\mathbb{C}/\mathbb{Z})^N
$$

Define the semantic envelope:

$$
h_j(x) := \sum_{m \in \mathbb{Z}^N} \exp\left( - (x - x_j + m)^\top \Lambda_j^{-1} (x - x_j + m) \right)
$$

Define the emergent contribution coefficient:

$$
\alpha_j := \|\hat{T}_j\|^2 \cdot \int_{\mathbb{T}^N} h_j(x)^2 dx
$$

Define the emergent global spectral coefficients:

$$
\hat{w}_k := \sum_j \alpha_j \hat{T}_j \cdot \hat{h}_{j,k}, \quad \hat{h}_{j,k} := \int_{\mathbb{T}^N} \phi_k(x) h_j(x) dx
$$

#### Definition: Emergent Global Spectrum

The set $\{ \hat{w}_k \}$ does not constitute a stored structure in the memory system. It is not a persistent part of the system state.

Instead, $\hat{w}_k$ is an emergent pseudo-structure, defined functionally and produced as the coherent spectral superposition of contributions $C_j \in \mathcal{M}$.

This quantity is utilized in projection and update operations, but it is not stored as a primary memory object.

---

### 2. Dual Structure of the CPSF Field

Let the CPSF field be defined as:

$$
W : \mathbb{T}^N \to \mathbb{C}^C, \quad W(x) := \sum_j \alpha_j \hat{T}_j h_j(x)
$$

Then $W$ admits a dual interpretation:

#### (a) Geometric Level

* Domain: $\mathbb{T}^N \subset \mathbb{C}^N$: global coordinate space,
* Provides the geometric substrate for semantic contributions,
* Toroidal topology ensures periodicity and continuity.

#### (b) Semantic Level

* Codomain: $\mathbb{C}^C$: complex semantic content,
* The value $W(x)$ arises from the coherent superposition of localized spectral contributions.
