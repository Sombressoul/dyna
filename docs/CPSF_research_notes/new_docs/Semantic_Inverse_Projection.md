## Semantic Inverse Projection

This section introduces a mathematically rigorous inverse operation within the CPSF architecture, allowing the inference of a projection coordinate $\ell^* = (z^*, \vec{d}^*) \in \mathbb{T}_\mathbb{C}^N \times \mathbb{S}_\mathbb{C}^{2N-1}$ that best aligns with a desired semantic target $T^* \in \mathbb{C}^S$.

---

### Definition

Let $\mathcal{J} \subset \mathbb{N}$ index a finite collection of field contributions:

$$
C_j := (\ell_j = (z_j, \vec{d}_j),\; \hat{T}_j,\; \sigma_j^{\parallel},\; \sigma_j^{\perp},\; \alpha_j), \quad j \in \mathcal{J}
$$

with:

* $z_j \in \mathbb{T}_\mathbb{C}^N$ — toroidal spatial coordinate,
* $\vec{d}_j \in \mathbb{S}_\mathbb{C}^{2N-1}$ — complex unit direction vector,
* $\hat{T}_j \in \mathbb{C}^S$ — spectral content vector,
* $\alpha_j \in \mathbb{R}_{\ge 0}$ — scalar weight.

Let $T^* \in \mathbb{C}^S$ be a fixed target semantic vector.

---

### Step 1: Envelope Mass

Let $\psi_j^{\mathbb{T}}(z, \vec{d})$ be the periodized envelope defined as:

$$
\psi_j^{\mathbb{T}}(z, \vec{d}) := \sum_{n \in \Lambda} \rho_j\big( \iota(\tilde{z} - \tilde{z}_j + n, \vec{d} - \vec{d}_j) \big)
$$

Then define:

$$
V_j := \int_{\mathbb{T}_\mathbb{C}^N} \int_{\mathbb{S}_\mathbb{C}^{2N-1}} \psi_j^{\mathbb{T}}(z, \vec{d}) \, d\sigma(\vec{d}) \, d\mu(z)
$$

---

### Step 2: Affinity Weight

Define the scalar similarity score:

$$
S_j := |\langle \hat{T}_j, T^* \rangle|
$$

and the unnormalized contribution weight:

$$
\tilde{w}_j := \alpha_j \cdot S_j \cdot V_j \in \mathbb{R}_{\ge 0}
$$

Normalize over the index set $\mathcal{J}$:

$$
w_j := \frac{\tilde{w}_j}{\sum\limits_{k \in \mathcal{J}} \tilde{w}_k} \in [0,1], \quad \sum_{j \in \mathcal{J}} w_j = 1
$$

---

### Step 3: Inverse Projection Coordinate

Define the inverse projection coordinate:

$$
\ell^*(T^*) := (z^*, \vec{d}^*)
$$

with:

* $z^* := \left( \sum\limits_{j \in \mathcal{J}} w_j \cdot z_j \right) \mod \Lambda$
* $\vec{d}^* := \frac{\sum\limits_{j \in \mathcal{J}} w_j \cdot \vec{d}_j}{\left\|\sum\limits_{j \in \mathcal{J}} w_j \cdot \vec{d}_j\right\|} \in \mathbb{S}_\mathbb{C}^{2N-1}$

---

### Properties

* The operation $T^* \mapsto \ell^*(T^*)$ is $\mathcal{C}^\infty$ with respect to all parameters $(\hat{T}_j, z_j, \vec{d}_j, \alpha_j)$.
* The result $\ell^* \in \mathbb{T}_\mathbb{C}^N \times \mathbb{S}_\mathbb{C}^{2N-1}$ lies in the canonical projection space.
* The envelope integrals $V_j \in \mathbb{R}_{> 0}$ are finite for all $j \in \mathcal{J}$ due to rapid decay of $\psi_j^{\mathbb{T}}$.
* The weights $w_j$ induce a probability distribution over contributions $C_j$, jointly informed by geometry and semantic affinity.

---

### Interpretation

The output $\ell^*(T^*)$ identifies the projection coordinate from which the field response is most semantically aligned with $T^*$, according to the current contribution set. It serves as a differentiable, geometry-aware inverse operator within the CPSF framework.
