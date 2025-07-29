## Semantic Inverse Projection

Define $\ell^*(T^*) = (z^*, \vec{d}^*) \in \mathbb{T}_\mathbb{C}^N \times \mathbb{S}^{2N-1}_\text{unit}$ as the coordinate maximizing alignment with target $T^* \in \mathbb{C}^S \setminus \{0\}$, based on field contributions $C_j = (\ell_j, \hat{T}_j, \sigma_j^{\parallel}, \sigma_j^{\perp}, \alpha_j)$ (see: *"Core Terms — Field Contribution"*).

---

### Step 1: Directional modulation

Let $\mathcal{J} \subset \mathbb{N}$ index a finite set:

$$
C_j = (\ell_j = (z_j, \vec{d}_j), \hat{T}_j, \sigma_j^{\parallel}, \sigma_j^{\perp}, \alpha_j), \quad j \in \mathcal{J}.
$$

Compute:

$$
\vec{d}^* = \frac{\sum\limits_{j \in \mathcal{J}} \alpha_j \langle \hat{T}_j, T^* \rangle \vec{d}_j}{\left\| \sum\limits_{j \in \mathcal{J}} \alpha_j \langle \hat{T}_j, T^* \rangle \vec{d}_j \right\|}
$$

Then:

$$
\gamma_j = \alpha_j \langle \hat{T}_j, T^* \rangle \cdot \delta \vec{d}_j
$$

where $\delta \vec{d}_j \in T_{\vec{d}_j} \mathbb{S}^{2N-1}_\text{unit}$ is the directional offset between $\vec{d}^*$ and $\vec{d}_j$ (see: *"Core Terms — Directional Offset and Angular Distance"*).

---

### Step 2: Semantic-weighted envelope accumulation

Define the accumulated semantic influence envelope:

$$
\Psi(z) := \sum_{j \in \mathcal{J}} \|\gamma_j\|^2 \cdot \psi_j^{\mathbb{T}}(z, \vec{d}^*), \quad z \in \mathbb{T}_\mathbb{C}^N
$$

Then define the spatial coordinate $z^* \in \mathbb{T}_\mathbb{C}^N$ as the barycenter of $\Psi(z)$ with respect to the Haar measure (see: *"Core Terms — Projection Space Measure"*):

$$
z^* := \frac{\displaystyle \int_{\mathbb{T}_\mathbb{C}^N} \Psi(z) \cdot z \; d\mu(z)}{\displaystyle \int_{\mathbb{T}_\mathbb{C}^N} \Psi(z) \; d\mu(z)}
$$

---

### Step 3: Directional boundary projection

Let $z^* \in \mathbb{T}_\mathbb{C}^N$ and $\vec{d}^* \in \mathbb{S}^{2N-1}_\text{unit}$ be defined by Step 2. Define a ray:

$$
z(\lambda) := z^* - \lambda \cdot \vec{d}^*, \quad \lambda \in \mathbb{R}_{\geq 0}
$$

Define the projected field response:

$$
T^{\text{proj}}(\lambda) := \sum_{j \in \mathcal{J}} \alpha_j \cdot \psi_j^{\mathbb{T}}(z^*, \vec{d}^*) \cdot e^{- \frac{1}{2} \lambda^2 / \sigma_j^2} \cdot \hat{T}_j
$$

Define the normalized deviation:

$$
\tau^2(\lambda) := \frac{ \|T^*\|^2 - 2 \Re \langle T^{\text{proj}}(\lambda), T^* \rangle + \|T^{\text{proj}}(\lambda)\|^2 }{ \|T^* - T(z^*, \vec{d}^*)\|^2 }
$$

Then define the boundary-projected coordinate:

$$
z^{**} := z^* - \lambda^* \cdot \vec{d}^*
$$

where $\lambda^* \in \mathbb{R}_{>0}$ is chosen such that:

$$
\lambda^* := \underset{\lambda \ge 0}{\argmin} \; \|T^{\text{proj}}(\lambda) - T^*\|^2
$$

Alternatively, $\lambda^*$ may be determined as the unique root of:

$$
\frac{d}{d\lambda} \tau^2(\lambda) = 0
$$

---

### About $\lambda^*$

The $\lambda^*$ is the unique positive value that satisfies:

$$
\sum_{j \in \mathcal{J}} \frac{1}{\sigma_j^2} e^{- \frac{1}{2} (\lambda^*)^2 / \sigma_j^2} \cdot \Re \langle \alpha_j \psi_j^{\mathbb{T}}(z^*, \vec{d}^*) \hat{T}_j, T^* \rangle = \frac{1}{2} \sum_{j, k \in \mathcal{J}} \left( \frac{1}{\sigma_j^2} + \frac{1}{\sigma_k^2} \right) \langle \alpha_j \psi_j^{\mathbb{T}}(z^*, \vec{d}^*) \hat{T}_j, \alpha_k \psi_k^{\mathbb{T}}(z^*, \vec{d}^*) \hat{T}_k \rangle e^{- \frac{1}{2} (\lambda^*)^2 \left( \frac{1}{\sigma_j^2} + \frac{1}{\sigma_k^2} \right)}
$$

After simplifying the given expressions, $\lambda^*$ is defined as the positive root of the following transcendental equation:

$$
\boxed{
\sum_{j \in \mathcal{J}} a_j e^{- \frac{1}{2} (\lambda^*)^2 a_j} c_j = \frac{1}{2} \sum_{j, k \in \mathcal{J}} (a_j + a_k) \langle b_j, b_k \rangle e^{- \frac{1}{2} (\lambda^*)^2 (a_j + a_k)}
}
$$

where the terms are defined as:

- $a_j = \frac{1}{\sigma_j^2}$
- $b_j = \alpha_j \psi_j^{\mathbb{T}}(z^*, \vec{d}^*) \hat{T}_j$
- $c_j = \Re \langle b_j, T^* \rangle$

### Explanation of Terms:
- **$\mathcal{J}$**: The index set over which the sums are taken.
- **$\sigma_j^2$**: Variance associated with index $j$.
- **$\alpha_j$**: A scalar coefficient for index $j$.
- **$\psi_j^{\mathbb{T}}(z^*, \vec{d}^*)$**: A function or transformation evaluated at optimal points $z^*$ and $\vec{d}^*$.
- **$\hat{T}_j$**: A vector or operator associated with index $j$.
- **$T^*$**: A reference vector or target variable.
- **$\Re \langle \cdot, \cdot \rangle$**: The real part of an inner product.
- **$\langle b_j, b_k \rangle$**: The inner product between $b_j$ and $b_k$.
