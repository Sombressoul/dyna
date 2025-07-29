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

### Step 3: TODO

**Given** $z^*$ and $\vec{d}^*$ from Step 1, **find** $\lambda \in \mathbb{R}$ such that $\ell^* = (z^* - \lambda \cdot \vec{d}^*,\ \vec{d}^*)$ **minimizes the discrepancy** between $T(\ell^*)$ and the target $T^* \in \mathbb{C}^S$.

**In other words**, $\lambda$ marks the boundary beyond which the error between $T(\ell^*)$ and $T^*$ starts to increase.

**Constraints:**

1. Iterative methods are prohibited.
2. Numerical methods (e.g., $\arg\max$, $\arg\min$, etc.) are prohibited.
3. The solution must be in closed form.
4. The solution must be strictly analytic.
5. The solution must be one-step.
6. The solution must be smooth.
7. The solution must be numerically stable.
8. The solution must be practically implementable.
