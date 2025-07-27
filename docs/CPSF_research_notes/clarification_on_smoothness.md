**Smoothness and uniform convergence of derivatives.**

Since $\rho_j(w) \in C^\infty(\mathbb{C}^{2N})$ and all terms in the sum defining $\psi_j^{\mathbb{T}}$ decay exponentially in the lattice index $n \in \Lambda$, it follows that the lattice sum:

$$
\psi_j^{\mathbb{T}}(z, \vec{d}) := \sum_{n \in \Lambda} \rho_j\left( \iota(\tilde{z} - \tilde{z}_j + n, \delta \vec{d}) \right)
$$

and all its partial derivatives $\partial^k_{\vec{d}} \psi_j^{\mathbb{T}}$ converge **uniformly** over compact subsets of $\mathbb{T}_\mathbb{C}^N \times \mathbb{S}^{2N-1}_\text{unit}$.

This holds because:

- The directional offset $\delta \vec{d} \in C^\infty(\mathbb{S}^{2N-1}_\text{unit})$;
- The embedding $\iota$ is linear and smooth;
- The composition $\rho_j \circ \iota \circ (\cdot, \delta \vec{d})$ is $C^\infty$ jointly in $z, \vec{d}$;
- The Gaussian decay ensures domination of all derivatives by an integrable majorant.

Hence, $\psi_j^{\mathbb{T}} \in C^\infty(\mathbb{T}_\mathbb{C}^N \times \mathbb{S}^{2N-1}_\text{unit})$, and all error projections and updates defined through it are smooth in direction.

---

**Smooth dependence of $\Sigma_j$ on $\vec{d}_j$.**

The map $\vec{d}_j \mapsto R(\vec{d}_j) \in \mathrm{U}(N)$ is smooth by construction (see: *"Core Terms — Orthonormal Frame"*), and so is the extended block-diagonal frame $\mathcal{R}(\vec{d}_j) \in \mathrm{U}(2N)$.

Since the attenuation matrix $D_j \in \mathbb{R}^{2N \times 2N}$ is constant with respect to $\vec{d}_j$, it follows that the covariance matrix:

$$
\Sigma_j := \mathcal{R}(\vec{d}_j)^\dagger \cdot D_j \cdot \mathcal{R}(\vec{d}_j)
$$

depends **smoothly** on $\vec{d}_j \in \mathbb{S}^{2N-1}_\text{unit}$. Thus, all quantities defined via $\Sigma_j$, including $\rho_j(w)$, $\psi_j^{\mathbb{T}}$, and projection updates $\Delta \hat{T}_j$, inherit this smooth dependence.

All notation used above is consistent with the CPSF core definitions:
- $\mathbb{T}_\mathbb{C}^N$ denotes the complex torus defined by $\mathbb{C}^N / \Lambda$;
- $\mathbb{S}^{2N-1}_\text{unit} \subset \mathbb{C}^N$ is the unit sphere with Hermitian norm;
- $\iota$ is the standard embedding map into $\mathbb{C}^{2N}$;
- $R(\vec{d}) \in \mathrm{U}(N)$, $\mathcal{R}(\vec{d}) \in \mathrm{U}(2N)$, $D_j \in \mathbb{R}^{2N \times 2N}$, and $\Sigma_j \in \mathbb{C}^{2N \times 2N}$ are defined as in *"Core Terms — Geometric Covariance Matrix"*.

---

