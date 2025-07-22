### Functional Role of $\Sigma_j$

The matrix $\Sigma_j \in \mathbb{C}^{2N \times 2N}$ defines the covariance of an anisotropic complex-valued Gaussian envelope on the extended projection space $\mathbb{C}^{2N} \cong \mathbb{C}^N_{\text{pos}} \oplus \mathbb{C}^N_{\text{dir}}$, aligned with the projection ray direction $\vec{d}_j$.

Let the projection coordinate $\ell_j := (z_j, \vec{d}_j) \in \mathbb{T}_\mathbb{C}^N \times \mathbb{C}^N$. To define localization relative to $\ell_j$, we introduce a lift $\tilde{z} \in \mathbb{C}^N$ of the toroidal position $z \in \mathbb{T}_\mathbb{C}^N$, and analogously $\tilde{z}_j \in \mathbb{C}^N$ such that $\tilde{z} \equiv z$, $\tilde{z}_j \equiv z_j$ modulo $\Lambda$.

Define the canonical embedding:

$\iota : \mathbb{C}^N \times \mathbb{C}^N \rightarrow \mathbb{C}^{2N}, \quad (u, v) \mapsto w := \begin{bmatrix} u \\ v \end{bmatrix}$

and set:

$w := \iota(\tilde{z} - \tilde{z}_j, \vec{d} - \vec{d}_j) \in \mathbb{C}^{2N}$

To explicitly define $\Sigma_j$, we introduce the unitary matrix $R(\vec{d}_j) \in \mathrm{U}(N)$, known as the orthonormal frame aligned with $\vec{d}_j$, satisfying:

* $R(\vec{d}_j) e_1 = \vec{d}_j$
* $R(\vec{d}_j)^\dagger R(\vec{d}_j) = I_N$

A canonical choice of $R(\vec{d}_j)$ is given via the exponential of an anti-Hermitian generator using a generalized Rodrigues construction. Let:

$u := e_1, \quad v := \vec{d}_j, \quad w := v - \langle v, u \rangle u$

If $\|w\| > 0$, define:

$\hat{w} := \frac{w}{\|w\|}, \quad A := \hat{w} u^\dagger - u \hat{w}^\dagger \in \mathfrak{u}(N)$

Let $\theta := \arccos(\Re \langle v, u \rangle)$. Then:

$R(\vec{d}_j) := e^{\theta A} \in \mathrm{U}(N)$

This yields a minimal rotation in the complex plane mapping $e_1$ to $\vec{d}_j$, with the remaining columns forming an orthonormal completion. The case $\vec{d}_j = e^{i\phi} e_1$ reduces to identity rotation. See *CPSF: Orthonormal Frame Construction* for further discussion.

We then construct the extended orthonormal frame:

$\mathcal{R}(\vec{d}_j) := \mathrm{diag}(R(\vec{d}_j), R(\vec{d}_j)) \in \mathrm{U}(2N)$

and define the diagonal attenuation matrix:

$D_j := \mathrm{diag}(\sigma_j^{\parallel}, \underbrace{\sigma_j^{\perp}, \dotsc, \sigma_j^{\perp}}_{N-1}, \sigma_j^{\parallel}, \underbrace{\sigma_j^{\perp}, \dotsc, \sigma_j^{\perp}}_{N-1}) \in \mathbb{R}^{2N \times 2N}$

Assume that both attenuation scalars $\sigma_j^{\parallel}, \sigma_j^{\perp} \in \mathbb{R}_{>0}$ are strictly positive real numbers. Consequently, the diagonal matrix $D_j$ is positive definite, and the similarity transformation $\Sigma_j = \mathcal{R}(\vec{d}_j)^\dagger D_j \mathcal{R}(\vec{d}_j)$ implies that $\Sigma_j$ is Hermitian and strictly positive definite. Therefore, $\Sigma_j^{-1}$ exists and is also Hermitian positive definite.

Then the covariance matrix $\Sigma_j$ is defined by:

$\Sigma_j := \mathcal{R}(\vec{d}_j)^\dagger \cdot D_j \cdot \mathcal{R}(\vec{d}_j)$

This definition ensures that the Gaussian envelope is anisotropic and aligned with the ray direction $\vec{d}_j$ in both spatial and directional subspaces.

The unnormalized Gaussian envelope centered at $\ell_j$ is:

$\rho_j(w) := \exp\left( -\pi \left\langle \Sigma_j^{-1} w, w \right\rangle \right)$

where the Hermitian inner product is:

$\langle u, v \rangle := \sum_{k=1}^{2N} \overline{u_k} v_k$

To restore toroidal periodicity, define the periodized envelope via lattice summation:

$\psi_j^{\mathbb{T}}(z, \vec{d}) := \sum_{n \in \Lambda} \rho_j\left( \iota(\tilde{z} - \tilde{z}_j + n, \vec{d} - \vec{d}_j) \right)$

Although the lifted coordinates $\tilde{z}, \tilde{z}_j \in \mathbb{C}^N$ are not uniquely defined modulo $\Lambda$, the periodized envelope $\psi_j^{\mathbb{T}}(z, \vec{d})$ is invariant under these choices. This follows from the fact that the summation over $n \in \Lambda$ effectively integrates over all toroidal shifts, rendering the final value independent of the specific representatives $\tilde{z}, \tilde{z}_j$.

This function is smooth, $\Lambda$-periodic in $z$, and rapidly decaying in $\vec{d}$, assuming $\Re(\Sigma_j^{-1}) > 0$.

Let the complex vector space of field values be $\mathbb{C}^S$. Define the global field response:

$T : \mathbb{T}_\mathbb{C}^N \times \mathbb{C}^N \rightarrow \mathbb{C}^S, \quad T(z, \vec{d}) := \sum_{j \in \mathcal{J}} \alpha_j \cdot \psi_j^{\mathbb{T}}(z, \vec{d}) \cdot \hat{T}_j$

where $\mathcal{J}$ is the index set of contributions and $\hat{T}_j \in \mathbb{C}^S$ is the spectral content vector of $C_j$.

Let $T^{\text{ref}}(z, \vec{d})$ denote the expected projection response. Then the discrepancy is:

$\Delta T(z, \vec{d}) := T^{\text{ref}}(z, \vec{d}) - T(z, \vec{d})$

Assume the Hilbert space $L^2(\mathbb{T}_\mathbb{C}^N \times \mathbb{C}^N; \mathbb{C}^S)$ with inner product:

$\langle f, g \rangle := \int_{\mathbb{T}_\mathbb{C}^N \times \mathbb{C}^N} \sum_{s=1}^S \overline{f_s(z, \vec{d})} \cdot g_s(z, \vec{d}) \, d\mu(z) \, d\nu(\vec{d})$

where $d\mu(z)$ is the normalized Haar measure and $d\nu(\vec{d})$ is Lebesgue measure.

Then the orthogonal projection of $\Delta T$ onto the mode $\psi_j^{\mathbb{T}}$ yields:

$\Delta \hat{T}_j = \frac{1}{\alpha_j} \cdot \frac{ \int \overline{\psi_j^{\mathbb{T}}(z, \vec{d})} \cdot \Delta T(z, \vec{d}) \, d\mu(z) \, d\nu(\vec{d}) }{ \int |\psi_j^{\mathbb{T}}(z, \vec{d})|^2 \, d\mu(z) \, d\nu(\vec{d}) }$

This expression defines an analytic update of $\hat{T}_j$, minimizing the squared $L^2$-error under the localization induced by $\Sigma_j$.

Thus, $\Sigma_j$ not only defines spatial and directional extent of $C_j$, but also enables precise localization of semantic error $\Delta T$ onto its mode $\hat{T}_j$.
