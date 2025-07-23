## Directional Decomposition in Aligned Local Frames

This section introduces a refined formulation of directional deviation within CPSF by expressing projection directions relative to a locally aligned unitary basis. This replaces the naive linear difference $\vec{d} - \vec{d}_j$ with a geometrically consistent decomposition in the orthonormal frame $R(\vec{d}_j)$ already present in CPSF.

---

### Motivation and Context

In the standard CPSF formulation, the relative offset vector
$$
w := \iota(\tilde{z} - \tilde{z}_j,\ \vec{d} - \vec{d}_j) \in \mathbb{C}^{2N}
$$
is used as the argument of the Gaussian envelope
$$
\rho_j(w) := \exp\left( -\pi \langle \Sigma_j^{-1} w, w \rangle \right).
$$

While this construction is valid when $\vec{d}$ and $\vec{d}_j$ are sufficiently close, the term $\vec{d} - \vec{d}_j$ does not remain on the unit sphere and becomes geometrically misleading under large deviations. This can lead to inconsistencies with the unitarily invariant measure $d\sigma(\vec{d})$ and incorrect localization behavior.

To resolve this, we introduce a new formulation based on expressing $\vec{d}$ in the orthonormal basis defined by $R(\vec{d}_j)$.

---

### Frame-Based Directional Decomposition

Let $R(\vec{d}_j) \in \mathrm{U}(N)$ be the unitary matrix such that
$$
R(\vec{d}_j) e_1 = \vec{d}_j,
$$
and its remaining columns span $\vec{d}_j^\perp$ orthonormally. Then any unit vector $\vec{d} \in \mathbb{S}^{2N-1}$ may be expressed in this frame as:
$$
\vec{d} = R(\vec{d}_j) \cdot \begin{bmatrix} \alpha \\ \vec{\xi} \end{bmatrix}, \quad \alpha \in \mathbb{C},\ \vec{\xi} \in \mathbb{C}^{N-1},\ \text{with } |\alpha|^2 + \|\vec{\xi}\|^2 = 1.
$$

Define the aligned lifted spatial offset as:
$$
\vec{u} := R(\vec{d}_j)^\dagger (\tilde{z} - \tilde{z}_j) \in \mathbb{C}^N.
$$
Then the total argument vector becomes:
$$
w' := \begin{bmatrix} \vec{u} \\ \begin{bmatrix} \alpha - 1 \\ \vec{\xi} \end{bmatrix} \end{bmatrix} \in \mathbb{C}^{2N}.
$$
This expresses the deviation entirely in the local frame aligned with $\vec{d}_j$.

---

### Reformulated Gaussian Envelope

Under this change of variables, the unnormalized Gaussian envelope becomes:
$$
\rho_j(w) = \exp\left( -\pi \langle D_j w', w' \rangle \right),
$$
where $D_j$ is the diagonal attenuation matrix defined as:
$$
D_j := \mathrm{diag}(\sigma_j^\parallel, \underbrace{\sigma_j^\perp, \dotsc, \sigma_j^\perp}_{N-1},\ \sigma_j^\parallel, \underbrace{\sigma_j^\perp, \dotsc, \sigma_j^\perp}_{N-1}) \in \mathbb{R}^{2N \times 2N}.
$$

This formulation ensures that the geometry of the Gaussian is fully compatible with the sphere's tangent structure at $\vec{d}_j$.

---

### Periodization and Field Construction

The periodized envelope is defined analogously:
$$
\psi_j^{\mathbb{T}}(z, \vec{d}) := \sum_{n \in \Lambda} \rho_j\left( \begin{bmatrix} \vec{u} + R(\vec{d}_j)^\dagger n \\ \begin{bmatrix} \alpha - 1 \\ \vec{\xi} \end{bmatrix} \end{bmatrix} \right).
$$

All subsequent constructions — the global field response $T(z, \vec{d})$, the semantic error $\Delta T(z, \vec{d})$, and the projection update $\Delta \hat{T}_j$ — remain **unchanged in form**, but are now grounded in a strictly geometrically consistent local representation of direction.

---

### Justification and Consistency

This modification preserves all functional aspects of CPSF:

* The decomposition is made in a unitary frame already used to construct $\Sigma_j$;
* No change to measures or domains is required — only the internal expression of $\vec{d}$ is reparameterized;
* All existing definitions of $T$, $\psi_j^{\mathbb{T}}$, and $\Delta \hat{T}_j$ remain valid and consistent;
* The construction becomes fully compatible with the unitarily invariant measure $d\sigma(\vec{d})$.

This reformulation should be interpreted not as a structural modification of CPSF, but as a **geometric refinement** of how directional deviations are computed within its existing framework.

