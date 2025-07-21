DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT DRAFT

**Hierarchical Structure of CPSF (Continuous Projective Semantic Fields)**

---

## 1. Coordinate and Spectral Basis

### 1.1. Coordinate Space

- $x \in \mathbb{C}^N$: global complex coordinate space
- $\mathbb{T}^N$: toroidal topology $(\mathbb{C} / \mathbb{Z})^N$
- All projection geometry is defined in $\mathbb{C}^D$ with toroidal wrap-around

### 1.2. Spectral Basis

- $\phi_k(x) = e^{2\pi i \langle k, x \rangle}$: complex toroidal harmonics
- $\hat{w}_k \in \mathbb{C}$: spectral coefficients
- $k \in \mathbb{Z}^N$: discrete frequency lattice

---

## 2. Projection Geometry (Ray / View)

- $\vec{o} \in \mathbb{T}^D$, $\vec{d} \in \mathbb{C}^D$, $\|\vec{d}\| = 1$
- Full ray vector $\ell$ is interpreted modulo 1 coordinate-wise

---

## 3. Projection Interface

### 3.1. Projection Kernel

Let $\ell = (\vec{o}, \vec{d}) \in \mathbb{T}^{2D} \subset \mathbb{C}^{2D}$, where $\vec{o} \in \mathbb{T}^D$, $\vec{d} \in \mathbb{C}^D$, $\|\vec{d}\| = 1$.

Let $R_j \in \mathrm{U}(D)$ such that:
$R_j[:,1] = \vec{d}_j, \quad R_j^\dagger R_j = I_D$

Define:
$\mathcal{R}_j := \mathrm{diag}(R_j, R_j) \in \mathrm{U}(2D)$

Let $\sigma_j^{\parallel}, \sigma_j^{\perp} \in \mathbb{R}_{>0}$.

Define:

$$
\Sigma_j := \mathcal{R}_j^\dagger
\begin{bmatrix}
\sigma_j^{\parallel} & 0 & 0 & 0 \\
0 & \sigma_j^{\perp} I_{D-1} & 0 & 0 \\
0 & 0 & \sigma_j^{\parallel} & 0 \\
0 & 0 & 0 & \sigma_j^{\perp} I_{D-1}
\end{bmatrix}
\mathcal{R}_j \in \mathbb{C}^{2D \times 2D}
$$

Let $\ell_j = (\vec{o}_j, \vec{d}_j) \in \mathbb{T}^{2D}$.

Define:

$$
\psi_j^{\mathbb{T}}(\ell) := \sum_{n \in \mathbb{Z}^{2D}} \exp\left[
  - (\ell - \ell_j + n)^\dagger \Sigma_j^{-1} (\ell - \ell_j + n)
\right]
$$

Properties:

- $\forall m \in \mathbb{Z}^{2D} \Rightarrow \psi_j^{\mathbb{T}}(\ell + m) = \psi_j^{\mathbb{T}}(\ell)$

- $\psi_j^{\mathbb{T}} \in \mathcal{C}^\infty(\mathbb{T}^{2D})$

- Define the parameter set:
   $\Theta_j := \{ \ell_j, \vec{d}_j, \sigma_j^{\parallel}, \sigma_j^{\perp} \}$
   Then:
   $\forall \theta \in \Theta_j, \quad \frac{\partial \psi_j^{\mathbb{T}}}{\partial \theta} \text{ exists and is holomorphic under Wirtinger calculus}$

- $\exists \lambda \in \mathbb{R}_{>0}, C \in \mathbb{R}_{>0}$ such that:

$$
\forall \ell \in \mathbb{T}^{2D}, \quad \psi_j^{\mathbb{T}}(\ell) \le C \cdot \exp(-\lambda \cdot \mathrm{dist}_{\mathbb{T}}^2(\ell, \ell_j))
$$

The function $\psi_j^{\mathbb{T}} \colon \mathbb{T}^{2D} \to \mathbb{R}_{>0}$ defines a smooth, toroidally periodic, anisotropic Gaussian sum with unitary-invariant covariance structure $\Sigma_j$, compatible with spectral projection operators $\psi_k(\ell) := \int_{\mathbb{T}^N} \phi_k(x) K(x,\ell) dx$ under appropriate choice of projection map $\Pi : \mathbb{T}^N \to \mathbb{T}^{2D}$.


### 3.2. Projection Operator

Let the memory field be defined spectrally:

$$
W(x) = \sum_{k \in \mathbb{Z}^N} \hat{w}_k \cdot \phi_k(x), \quad \phi_k(x) := e^{2\pi i \langle k, x \rangle}, \quad x \in \mathbb{T}^N \subset \mathbb{C}^N
$$

Each memory contribution is parametrized as:

$$
C_j := (\ell_j, x_j, T_j, \Sigma_j, \Lambda_j, \alpha_j),
\quad \ell_j \in \mathbb{T}^{2D}, \quad x_j \in \mathbb{T}^N,
\quad T_j \in \mathbb{C}^C,
\quad \Sigma_j \in \mathbb{S}_{++}^{2D},
\quad \Lambda_j \in \mathbb{S}_{++}^{N},
\quad \alpha_j \in [0,1]
$$

#### 3.2.1. Composite Kernel

Define the spatial projection envelope:

$$
\psi_j^{\mathbb{T}}(\ell) := \sum_{n \in \mathbb{Z}^{2D}} \exp\left[ - (\ell - \ell_j + n)^\dagger \Sigma_j^{-1} (\ell - \ell_j + n) \right]
$$

Define the spectral semantic envelope:

$$
h_j(x) := \sum_{m \in \mathbb{Z}^N} \exp\left[ - (x - x_j + m)^\top \Lambda_j^{-1} (x - x_j + m) \right]
$$

Then the joint projection kernel is:

$$
K_j(x, \ell) := \psi_j^{\mathbb{T}}(\ell) \cdot h_j(x)
$$

#### 3.2.2. Projected Response

Define the projection operator:

$$
\psi_k(\ell) := \int_{\mathbb{T}^N} \phi_k(x) \cdot K(x, \ell) dx, \quad K(x, \ell) := \sum_j \alpha_j T_j \cdot K_j(x, \ell)
$$

with spectral coefficient:

$$
\hat{h}_{j,k} := \int_{\mathbb{T}^N} \phi_k(x) h_j(x) dx
$$

Then the projected response is:

$$
T(\ell) = \sum_k \hat{w}_k \cdot \psi_k(\ell) = \sum_j \alpha_j T_j \cdot \psi_j^{\mathbb{T}}(\ell) \cdot \int_{\mathbb{T}^N} h_j(x)^2 dx
$$

#### 3.2.3. Spectral Update

Given target $T^* \in \mathbb{C}^C$ at location $\ell^* \in \mathbb{T}^{2D}$, define:

$$
\psi_k(\ell^*) := \sum_j \alpha_j T_j \cdot \psi_j^{\mathbb{T}}(\ell^*) \cdot \hat{h}_{j,k}
$$

Then solve for correction $\Delta \hat{w}_k$ from:

$$
\sum_k (\hat{w}_k + \Delta \hat{w}_k) \cdot \psi_k(\ell^*) = T^* \quad \Rightarrow \quad \Delta \hat{w} := \operatorname{argmin}_w \left\| \sum_k w_k \cdot \psi_k(\ell^*) - (T^* - T(\ell^*)) \right\|^2
$$

#### 3.2.4. Analytic Properties

* $T(\ell) \in \mathcal{C}^\infty(\mathbb{T}^{2D})$
* $\forall \theta \in \{ \hat{w}_k, \ell_j, x_j, \Sigma_j, \Lambda_j, \alpha_j \}, \; \frac{\partial T}{\partial \theta} \in \mathbb{C}$ exists and is Wirtinger-holomorphic
* $\{ \psi_k(\ell) \}_k$ forms a linearly independent system in $L^2(\mathbb{T}^{2D})$
* The operator $\mathcal{P}_\ell[W]$ is smooth, toroidally periodic, and spectrally invertible

---

## 4. Memory Contribution

Let each contribution be defined as the 6-tuple:

$$
C_j := (\ell_j, x_j, T_j, \Sigma_j, \Lambda_j, \alpha_j)
\in \mathbb{T}^{2D} \times \mathbb{T}^N \times \mathbb{C}^C \times \mathbb{S}_{++}^{2D} \times \mathbb{S}_{++}^N \times [0,1]
$$

with:

* $\ell_j \in \mathbb{T}^{2D} \subset \mathbb{C}^{2D}$: projection coordinate (origin and direction),
* $x_j \in \mathbb{T}^N \subset \mathbb{C}^N$: semantic center in field space,
* $T_j \in \mathbb{C}^C$: content vector,
* $\Sigma_j \in \mathbb{S}_{++}^{2D}$: anisotropic projection covariance,
* $\Lambda_j \in \mathbb{S}_{++}^N$: semantic envelope covariance,
* $\alpha_j \in [0,1]$: contribution weight.

Let the memory system be the finite collection:

$$
\mathcal{M} := \{ C_j \}_{j=1}^N
$$

Each $C_j \in \mathcal{M}$ induces a localized projection kernel:

$$
K_j(x, \ell) := \psi_j^{\mathbb{T}}(\ell) \cdot h_j(x)
\quad \text{with} \quad
\psi_j^{\mathbb{T}}(\ell),\; h_j(x) \text{ defined in Section 3.2.1}
$$

These contributions define the entire memory field $W(x)$, its spectral coefficients $\hat{w}_k$, and its projection behavior $T(\ell)$ via Section 3.2.

---

## 5. Memory Field

### 5.1. Field Representations

### 5.1.1. Spectral Representation

Let the memory field be represented as a superposition of localized semantic contributions:

$$
W(x) := \sum_{j=1}^N \alpha_j \cdot T_j \cdot h_j(x), \quad x \in \mathbb{T}^N \subset \mathbb{C}^N
$$

where:

* $\alpha_j \in [0,1]$ — scalar contribution weight,
* $T_j \in \mathbb{C}^C$ — content vector,
* $h_j(x) \colon \mathbb{T}^N \to \mathbb{R}_{>0}$ — smooth, toroidally periodic envelope defined as:

$$
h_j(x) := \sum_{m \in \mathbb{Z}^N} \exp\left[ - (x - x_j + m)^\top \Lambda_j^{-1} (x - x_j + m) \right]
$$

with parameters:

* $x_j \in \mathbb{T}^N \subset \mathbb{C}^N$ — semantic center of contribution,
* $\Lambda_j \in \mathbb{S}_{++}^N$ — positive-definite covariance matrix.

The spectral coefficients $\hat{w}_k \in \mathbb{C}^C$ are defined by:

$$
\hat{w}_k := \int_{\mathbb{T}^N} W(x) \cdot \overline{\phi_k(x)} \, dx = \sum_{j=1}^N \alpha_j T_j \cdot \hat{h}_{j,k}
\quad \text{where} \quad
\phi_k(x) := e^{2\pi i \langle k, x \rangle}
$$

and:

$$
\hat{h}_{j,k} := \int_{\mathbb{T}^N} \phi_k(x) \cdot h_j(x) dx
$$

**Properties:**

* $W(x) \in \mathcal{C}^\infty(\mathbb{T}^N; \mathbb{C}^C)$
* The field is fully defined by the set $\{ C_j \}$
* Spectral expansion $W(x) = \sum_k \hat{w}_k \phi_k(x)$ holds with:

  $$
  \hat{w}_k = \sum_j \alpha_j T_j \cdot \hat{h}_{j,k}
  $$

This representation makes the field $W(x)$ explicitly interpretable as a smooth sum of localized memory contributions.

### 5.1.2. Geometric Representation

Let the projection response be:

$$
T(\ell) := \sum_j \alpha_j \cdot T_j \cdot \psi_j^{\mathbb{T}}(\ell)
$$

where:

- $\psi_j^{\mathbb{T}}(\ell)$ is defined via extended anisotropic kernel
- $\alpha_j$ is the scalar contribution relevance weight

### 5.1.3. Differentiability

Let $T(\ell) := \sum_j \alpha_j T_j \psi_j^{\mathbb{T}}(\ell)$, where each $\psi_j^{\mathbb{T}}(\ell)$ is defined by the series

$$
\psi_j^{\mathbb{T}}(\ell) := \sum_{n \in \mathbb{Z}^{2D}} \exp\left( - (\ell - \ell_j + n)^\dagger \Sigma_j^{-1} (\ell - \ell_j + n) \right),
$$

which converges absolutely and uniformly on compact subsets of $\mathbb{T}^{2D}$, and is infinitely differentiable with respect to the parameters of $C_j$, with

$$
\begin{aligned}
&\vec{o}_j \in \mathbb{T}^D \subset \mathbb{C}^D, \quad \vec{d}_j \in \mathbb{C}^D, \quad T_j \in \mathbb{C}^C, \\
&\sigma_j^{\parallel}, \sigma_j^{\perp} \in \mathbb{R}_{> 0}, \quad \alpha_j \in [0,1]
\end{aligned}
$$

Then:

$$
\frac{\partial T}{\partial \bar{z}} = 0 \quad \forall z \in \{ \vec{o}_j, \vec{d}_j, T_j \}
$$

and

$$
\frac{\partial T}{\partial \theta} \in \mathbb{C}, \quad \forall \theta \in \{ \sigma_j^{\parallel}, \sigma_j^{\perp}, \alpha_j \}
$$

Therefore, $T(\ell) \in \mathcal{C}^\infty$ with respect to $\ell$ and all parameters of $C_j$, with complex parameters treated via Wirtinger calculus.

All field-level operations preserve differentiability.

---

## 6. Core Operations

### 6.1. Field Operations

#### 6.1.1. Basic CRUD

**CREATE**

$$
\textbf{CREATE:} \quad \mathcal{M} \leftarrow \mathcal{M} \cup \{ C_j \} \\
$$

**READ**  

Given memory field $W(x)$, the projected response at direction $\ell \in \mathbb{T}^{2D}$ is defined by:

$$
T(\ell) := \sum_k \hat{w}_k \cdot \psi_k(\ell)
\quad \text{where} \quad
\psi_k(\ell) := \int_{\mathbb{T}^N} \phi_k(x) \cdot K(x, \ell)\, dx
$$

This defines a linear projection operator $\mathcal{P}_\ell[W] = T(\ell)$, smooth and spectrally invertible.

**UPDATE**

Given a target $T^* \in \mathbb{C}^C$ at direction $\ell^*$, compute spectral correction:

$$
\Delta \hat{w} := \operatorname{argmin}_w \left\| \sum_k w_k \cdot \psi_k(\ell^*) - (T^* - T(\ell^*)) \right\|^2
$$

This defines a smooth inverse update on the field $W(x)$ via its spectral coefficients $\hat{w}_k$.

**DELETE**
$$
\textbf{DELETE:} \quad \alpha_j \leftarrow 0 \quad \text{or} \quad \sigma_j^{\parallel}, \sigma_j^{\perp} \to \infty
$$

DELETE uses parameter blow-up as an asymptotic procedure; not included in differentiable range


#### 6.1.2. Find

Assume: 
$$
\quad \ell_j \in \mathbb{T}^{2D} \text{ for all } j
$$

Let:

$$
\begin{aligned}
&C_j := (\vec{o}_j, \vec{d}_j, T_j, \sigma_j^{\parallel}, \sigma_j^{\perp}, \alpha_j), \quad j = 1, \dots, N \\
&\ell_j := (\vec{o}_j, \vec{d}_j) \in \mathbb{T}^{2D} \subset \mathbb{C}^{2D} \\
&T_j \in \mathbb{C}^C, \quad \alpha_j \in [0, 1] \\
&T^* \in \mathbb{C}^C \quad \text{(target vector)} \\
&\tau \in \mathbb{R}_{>0} \quad \text{(semantic temperature)}
\end{aligned}
$$

Define:

$$
\| T_j - T^* \|^2 := \sum_{c=1}^C \left| (T_j)_c - (T^*)_c \right|^2
$$

Then the inverse query solution is given by:

$$
\ell^* := \operatorname{mod}_1 \left( \sum_{j=1}^N \ell_j \cdot \frac{ \alpha_j \cdot \exp\left( - \frac{ \| T_j - T^* \|^2 }{2\tau^2} \right) }{ \sum_k \alpha_k \cdot \exp\left( - \frac{ \| T_k - T^* \|^2 }{2\tau^2} \right) } \right)
$$


This expression is closed-form and differentiable with respect to all parameters.

#### 6.1.3. FindParametric

Let $T^* \in \mathbb{C}^C$ be a query vector and $\tau \in \mathbb{R}_{>0}$ a fixed temperature parameter.

Let $\{C_j\}_{j=1}^N$, where
$$
C_j := (\ell_j, T_j, \Sigma_j, \alpha_j), \quad \ell_j := (\vec{o}_j, \vec{d}_j) \in \mathbb{T}^{2D}, \quad T_j \in \mathbb{C}^C, \quad \Sigma_j \in \mathbb{S}_{++}^{2D}, \quad \alpha_j \in [0,1].
$$

Define the similarity kernel
$$
K_j(T^*) := \exp\left( - \frac{ \| T_j - T^* \|^2 }{2\tau^2} \right),
\quad \text{where } \| \cdot \|^2 := \langle \cdot, \cdot \rangle_{\mathbb{C}^C}.
$$

Then define
$$
w_j := \frac{ \alpha_j \cdot K_j(T^*) }{ \sum\limits_{k=1}^N \alpha_k \cdot K_k(T^*) },
\quad \text{with } w_j \in [0,1], \quad \sum_j w_j = 1.
$$

Let the geometric target be
$$
\ell^* := \operatorname{mod}_1 \left( \sum_{j=1}^N w_j \cdot \ell_j \right) \in \mathbb{T}^{2D},
$$

and the kernel estimate
$$
\Sigma^* := \sum_{j=1}^N w_j \cdot \Sigma_j \in \mathbb{S}_{++}^{2D}.
$$

Equivalently, if each $\Sigma_j$ is generated by
$$
\Sigma_j := \mathcal{R}_j^\dagger \cdot 
\mathrm{diag}(\sigma_j^\parallel, \sigma_j^\perp I_{D-1}, \sigma_j^\parallel, \sigma_j^\perp I_{D-1}) \cdot 
\mathcal{R}_j,
$$
define
$$
\sigma^{\parallel *} := \sum_j w_j \cdot \sigma_j^{\parallel}, \qquad
\sigma^{\perp *} := \sum_j w_j \cdot \sigma_j^{\perp},
$$
and reconstruct
$$
\Sigma^* := \mathcal{R}^{\dagger} \cdot 
\mathrm{diag}(\sigma^{\parallel *}, \sigma^{\perp *} I_{D-1}, \sigma^{\parallel *}, \sigma^{\perp *} I_{D-1}) \cdot 
\mathcal{R}.
$$

The output is
$$
\operatorname{FindParametric}(T^*) := (\ell^*, \Sigma^*) \in \mathbb{T}^{2D} \times \mathbb{S}_{++}^{2D}.
$$

> **Note**: in the decomposed reconstruction of $\Sigma^*$ via $(\sigma^{\parallel *}, \sigma^{\perp *})$, the rotation matrix $\mathcal{R} \in \mathbb{U}(2D)$ may be chosen as:
> 
> - either a fixed canonical rotation (e.g., identity matrix),
> - or a weighted average of $\{\mathcal{R}_j\}$ with respect to weights $w_j$,
> 
> provided that the resulting $\Sigma^* \in \mathbb{S}_{++}^{2D}$ remains positive-definite.
> 
> In practice, $\mathcal{R} := I$ is sufficient in isotropic cases.

This operation is analytic and differentiable with respect to all parameters $\{ T_j, \alpha_j, \Sigma_j, \ell_j \}$ via Wirtinger calculus.

---

## 7. Field Dynamics

$\frac{\partial^2 W}{\partial t^2} = c^2 \nabla^2 W + S(x, t) \quad \text{or} \quad i\hbar \frac{\partial \psi}{\partial t} = \hat{H} \psi$

- $S(x,t)$: source term

---

## 8. Complex Gradients

All derivatives via Wirtinger calculus

---

## 9. Address–Value Separation

### 9.1. Addressing

- $x_{\text{addr}} \in \mathbb{T}^A \subset \mathbb{C}^A$, $x_{\text{value}} \in \mathbb{C}^V$

### 9.2. Toroidally Compatible Semantic–Address Map

Let:

- $W \in \mathbb{C}^{V \times A}$, full rank
- $W^+$: Moore–Penrose pseudoinverse

Define: $\phi(x) := (e^{2\pi i x_1}, \dots, e^{2\pi i x_A})$ $\phi^{-1}(z) := \left( \frac{1}{2\pi i} \log z_1, \dots, \frac{1}{2\pi i} \log z_A \right) \mod 1$ $f(x_{\text{addr}}) := W \phi(x_{\text{addr}}), \quad f^{-1}(x_{\text{value}}) := \phi^{-1}(W^+ x_{\text{value}})$ with $\log$ understood as principal branch and $z \in (\mathbb{C}^*)^A$


Properties:

- $f$ holomorphic, $f^{-1}(f(x)) = x \mod 1$
- Compatible with $\phi_k(x) = e^{2\pi i \langle k, x \rangle}$

---

## Summary Tree

```plaintext
CPSM
├── Coordinates: ℂ^N with torus topology
├── Spectral Basis: φ_k = exp(2πi⟨k,x⟩), k ∈ ℤ^N
├── Rays: ℓ = (o, d) ∈ ℂ^{2D} mod 1
├── Kernel: Toroidal Gaussian ψ_j(ℓ)
├── Contributions: (o_j, d_j, T_j, σ_j^∥, σ_j^⊥, α_j)
├── Field: spectral W(x), geometric T(ℓ)
├── Ops: Read, Create, Update, Delete, Find, FindParametric
├── Dynamics: wave / Schrödinger
└── Address–Value:
    ├── x_addr ∈ ℂ^A mod 1
    ├── x_value ∈ ℂ^V
    └── x_value = W · exp(2πix_addr), invertible via log
```

