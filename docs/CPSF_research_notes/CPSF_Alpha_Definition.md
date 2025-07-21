## CPSF: Definition and Structure of the Coefficient $ \alpha_j $

Let $ D, N, C \in \mathbb{N} $, $ \mathbb{T}^{2D} := (\mathbb{C} / \mathbb{Z})^{2D} $, $ \mathbb{T}^N \subset \mathbb{C}^N $, and $ \mathbb{Z}^N $ be the discrete frequency lattice. Let $ \phi_k(x) := e^{2\pi i \langle k, x \rangle} $, $ k \in \mathbb{Z}^N $ be the global toroidal Fourier basis.

---

## Definition: Emergent Contribution Coefficient $ \alpha_j $

Let a memory contribution be defined as:
$$
C_j := (\ell_j, x_j, \hat{T}_j, \Lambda_j, k_j, \Gamma_j)
$$
where:
- $ \ell_j \in \mathbb{T}^{2D} \subset \mathbb{C}^{2D} $: projection coordinate,
- $ x_j \in \mathbb{T}^N \subset \mathbb{C}^N $: semantic center,
- $ \hat{T}_j \in \mathbb{C}^C $: semantic spectral vector,
- $ \Lambda_j \in \mathbb{S}_{++}^{N} $: semantic envelope covariance,
- $ k_j \in \mathbb{R}^N $: spectral center,
- $ \Gamma_j \in \mathbb{S}_{++}^N $: spectral covariance matrix.

Define the toroidal semantic kernel:
$$
h_j(x) := \sum_{m \in \mathbb{Z}^N} \exp\left( - (x - x_j + m)^\top \Lambda_j^{-1} (x - x_j + m) \right)
$$

Define the semantic field contribution:
$$
W_j(x) := h_j(x) \cdot \hat{T}_j \in \mathbb{C}^C
$$

Then the emergent contribution coefficient $ \alpha_j $ is defined as:
$$
\boxed{\alpha_j := \mathcal{E}_j := \|\hat{T}_j\|^2 \cdot \int_{\mathbb{T}^N} h_j(x)^2 dx}
$$

---

## Properties

1. $ \alpha_j \in \mathbb{R}_{\ge 0} $.
2. $ \alpha_j = \mathcal{A}(C_j) $ is a deterministic, differentiable functional of $ C_j $.
3. $ \alpha_j $ is an emergent, non-parametric energy measure.
4. $ \alpha_j $ scales the projection and spectral contributions:
   $$
   \hat{w}_k = \sum_j \alpha_j T_j \cdot \hat{h}_{j,k}, \quad
   T(\ell) = \sum_k \hat{w}_k \cdot \psi_k(\ell), \quad
   \psi_k(\ell) := \sum_j \alpha_j T_j \cdot \psi_j^{\mathbb{T}}(\ell, k; \alpha_j) \cdot \hat{h}_{j,k}
   $$
5. $ \alpha_j $ modifies the shape of the projection kernel:
   $$
   \Sigma_j := \Sigma_j(\alpha_j), \quad \Gamma_j := \Gamma_j(\alpha_j),
   \quad \psi_j^{\mathbb{T}}(\ell, k; \alpha_j) := \psi_{j,\mathrm{geo}}(\ell; \Sigma_j(\alpha_j)) \cdot \psi_{j,\mathrm{spec}}(k; \Gamma_j(\alpha_j))
   $$
6. $ \alpha_j \to 0 \Leftrightarrow \|\hat{T}_j\| \to 0 $ or $ \Lambda_j \to \infty $.
7. $ \alpha_j $ admits full Wirtinger-differentiability with respect to $ \hat{T}_j, \Lambda_j $; its influence on $ \psi_j^{\mathbb{T}} $ propagates via the chain rule.
8. $ \alpha_j $ determines effective projection mass and curvature; its effects are suppressed by orthogonality:
   $$
   \langle T_j, T_k \rangle \approx 0 \Rightarrow \text{interference suppression despite } \alpha_j \uparrow
   $$

---

## Role in Operations

### Projection (READ):
$$
T(\ell) = \sum_k \hat{w}_k \cdot \psi_k(\ell)
\quad \text{with} \quad
\psi_k(\ell) := \sum_j \alpha_j T_j \cdot \psi_j^{\mathbb{T}}(\ell, k; \alpha_j) \cdot \hat{h}_{j,k}
$$

### Spectrum (Fourier Expansion):
$$
\hat{w}_k = \sum_j \alpha_j T_j \cdot \hat{h}_{j,k}
$$

### Update (WRITE):
$$
\partial_{\hat{T}_j} \alpha_j \ne 0, \quad \text{thus } \alpha_j \text{ can be modified via } \hat{T}_j
$$

### Forget (DELETE):
$$
\alpha_j \downarrow \Leftrightarrow \|\hat{T}_j\| \to 0 \text{ or } \Lambda_j \to \infty
$$

### Find:
$$
\ell^* = \operatorname{mod}_1\left(\sum_j \ell_j \cdot \frac{\alpha_j \cdot \exp(-\|T_j - T^*\|^2 / 2\tau^2)}{\sum_k \alpha_k \cdot \exp(-\|T_k - T^*\|^2 / 2\tau^2)}\right)
$$

### Parametric Find:
$$
\Sigma^* := \sum_j w_j \cdot \Sigma_j(\alpha_j), \quad w_j := \frac{\alpha_j \cdot \exp(-\|T_j - T^*\|^2 / 2\tau^2)}{\sum_k \alpha_k \cdot \exp(-\|T_k - T^*\|^2 / 2\tau^2)}
$$

---

## Interpretation

- $ \alpha_j $ is the emergent scalar energy of contribution $ C_j $, reflecting total localized excitation.
- It induces curvature of the geometric and spectral projection kernels: $ \Sigma_j(\alpha_j), \Gamma_j(\alpha_j) $.
- It governs contribution strength, projection localization, and field modulation.
- It is semantically grounded, dynamically reversible, and fully differentiable.

---

## Summary

$$
\boxed{\alpha_j := \|\hat{T}_j\|^2 \cdot \int h_j(x)^2 dx \quad \text{(emergent semantic energy of } C_j)}
$$

$ \alpha_j $ is a semantically induced, geometrically active, differentiable operator of local field curvature and projection mass in CPSF.

