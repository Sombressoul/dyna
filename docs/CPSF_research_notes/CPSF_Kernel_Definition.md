## CPSF: Definition and Structure of the Projection Kernel

Let $ D, N \in \mathbb{N} $, $ \mathbb{T}^{2D} := (\mathbb{C} / \mathbb{Z})^{2D} $, $ \mathbb{Z}^N $ be a discrete spectral lattice.

---

## Projection Kernel $ \psi_j^{\mathbb{T}}(\ell, k; \alpha_j) $

Let:
- $ \ell \in \mathbb{T}^{2D} \subset \mathbb{C}^{2D} $, $ \ell_j := (\vec{o}_j, \vec{d}_j) \in \mathbb{T}^{2D} $, $ \|\vec{d}_j\| = 1 $
- $ R_j \,{\in}\, \mathrm{U}(D),\ R_j[:,1] := \vec{d}_j $
- $ \mathcal{R}_j := \mathrm{diag}(R_j, R_j) \in \mathrm{U}(2D) $
- $ \sigma_j^{\parallel}, \sigma_j^{\perp} \in \mathbb{R}_{> 0} $
- $ \Sigma_j := \Sigma_j(\alpha_j) := \mathcal{R}_j^{\dagger} \cdot \mathrm{diag}(\sigma_j^{\parallel}(\alpha_j), \sigma_j^{\perp}(\alpha_j) I_{D-1}, \sigma_j^{\parallel}(\alpha_j), \sigma_j^{\perp}(\alpha_j) I_{D-1}) \cdot \mathcal{R}_j \in \mathbb{C}^{2D \times 2D} $
- $ k \in \mathbb{Z}^N,\ k_j \in \mathbb{R}^N,\ \Gamma_j := \Gamma_j(\alpha_j) \in \mathbb{S}_{++}^N $

Define:

$$
\psi_{j,\text{geo}}^{\mathbb{T}}(\ell; \Sigma_j(\alpha_j)) := 
\sum_{n \in \mathbb{Z}^{2D}} \exp\left( 
- \tfrac{1}{2} (\ell - \ell_j + n)^\dagger \Sigma_j(\alpha_j)^{-1} (\ell - \ell_j + n) 
\right)
$$
$$
\psi_{j,\text{spec}}^{\mathbb{T}}(k; \Gamma_j(\alpha_j)) := 
\exp\left( - \tfrac{1}{2} (k - k_j)^\top \Gamma_j(\alpha_j)^{-1} (k - k_j) \right)
$$


---

## Properties

1. $ \psi_j^{\mathbb{T}} \in \mathcal{C}^{\infty}(\mathbb{T}^{2D} \times \mathbb{Z}^N; \mathbb{C}) $
2. $ \forall m \in \mathbb{Z}^{2D},\ \psi_j^{\mathbb{T}}(\ell + m, k) = \psi_j^{\mathbb{T}}(\ell, k) $
3. $ \exists C, \lambda > 0:\ \psi_{j,\text{geo}}^{\mathbb{T}}(\ell) \le C \cdot \exp(-\lambda \cdot \mathrm{dist}_{\mathbb{T}}^2(\ell, \ell_j)) $
4. $ \forall \theta \in \{ \ell_j, \vec{d}_j, \sigma_j^{\parallel}, \sigma_j^{\perp}, k_j, \Gamma_j, \alpha_j \},\ \frac{\partial \psi_j^{\mathbb{T}}}{\partial \theta} \in \mathbb{C} $ exists and is Wirtinger-holomorphic
5. $ \psi_j^{\mathbb{T}} $ defines a doubly spectral localization kernel over $ \ell \in \mathbb{T}^{2D} $, $ k \in \mathbb{Z}^N $

---

## Requirements

- $ \psi_j^{\mathbb{T}} : \mathbb{T}^{2D} \times \mathbb{Z}^N \to \mathbb{C} $ must be:
  1. Periodic in $ \ell $, toroidally localized
  2. Smooth in all continuous parameters
  3. Exponentially decaying in $ \ell $, localized in $ k $
  4. Holomorphic in $ \theta \in \{ \ell_j, \vec{d}_j, \Sigma_j, k_j, \Gamma_j, \alpha_j \} $
  5. Closed-form, non-sampled, non-parametric
  6. Compatible with spectral projection over $ \phi_k(x) = e^{2\pi i \langle k,x \rangle} $

---

## Use

- Enters the projected spectral response:
  $$
  \psi_k(\ell) = \sum_j \alpha_j T_j \cdot \psi_j^{\mathbb{T}}(\ell, k; \alpha_j) \cdot \hat{h}_{j,k}
  $$
- Enables dual localization in geometry and frequency for reversible projection:
  $$
  T(\ell) = \sum_k \hat{w}_k \cdot \psi_k(\ell)
  $$
