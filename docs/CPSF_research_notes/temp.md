## Space Topology

\(x \in \mathbb{C}^N\): global complex coordinate space

\(\mathbb{T}^N\): toroidal topology \((\mathbb{C} / \mathbb{Z})^N\)


## Projection Geometry

\(\ell = (\vec{o}, \vec{d}) \in \mathbb{T}^{2D} \subset \mathbb{C}^{2D}\), where \(\vec{o} \in \mathbb{T}^D\), \(\vec{d} \in \mathbb{C}^D\), \(\|\vec{d}\| = 1\).

Full ray vector \(\ell\) is interpreted modulo 1 coordinate-wise.

Let \(R_j \in \mathrm{U}(D)\) such that \(R_j[:,1] = \vec{d}_j\) and \(R_j^\dagger R_j = I_D\).

Define the unitary block-diagonal matrix:
\[\mathcal{R}_j := \mathrm{diag}(R_j, R_j) \in \mathrm{U}(2D)\]


## Projection Kernel

Define anisotropic Gaussian envelope:
\[\Sigma_j := \mathcal{R}_j^\dagger
\begin{bmatrix}
\sigma_j^{\parallel} & 0 & 0 & 0 \\
0 & \sigma_j^{\perp} I_{D-1} & 0 & 0 \\
0 & 0 & \sigma_j^{\parallel} & 0 \\
0 & 0 & 0 & \sigma_j^{\perp} I_{D-1}
\end{bmatrix}
\mathcal{R}_j \in \mathbb{C}^{2D \times 2D}
\]

Toroidal Gaussian kernel:
\[\psi_j^{\mathbb{T}}(\ell) := \sum_{n \in \mathbb{Z}^{2D}} \exp\left[
  - (\ell - \ell_j + n)^\dagger \Sigma_j^{-1} (\ell - \ell_j + n)
\right]\]


## Projection Operator

Define toroidal harmonic basis:
\[\phi_k(x) = e^{2\pi i \langle k, x \rangle}, \quad k \in \mathbb{Z}^N\]

Define spectral projection operator:
\[\psi_k(\ell) := \int_{\mathbb{T}^N} \phi_k(x) \cdot K(x, \ell) dx\]

where:
\[K(x, \ell) := \sum_j \alpha_j T_j \cdot K_j(x, \ell), \quad K_j(x, \ell) := \psi_j^{\mathbb{T}}(\ell) \cdot h_j(x)\]

Spatial envelope:
\[h_j(x) := \sum_{m \in \mathbb{Z}^N} \exp\left[ - (x - x_j + m)^\top \Lambda_j^{-1} (x - x_j + m) \right]\]


## Memory Contribution

Each contribution is a 6-tuple:
\[(\ell_j, x_j, T_j, \Sigma_j, \Lambda_j, \alpha_j)\]
with:
- \(\ell_j \in \mathbb{T}^{2D}\): origin and direction
- \(x_j \in \mathbb{T}^N\): semantic center
- \(T_j \in \mathbb{C}^C\): content
- \(\Sigma_j \in \mathbb{S}_{++}^{2D}\): directional covariance
- \(\Lambda_j \in \mathbb{S}_{++}^{N}\): semantic covariance
- \(\alpha_j \in [0,1]\): weight


## Memory Field

Spectral field:
\[W(x) = \sum_{j=1}^N \alpha_j \cdot T_j \cdot h_j(x)\]

Spectral coefficients:
\[\hat{w}_k := \int_{\mathbb{T}^N} W(x) \cdot \overline{\phi_k(x)} dx = \sum_{j=1}^N \alpha_j T_j \cdot \hat{h}_{j,k}\]

Projection:
\[T(\ell) = \sum_k \hat{w}_k \cdot \psi_k(\ell) = \sum_j \alpha_j T_j \cdot \psi_j^{\mathbb{T}}(\ell) \cdot \int h_j(x)^2 dx\]
