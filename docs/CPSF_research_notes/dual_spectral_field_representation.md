## Dual-Spectral Field Representation

In the CPSF model, memory is structured as a smooth, continuous, and differentiable tensor field defined over a complex toroidal spectral domain. This field exhibits a fundamental **dual-spectral structure**, in which:

1. **Geometric spectral space** defines the coordinates of the field,
2. **Semantic spectral content** is stored at each coordinate point.

This results in a field-of-fields architecture, where local semantics are embedded in a global spectrally-defined geometry.

---

### 1. Geometric Spectral Space

Let \( x \in \mathbb{T}^N \subset \mathbb{C}^N \) denote points in a global complex toroidal coordinate space. This space is not defined directly in terms of metric distance, but instead via its **global Fourier basis**:

\[
\phi_k(x) = e^{2\pi i \langle k, x \rangle}, \quad k \in \mathbb{Z}^N
\]

This basis defines the geometry of the memory field: the topology, continuity, and periodicity of space. The coordinates \( x \) are implicitly spectral: every spatial relationship is governed by interference of these harmonics.

---

### 2. Semantic Spectral Field

Let \( W(x) \in \mathbb{C}^S \) be the semantic field defined over \( x \in \mathbb{T}^N \). At each point \( x \), instead of a single value, we store a **semantic spectrum** \( W(x) \): a complex-valued vector over a separate semantic basis (not explicitly parameterized here).

This turns the global field into a tensor-valued distribution:

\[ W: \mathbb{T}^N \to \mathbb{C}^S \]

---

### 3. Memory Contributions as Dual-Spectral Objects

Each memory contribution \( C_j \) consists of:

\[ C_j = (\ell_j, \Sigma_j, \hat{T}_j) \]

where:

- \( \ell_j = (\vec{o}_j, \vec{d}_j) \in \mathbb{T}^{2D} \subset \mathbb{C}^{2D} \): geometric center and direction in spectral coordinates,
- \( \Sigma_j \in \mathbb{S}_{++}^{2D} \): anisotropic projection kernel (defining spatial spread in geometric spectrum),
- \( \hat{T}_j \in \mathbb{C}^S \): semantic spectral vector.

The semantic contribution from \( C_j \) is distributed into the field via a smooth toroidal Gaussian:

\[
W_j(x) = h_j(x) \cdot \hat{T}_j
\]

where:

\[
h_j(x) := \sum_{m \in \mathbb{Z}^N} \exp\left[ -(x - x_j + m)^\top \Lambda_j^{-1} (x - x_j + m) \right]
\]

The total semantic field is then:

\[
W(x) = \sum_j W_j(x) = \sum_j h_j(x) \cdot \hat{T}_j
\]

---

### 4. Reading via Spectral Projection

A projection kernel \( K(x, \ell) \) defined by a complex anisotropic Gaussian \( \psi_j^{\mathbb{T}}(\ell) \) and the field \( h_j(x) \) is used to extract semantic meaning from the field:

\[
T(\ell) = \int_{\mathbb{T}^N} K(x, \ell) \cdot W(x) \, dx = \sum_j \psi_j^{\mathbb{T}}(\ell) \cdot \hat{T}_j \cdot \int_{\mathbb{T}^N} h_j(x)^2 dx
\]

This results in a projected semantic vector \( T(\ell) \in \mathbb{C}^S \), synthesized by spectral interference of locally overlapping semantic contributions.

---

### 5. Interpretation

- **Global geometry** is encoded spectrally (Fourier space \( \mathbb{Z}^N \))
- **Local semantics** are encoded spectrally (vector \( \hat{T}_j \in \mathbb{C}^S \))
- Memory = distribution of semantic spectra over spectral coordinates
- Reading = interference-aware projection of local spectra via geometric kernels

This dual-spectral formulation underlies the CPSF model: geometric information and semantic meaning are both stored and processed spectrally — but in orthogonal, interwoven domains.

---

### 6. Summary

> **CPSF memory is a complex-valued tensor field over a geometrically spectral space, where each coordinate stores an embedded semantic spectrum.**

This representation enables reversible, differentiable, and analytically tractable memory operations — integrating spatial localization with spectral semantic interaction in a unified field framework.

