## $\lambda^*$

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
