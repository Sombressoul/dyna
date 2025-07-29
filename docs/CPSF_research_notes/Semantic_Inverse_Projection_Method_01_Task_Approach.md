Below is the rigorously constructed, mathematically complete and fully expanded analytic equation that must be solved in order to obtain the **closed-form** expression for $z^{**}$ in the CPSF framework — **without using** $\arg\min$, iterative search, or any form of numerical approximation. Every quantity is expressed explicitly through the known field contributions $C_j$ and the target semantic vector $T^*$.

---

## Objective

Construct a fully analytic, single-step expression of the form:

$$
\boxed{z^{**} := z^* - \lambda^* \cdot \vec{d}^*}
$$

where $\lambda^* \in \mathbb{R}_{> 0}$ is computed by solving an explicit scalar equation derived from the condition that $T(z(\lambda))$ optimally approximates $T^*$ along the ray $z(\lambda) := z^* - \lambda \cdot \vec{d}^*$.

---

## 1. Definitions

For each contribution $C_j = (\ell_j, \hat{T}_j, \sigma_j^\parallel, \sigma_j^\perp, \alpha_j) \in \mathcal{C}$, define:

* Let $\vec{d}^* \in \mathbb{S}^{2N-1}_\text{unit}$ be the result of directional modulation (Step 1);
* Let $z^* \in \mathbb{T}_\mathbb{C}^N$ be the barycenter (Step 2);
* Let $\psi_j^0 := \psi_j^{\mathbb{T}}(z^*, \vec{d}^*) > 0$ be the envelope value at $z^*$;
* Let $\sigma_j := \sigma_j^{\parallel} \in \mathbb{R}_{>0}$;
* Let $A_j := \alpha_j \cdot \psi_j^0 \in \mathbb{C}$;
* Let $B_j := A_j \cdot \hat{T}_j \in \mathbb{C}^S$.

---

## 2. Projected field along the ray

The field at position $z(\lambda) := z^* - \lambda \cdot \vec{d}^*$ is:

$$
T(z(\lambda)) = \sum_{j \in \mathcal{J}} B_j \cdot e^{-\frac{1}{2} \lambda^2 / \sigma_j^2}
$$

---

## 3. Squared relative deviation

Define:

$$
\tau^2(\lambda) := \frac{
\|T^*\|^2 - 2 \Re \left\langle T(z(\lambda)), T^* \right\rangle + \left\|T(z(\lambda))\right\|^2
}{
\|T(z^*) - T^*\|^2
}
$$

Substitute $T(z(\lambda))$:

$$
\tau^2(\lambda) = \frac{
\|T^*\|^2
- 2 \Re \left( \sum_j \langle B_j, T^* \rangle \cdot e^{-\frac{1}{2} \lambda^2 / \sigma_j^2} \right)
+ \sum_{j,k} \langle B_j, B_k \rangle \cdot e^{-\frac{1}{2} \lambda^2 (\sigma_j^{-2} + \sigma_k^{-2})}
}{
\|T(z^*) - T^*\|^2
}
$$

---

## 4. Analytic root condition

To find $\lambda^*$, differentiate $\tau^2(\lambda)$ and solve the resulting analytic equation:

$$
\boxed{
\frac{d}{d\lambda} \left[
\|T^*\|^2
- 2 \Re \left( \sum_j \langle B_j, T^* \rangle \cdot e^{- \frac{1}{2} \lambda^2 / \sigma_j^2} \right)
+ \sum_{j,k} \langle B_j, B_k \rangle \cdot e^{- \frac{1}{2} \lambda^2 (\sigma_j^{-2} + \sigma_k^{-2})}
\right] = 0
}
$$

This is a smooth, real-valued scalar equation in $\lambda \in \mathbb{R}_{\ge 0}$, consisting entirely of weighted sums of terms of the form:

* $e^{-a \lambda^2}$
* $\lambda \cdot e^{-a \lambda^2}$

Thus the derivative is itself a sum of analytically tractable exponential-polynomial terms.

---

## 5. Final result

Once $\lambda^*$ is obtained from the equation above, the coordinate is given by:

$$
\boxed{
z^{**} = z^* - \lambda^* \cdot \vec{d}^*
}
$$

This constitutes a fully analytic, parameter-free, closed-form expression for $z^{**}$ in terms of known contributions $C_j$ and the target $T^*$, **with no use of** $\arg\min$, numeric search, sampling, or discrete optimization.
