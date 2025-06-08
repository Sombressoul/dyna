# Appendix D — Directional Projection Protocol

> *Mathematical formulation and implementation guidelines for coherent directional projection in Holographic Projection Memory (HPM) systems.*

---

## D.1 Directional Ray Model

Let $W(x)$ be a differentiable memory field defined over $x \in \mathbb{R}^N$, and let $\mathcal{P}(u)$ denote a projection hypersurface of dimension $N-1$, parametrized by $u \in \mathbb{R}^{N-1}$.

We define a **directional ray** from each point $u$ as:

$$
\ell_u(t) = \Phi(u) + t \cdot v, \quad t \in \mathbb{R}
$$

Where:

* $\Phi(u) \in \mathbb{R}^N$ is the embedding of point $u$ into the memory space,
* $v \in \mathbb{R}^N$ is a **global unit direction vector**, shared across all $u$,
* The system supports **bidirectional projection**, evaluating both $+v$ and $-v$ paths from $\Phi(u)$.

---

## D.2 Attenuation Model Along the Ray

Each directional ray has **exponentially decaying intensity** along its axis:

$$
A(t) = \exp\left(-\frac{t}{\tau}\right)
$$

Where:

* $t$ is the signed distance along the ray from the origin point $\Phi(u)$,
* $\tau > 0$ is the decay constant controlling the **penetration depth** of the beam.

---

## D.3 Kernel-Based Contribution of Memory Points

Let $x \in \mathbb{R}^N$ be a point in memory field $W(x)$. Define:

* $t\_x$ as the value minimizing $|x - \ell\_u(t)|^2$, i.e., the closest point on the ray,
* $d\_\perp = |x - \ell\_u(t\_x)|$ (perpendicular distance),
* $d\_\parallel = |t\_x|$ (longitudinal distance along the beam).

The **contribution kernel** is:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{d_\parallel}{\tau} \right)
$$

The projected value is then:

$$
T(u) = \int_{\mathbb{R}^N} W(x) \cdot K(x, \ell_u) \, dx
$$

---

## D.4 Gradient Computation

The projection $T(u)$ is differentiable with respect to both $W(x)$ and the ray parameters:

* $\frac{\partial T(u)}{\partial W(x)} = K(x, \ell_u)$
* $\nabla_{\Phi} T(u)$, $\nabla_{v} T(u)$ are computed via chain rule, involving $\nabla d\_\perp$, $\nabla d\_\parallel$.

This allows full backpropagation for optimizing both memory content and projection geometry.

---

## D.5 Local Update Without Backpropagation

If the target projection $T^*(u)$ is known (e.g., from an external embedding or codebook), then we define the projection error:

$\delta(u) = T^*(u) - T(u)$

We can update the memory directly via a **local rule**, without computing gradients from downstream loss:

$$
W(x) \leftarrow W(x) + \alpha \cdot \delta(u) \cdot K(x, \ell_u)
$$

Where $\alpha$ is a step size. This update is localized, semantically weighted, and supports **online self-imprinting** of memory during inference.

---

> *This protocol defines a coherent, energy-aware method for projecting, interpreting, and modifying distributed memory through directional ray-based mechanisms in high-dimensional neural fields.*
