DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED

---

## CPSF: Definition and Construction of the Orthonormal Basis  

Let $D \in \mathbb{N}$, $\vec{d} \in \mathbb{C}^D$ such that $\|\vec{d}\| = 1$, and let $\vec{e}_1 = (1, 0, \ldots, 0)^\top \in \mathbb{C}^D$.  

---

### Definition: Canonical Orthonormal Basis $R(\vec{d}) \in \mathrm{U}(D)$  

Define $R(\vec{d}) \in \mathbb{C}^{D \times D}$ as a unitarily orthonormal matrix satisfying the following conditions:  

1. **First column anchoring:**  
   $$
   R(\vec{d})[:,1] := \vec{d}
   $$

2. **Unitarity:**  
   $$
   R(\vec{d})^\dagger R(\vec{d}) = I_D
   $$

3. **Exponential construction:**  
   There exists an anti-Hermitian generator $A(\vec{d}) \in \mathfrak{u}(D)$ such that:  
   $$
   R(\vec{d}) := \exp(A(\vec{d}))
   \quad \text{and} \quad
   \exp(A(\vec{d})) \cdot \vec{e}_1 = \vec{d}
   $$

4. **Generator rank constraint:**  
   $$
   \operatorname{rank}(A(\vec{d})) \le 2
   \quad \text{and} \quad
   \operatorname{rank}(A(\vec{d})) = 0 \Leftrightarrow \vec{d} = \vec{e}_1
   $$

5. **Generator structure:**  
   Let:  
   $$
   \vec{u} := \frac{\vec{e}_1 - \vec{d}}{\|\vec{e}_1 - \vec{d}\|},
   \quad
   \vec{v} := \frac{\vec{e}_1 + \vec{d}}{\|\vec{e}_1 + \vec{d}\|}
   \quad \Rightarrow \quad
   A(\vec{d}) := \pi \cdot \left( \vec{u} \vec{v}^\dagger - \vec{v} \vec{u}^\dagger \right)
   $$  
   Then $A(\vec{d})^\dagger = -A(\vec{d})$, and $\operatorname{rank}(A(\vec{d})) \le 2$.  

6. **Rank justification:**  
   The matrix $A(\vec{d})$ is a linear combination of two rank-1 matrices.  
   Let $M_1 := \vec{u} \vec{v}^\dagger$ and $M_2 := \vec{v} \vec{u}^\dagger$, then:
   $$
   A(\vec{d}) = \pi(M_1 - M_2), \quad \operatorname{rank}(M_1 - M_2) \le \operatorname{rank}(M_1) + \operatorname{rank}(M_2) = 2
   $$
   Moreover, if $\vec{d} \notin \mathbb{C} \cdot \vec{e}_1$, then $\vec{u}$ and $\vec{v}$ are linearly independent, so $\operatorname{rank}(A(\vec{d})) = 2$.

7. **Smoothness:**  
   The map $\vec{d} \mapsto R(\vec{d})$ is infinitely differentiable on $S^{2D-1} \subset \mathbb{C}^D$.  

8. **Phase equivariance:**  
   $$
   \forall \phi \in \mathbb{R}, \quad
   R(e^{i\phi} \vec{d}) = R(\vec{d}) \cdot \mathrm{diag}(e^{i\phi}, 1, \ldots, 1)
   $$

---

### Properties  

- $R(\vec{d}) \in \mathrm{U}(D)$ for all $\vec{d} \in \mathbb{C}^D$, $\|\vec{d}\| = 1$  
- $R(\vec{d})$ depends solely on $\vec{d}$ without auxiliary randomness or Gram-Schmidt procedures  
- $R(\vec{d})$ is canonically defined and suitable for use in toroidally equivariant projection kernels
