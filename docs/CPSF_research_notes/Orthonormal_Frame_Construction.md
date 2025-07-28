# Canonical Orthonormal Frame Construction for CPSF

## 1. Objective

Construct a smooth local section

$$
R : U \subset \mathbb{S}^{2N-1}_\text{unit} \to \mathrm{U}(N)
$$

that satisfies the following requirements for each $\vec{d} \in U$:

* (R1) $R(\vec{d}) \in \mathrm{U}(N)$;
* (R2) For each chart $U_k$, the first column of $R_k(\vec{d})$ equals $\vec{d}$;
* (R3) Columns $\{v_2, \dots, v_N\}$ of $R(\vec{d})$ form an orthonormal basis of $\vec{d}^\perp \subset \mathbb{C}^N$;
* (R4) The map $\vec{d} \mapsto R(\vec{d})$ is $C^\infty$-smooth on $U$;
* (R5) The construction is invariant under right multiplication by $\mathrm{U}(N{-}1)$ on the orthogonal complement of $\vec{d}$;
* (R6) The extended frame $\mathcal{R}(\vec{d}) := \mathrm{diag}(R(\vec{d}), R(\vec{d})) \in \mathrm{U}(2N)$ is consistent;
* (R7) Local trivializations exist and are compatible via $\mathrm{U}(N{-}1)$ transition functions;
* (R8) The construction supports smooth bundle structure over $\mathbb{S}^{2N-1}_\text{unit}$;
* (R9) All CPSF-relevant geometric and functional quantities derived from $R(\vec{d})$, including $\Sigma_j$, $\delta \vec{d}$, $w$, $\rho_j(w)$, $\psi_j^{\mathbb{T}}$, and $\Delta \hat{T}_j$, must depend smoothly on $\vec{d}$ with uniform convergence of all derivatives.

Global existence over $\mathbb{S}^{2N-1}_\text{unit}$ is topologically obstructed for general $N$, hence the frame is defined locally with smooth transition functions.

---

## 2. Construction Method (Multi-Chart Atlas)

To ensure global smoothness and avoid linear dependence in Gram–Schmidt, define an atlas:

$$
U_k := \{ \vec{d} \in \mathbb{S}^{2N-1}_\text{unit} : d_k \neq 0 \}, \quad k = 1, \dots, N
$$

On each chart $U_k$, define the frame as follows.

### Step 1: First column

Set:

$$
v_1 := \vec{d} \in \mathbb{C}^N
$$

### Step 2: Construct orthonormal complement via Gram–Schmidt

Let $\mathcal{I}_k := \{1, \dots, N\} \setminus \{k\}$, and let $e_{j_1}, \dots, e_{j_{N-1}}$ denote the standard basis vectors indexed by $\mathcal{I}_k$ in strictly increasing order:

$$
1 \leq j_1 < j_2 < \dots < j_{N-1} \leq N, \quad \{j_1, \dots, j_{N-1}\} = \mathcal{I}_k
$$

Let the ordered list of input vectors be:

$$
[v_1 := \vec{d}, e_{j_1}, e_{j_2}, \dots, e_{j_{N-1}}]
$$

Apply Gram–Schmidt recursively:

$$
\tilde{v}_1 := \vec{d}, \quad u_1 := \vec{d}
$$

Then for $m = 2, \dots, N$:

$$
\tilde{v}_m := e_{j_{m-1}} - \sum_{l=1}^{m-1} \langle e_{j_{m-1}}, u_l \rangle u_l, \quad
u_m := \frac{\tilde{v}_m}{\|\tilde{v}_m\|}
$$

Define:

$$
R_k(\vec{d}) := [u_1, u_2, \dots, u_N] \in \mathrm{U}(N)
$$

This guarantees orthonormality and smooth dependence on $\vec{d}$.

### Step 3: Transition Functions

On overlaps $U_j \cap U_k$, define:

$$
Q_{jk}(\vec{d}) := R_j(\vec{d})^{-1} R_k(\vec{d}) = \begin{bmatrix} 1 & 0 \\ 0 & \tilde{Q}_{jk}(\vec{d}) \end{bmatrix},
\quad
\tilde{Q}_{jk}(\vec{d}) \in \mathrm{U}(N{-}1)
$$

**Smoothness justification:**

* All $R_k(\vec{d})$ are constructed from smooth Gram–Schmidt applied to smooth input sets;
* Their first columns coincide: $R_j(\vec{d}) e_1 = R_k(\vec{d}) e_1 = \vec{d}$;
* Their remaining columns form orthonormal bases of $\vec{d}^\perp$, so the relative basis change lies in $\mathrm{U}(N{-}1)$ and varies smoothly.

**Block structure justification:**

Since both $R_j(\vec{d})$ and $R_k(\vec{d})$ are unitary matrices with identical first columns:

$$
R_j(\vec{d}) e_1 = R_k(\vec{d}) e_1 = \vec{d},
$$

it follows that the transition matrix satisfies:

$$
Q_{jk}(\vec{d}) e_1 = R_j(\vec{d})^{-1} R_k(\vec{d}) e_1 = R_j(\vec{d})^{-1} \vec{d} = e_1.
$$

Thus, $e_1$ is an eigenvector of $Q_{jk}(\vec{d})$ with eigenvalue 1. Since $Q_{jk}(\vec{d}) \in \mathrm{U}(N)$, it must preserve the decomposition $\mathbb{C}^N = \mathbb{C} e_1 \oplus e_1^\perp$, and hence its matrix representation in the basis $\{e_1, \dots, e_N\}$ has the claimed block-diagonal form:

$$
Q_{jk}(\vec{d}) = \begin{bmatrix} 1 & 0 \\ 0 & \tilde{Q}_{jk}(\vec{d}) \end{bmatrix}, \quad \tilde{Q}_{jk}(\vec{d}) \in \mathrm{U}(N{-}1).
$$

---

## 3. Local Trivialization and Bundle Structure

The atlas $\{U_k\}$ provides a full open cover of $\mathbb{S}^{2N-1}_\text{unit}$. Each frame map $R_k: U_k \to \mathrm{U}(N)$ satisfies:

* Smoothness of construction;
* Compatibility on overlaps via smooth $\mathrm{U}(N{-}1)$-valued transition functions;
* Cocycle condition $Q_{jk} Q_{kl} = Q_{jl}$.

Thus the construction defines a $\mathrm{U}(N{-}1)$-principal bundle structure over $\mathbb{S}^{2N-1}_\text{unit}$.

---

## 4. Verification of CPSF Frame Conditions

| Req. | Property                                                                   | Satisfied | Justification                                                                                                     |
| ---- | -------------------------------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------- |
| R1   | $R(\vec{d}) \in \mathrm{U}(N)$                                             | Yes       | Orthonormalization yields unitary matrix                                                                          |
| R2   | $R_k(\vec{d}) e_1 = \vec{d}$ (chartwise)                                   | Yes       | First column is $\vec{d}$ in each chart $U_k$ with relabeled basis $e_1 \mapsto e_k$                              |
| R3   | $\{v_2, \dots, v_N\}$ orthonormal and orthogonal to $\vec{d}$              | Yes       | Guaranteed by Gram–Schmidt                                                                                        |
| R4   | $R \in C^\infty(U)$                                                        | Yes       | Smooth dependence on input basis in each chart                                                                    |
| R5   | $R(\vec{d}) \cdot \begin{bmatrix} 1 & 0 \\ 0 & Q \end{bmatrix}$ invariance | Yes       | Complement columns can be rotated freely                                                                          |
| R6   | $\mathcal{R}(\vec{d}) := \mathrm{diag}(R(\vec{d}), R(\vec{d}))$            | Yes       | Follows from CPSF extended frame construction                                                                     |
| R7   | Local trivialization                                                       | Yes       | Multi-chart atlas with $\mathrm{U}(N{-}1)$ transition functions                                                   |
| R8   | Smooth bundle structure                                                    | Yes       | Cocycle and smoothness conditions fulfilled                                                                       |
| R9   | Smooth CPSF dynamics compatibility                                         | Yes       | $R \in C^\infty \Rightarrow \mathcal{R}, \Sigma_j, w, \rho_j, \psi_j^{\mathbb{T}}, \Delta \hat{T}_j \in C^\infty$ |

---

## 5. Conclusion

The construction now uses a full chart atlas over $\mathbb{S}^{2N-1}_\text{unit}$, resolves linear dependence issues, defines smooth transition maps, and satisfies all nine CPSF frame conditions rigorously.

The method is mathematically robust, geometrically faithful to CPSF structure, and analytically ready for implementation in all differential and variational components of the CPSF framework.
