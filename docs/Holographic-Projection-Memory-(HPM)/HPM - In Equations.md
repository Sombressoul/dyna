# HPM - In Equations

## 1. Memory Field

**Continuous memory field:**

$$
W : \mathbb{R}^N \rightarrow \mathbb{R}^C
$$

**Discretized memory tensor:**

$$
W[x] \in \mathbb{R}^{D_1 \times D_2 \times \dots \times D_N \times C}, \quad x \in \mathbb{Z}^N
$$

**Voxel center positions:**

$$
x_i = \delta \cdot i, \quad i \in \mathbb{Z}^N, \quad \delta > 0
$$

**Trilinear interpolation:**

$$
W(x) = \sum_{i \in \mathcal{N}(x)} w_i(x) \cdot W[i], \quad \sum w_i(x) = 1
$$

**Continuous limit of discretized sum:**

$$
\lim_{\delta \to 0} \sum_{x \in \mathbb{Z}^N} W[x] \cdot K(x) \cdot \delta^N = \int W(x) \cdot K(x) \, dx
$$

**Multiscale memory hierarchy:**

$$
W^{(n)}(x) = \text{downsample}_n(W^{(0)}(x))
$$

---

## 2. Read Operation

**Projection surface:**

$$
\Phi : \mathbb{R}^{N-1} \rightarrow \mathbb{R}^N
$$

**Projection ray from surface point $u$:**

$$
\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u, \quad \|\mathbf{v}_u\| = 1
$$

**Direction re-normalization (to keep $|\mathbf v_u|{=}1$):**

$$
\mathbf{v}_u \leftarrow \frac{\mathbf{v}_u}{\max\bigl(\|\mathbf{v}_u\|_2, \varepsilon\bigr)}
$$

**Angular selectivity gate (orientation tuning):**

$$
S(\theta) = \exp\!\bigl(-\kappa \,\theta^{2}\bigr), 
\qquad 
\theta = \arccos\!\bigl(\langle \mathbf v_u,\mathbf v_{\text{pref}}\rangle\bigr)
$$

**Angle-modulated kernel (replace $K$ by $K_{\text{ang}}$ wherever selective beams are needed):**

$$
K_{\text{ang}}(x,\ell_u) = S(\theta) K(x,\ell_u)
$$

**Axial distance from surface to point $x$:**

$$
t(x) = (x - \Phi(u)) \cdot \mathbf{v}_u = \langle x - \Phi(u), \mathbf{v}_u \rangle
$$

**Clipped attenuation (restart decay at the true entry point):**

$$
A_{\text{clipped}}(t) = \exp\!\Bigl(-\tfrac{t - t_{\text{entry}}}{\tau_u}\Bigr),
\qquad t \in [\,t_{\text{entry}},\,t_{\text{exit}}\,]
$$

**Perpendicular distance to ray:**

$$
d_\perp(x) = \left\| x - (\Phi(u) + t(x) \cdot \mathbf{v}_u) \right\|
$$

**Projection kernel:**

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp(x)^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{t(x)}{\tau} \right)
$$

**Support cutoff condition:**

$$
K(x, \ell_u) = 0 \quad \text{if } d_\perp(x) > r_{\max} \text{ or } t(x) < 0
$$

**Unnormalized projection response:**

$$
T(u) = \int W(x) \cdot K(x, \ell_u) \, dx
$$

**Normalized projection response:**

$$
T(u) = \frac{\int W(x) \cdot K(x, \ell_u) \, dx}{\int K(x, \ell_u) \, dx}
$$

**Discrete approximation on voxel grid:**

$$
T(u) \approx \sum_{x \in \Omega_u} W[x] \cdot K(x, \ell_u)
$$

**Rasterized projection over ray path:**

$$
T(u) \approx \sum_{i=1}^N W[x_i] \cdot K(x_i, \ell_u), \quad x_i \in \text{Ray}(\ell_u)
$$

**Ray–AABB intersection limits (entry / exit clipping):**

$$
t_{\min} = \max_{i}\!
        \Bigl(\!
            \min\Bigl(\frac{b_i^{\text{min}}-\Phi_i(u)}{v_{u,i}},
                      \frac{b_i^{\text{max}}-\Phi_i(u)}{v_{u,i}}\Bigr)
        \Bigr)
$$
$$
t_{\max} = \min_{i}\!
        \Bigl(\!
            \max\Bigl(\frac{b_i^{\text{min}}-\Phi_i(u)}{v_{u,i}},
                      \frac{b_i^{\text{max}}-\Phi_i(u)}{v_{u,i}}\Bigr)
        \Bigr)
$$

> The ray is **active** if $t_{\max} > \max(0,t_{\min})$

---

## 3. Write Operation

**Projection error:**

$$
\delta(u) = T^*(u) - T(u)
$$

**Single-ray memory update:**

$$
\Delta W(x) = \alpha \cdot \delta(u) \cdot K(x, \ell_u)
$$

**Multi-ray accumulation:**

$$
\Delta W(x) = \sum_u \alpha_u \cdot \delta(u) \cdot K(x, \ell_u)
$$

**Multi-channel correction (spectral or vector-valued):**

$$
\Delta W(x) = \sum_k \delta_k(u) \cdot K_k(x, \ell_u)
$$

**Normalization (optional):**

$$
\Delta W(x) \leftarrow \frac{\Delta W(x)}{\int K(x, \ell_u) \, dx}
$$

**Update integration step:**

$$
W(x) \leftarrow W(x) + \Delta W(x)
$$

---

## 4. Search Procedure

**Multiscale memory at level $n$:**

$$
W^{(n)}(x), \quad \ell_u^{(n)}(t) = \Phi^{(n)}(u) + t \cdot \mathbf{v}_u
$$

**Directional projection at level $n$:**

$$
T^{(n)}(u) = \int W^{(n)}(x) \cdot K(x, \ell_u^{(n)}) \, dx
$$

**Scalar score function:**

$$
S(u) = f(T^{(n)}(u)), \quad f: \mathbb{R}^C \rightarrow \mathbb{R}
$$

**Top-k projection candidates:**

$$
\mathcal{U}^{(n)} = \text{top}_k \{ u : S(u) \text{ maximal} \}
$$

**Backprojection region from selected directions:**

$$
\mathcal{R}^{(n)} = \bigcup_{u \in \mathcal{U}^{(n)}} \{ x \in \mathbb{R}^N : \|x - \ell_u^{(n)}(t)\| < \varepsilon \}
$$

**Recursive descent:**

$$
W^{(n-1)}(x) \leftarrow \text{refine over } \mathcal{R}^{(n)}, \quad n \to n{-}1
$$

**Terminal resolution (LOD$_0$) result:**

$$
\mathcal{R}^{(0)} = \text{final candidate regions}
$$

**LOD-to-LOD beam-width schedule:**

$$
\sigma^{(n-1)} = \gamma\,\sigma^{(n)},
\qquad 0<\gamma<1
$$

---

## 5. Bidirectional Projection

**Forward and backward rays from point $u$:**

$$
\ell_u^{(+)}(t) = \Phi(u) + t \cdot \mathbf{v}_u
$$

$$
\ell_u^{(-)}(t) = \Phi(u) - t \cdot \mathbf{v}_u
$$

**Directional projections:**

$$
T^{(+)}(u) = \int W(x) \cdot K(x, \ell_u^{(+)}) \, dx
$$

$$
T^{(-)}(u) = \int W(x) \cdot K(x, \ell_u^{(-)}) \, dx
$$

**Combined projection options:**

$$
T(u) = T^{(+)}(u) + T^{(-)}(u)
$$

$$
T(u) = [T^{(+)}, T^{(-)}] \in \mathbb{R}^{2C}
$$

**Bidirectional update rule:**

$$
\Delta W(x) = \alpha \cdot \left( \delta^{(+)}(u) \cdot K(x, \ell_u^{(+)}) + \delta^{(-)}(u) \cdot K(x, \ell_u^{(-)}) \right)
$$

---

## 6. Gradients

**Gradient of projection with respect to memory field:**

$$
\frac{\partial T(u)}{\partial W(x)} = K(x, \ell_u)
$$

**Gradient with respect to projection surface:**

$$
\nabla_{\Phi(u)} T(u) = \int W(x) \cdot \nabla_{\Phi} K(x, \ell_u) \, dx
$$

**Gradient w.r.t. direction vector $\mathbf{v}_u$:**  

$$
\nabla_{\mathbf{v}_u} T(u) = \int W(x) \cdot \nabla_{\mathbf{v}} K(x, \ell_u) \, dx
$$

**Surrogate direction approximation (if $\mathbf{v}_u$ non-differentiable):**  

$$
\mathbf{v}_u \approx \frac{B - A}{\|B - A\|}, \quad A, B = \text{entry/exit points}
$$

**Gradient with respect to decay parameter $\tau$:**

$$
\frac{\partial T(u)}{\partial \tau} = \int W(x) \cdot K(x, \ell_u) \cdot \frac{t(x)}{\tau^2} \, dx
$$

**Gradient w.r.t. transverse width $\sigma$ (for Gaussian kernel):**  

$$
\frac{\partial T(u)}{\partial \sigma} = \int W(x) \cdot K(x, \ell_u) \cdot \frac{d_\perp(x)^2}{\sigma^3} \, dx
$$

**Surrogate gradient for traced rays:**

$$
\mathbf{v}_u \approx \frac{B - A}{\|B - A\|}, \quad A,B = \text{entry/exit points}
$$

---

## 7. Spectral Extension

**Spectral memory field representation:**

$$
W(x) = \sum_k \hat{w}_k(x) \cdot \phi_k(x), \quad \hat{w}_k(x) \in \mathbb{C}
$$

**Fourier or learned basis functions:**

$$
\phi_k(x) = e^{i 2 \pi f_k x} \quad \text{or} \quad \phi_k \in \mathcal{B}_{\text{learned}}
$$

**Directional projection per frequency:**

$$
T_k(u) = \int \hat{w}_k(x) \cdot \phi_k(x) \cdot K(x, \ell_u) \, dx
$$

**Spectral response vector:**

$$
T(u) = [T_0(u), T_1(u), \dots, T_{K-1}(u)] \in \mathbb{C}^K
$$

**Projection error in spectral space:**

$$
\delta_k(u) = T_k^*(u) - T_k(u)
$$

**Update for each spectral component:**

$$
\Delta \hat{w}_k(x) = \alpha \cdot \delta_k(u) \cdot \phi_k^*(x) \cdot K(x, \ell_u)
$$

**Reconstruction of memory update:**

$$
\Delta W(x) = \sum_k \Delta \hat{w}_k(x) \cdot \phi_k(x)
$$

**Spectral memory projection shape (discrete):**

$$
T(u) \in \mathbb{R}^{R_x \times R_y \times K \times 2}
$$

**Interpretation:** real and imaginary parts per frequency dimension.

---

### 8. Delta-Learning (Integrated Form)

**Continuous update across projection domain:**  

$$
\Delta W(x) = \int \alpha(u) \cdot \delta(u) \cdot K(x, \ell_u) \, du
$$

**Discrete approximation:**  

$$
\Delta W(x) \approx \sum_j \alpha_j \cdot \delta_j \cdot K(x, \ell_{u_j})
$$

**Online-plasticity (per-inference step) rule:**

$$
W_{t+1}(x) = W_{t}(x) + \eta_{t} \delta_{t}(u) K\bigl(x,\,\ell_{u,t}\bigr)
$$
$$
\eta_{t} = \frac{\eta_{0}}{1+\lambda t}
$$

---

### 9. Associative Backprojection Operator

**Adjoint projection operator:**  

$$
\mathcal{T}^*[\delta](x) := \int \delta(u) \cdot K(x, \ell_u) \, du
$$

**Linearity and superposition:**  

$$
\Delta W(x) = \sum_{i=1}^n \Delta W_i(x) = \sum_{i=1}^n \alpha_i \cdot \delta_i \cdot K(x, \ell_{u_i})
$$

---

### 10. Topological Divergence & Repulsion

**Gradient interference field:**  

$$
\vec{F}(x) = -\nabla_x \Delta W(x)
$$

**Overlap energy between memory modes:**  

$$
U_{ij} = \int \rho_i(x) \cdot \rho_j(x) \, dx
$$

**Repulsive interaction force:**  

$$
F_{ij} = \frac{x_i - x_j}{\sigma_i^2 + \sigma_j^2} \cdot U_{ij}
$$

**Cluster center dynamics:**  

$$
\frac{dx_i}{dt} = \sum_{j \ne i} F_{ij} - \gamma \cdot \dot{x}_i + \sqrt{2T} \cdot \xi(t)
$$

**Angular-divergence acceleration criterion:**

$$
\theta > \theta_{\text{crit}}
\Longrightarrow
\frac{d}{dt}\bigl\|x_1 - x_2\bigr\|
\ge \beta\,\sin^{2}\theta,
\qquad \beta>0
$$

---

### 11. Adaptive Kernel Width

**Local semantic density estimate:**  

$$
\rho_{\text{loc}}(x_i) = \sum_{j \ne i} U_{ij}
$$

**Density-modulated kernel width:**  

$$
\sigma_i = \sigma_0 \cdot \exp(-\beta \cdot \rho_{\text{loc}}(x_i))
$$

---

### 12. Kernel Variants

**Laplacian kernel:**  

$$
K_{\text{Lap}}(x, \ell) = \exp\left( -\frac{d(x, \ell)}{\sigma} \right)
$$

**Hybrid kernel (generalized decay):**  

$$
K_{\text{hybrid}}(x, \ell) = \exp\left( -\left( \frac{d(x, \ell)}{\sigma} \right)^\alpha \right), \quad \alpha \in (1, 2]
$$
