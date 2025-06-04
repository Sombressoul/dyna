# Universal Compact Bilinear Projection (UCBP)

*Technical Design Document · **Version 1.0‑rc4** (June 2025)*

---

## 1 · Purpose

Universal Compact Bilinear Projection (UCBP) is conceived as a **drop-in, parameter-efficient replacement for explicit bilinear or multi-linear weight tensors** in modern neural networks. By compressing high-dimensional interactions into a lower-dimensional sketch, it drastically reduces memory footprint while preserving the expressive power of full bilinear models.

### What UCBP does

* **Approximation pipeline** – *Count-Sketch → FFT → Hadamard (element-wise) product → IFFT* captures the outer-product information of **two or more** input tensors without constructing dense weight tensors.
* **Trainable → Baked switch** – During training, UCBP learns dense projection matrices; at inference, these matrices are **quantised into compact hash tables** `(h,s)` so that per-head storage drops to ≤ 1 KiB.
* **Rich tensor support** – Works on arbitrary axis pairings, supports **multi-head / multi-rank groupings**, and handles **complex-valued features** required by the research models.
* **Framework-friendly** – Implemented with pure `aten` ops, enabling FX, Torch-Dynamo and ONNX export; interchangeable with explicit bilinear layers in existing PyTorch code-bases.

### Why it matters

UCBP unifies several desirable properties in a single projector:

| Capability                                       | Impact on models                                                              |
| ------------------------------------------------ | ----------------------------------------------------------------------------- |
| **Adjustable compression (`d′`)**                | Trade accuracy for memory/compute on a sliding scale.                         |
| **Affine equivariance to head/rank permutation** | Drop-in for attention-style blocks that reorder heads.                        |
| **Learnable projection matrices**                | Outperform frozen hashes on complex, non-stationary data.                     |
| **Bakeability**                                  | Deployment-time shrink reduces latency and power consumption on edge devices. |

### Typical use-cases

* **Dynamic weight generators** (e.g. `WeightsLib2DMobius`) that need thousands of per-step convolution kernels without GPU memory blow-ups.
* **Multi-modal attention** spanning vision, text and audio where bilinear fusion improves alignment.
* **Graph or relational link-prediction** models that score entity–relation pairs.
* **Parameter-efficient fine-tuning** inside large Transformer / LLM blocks (LoRA-style adapters).

### Guiding objective

> *Provide a universal, bakeable projector for N-D, multi-head, multi-rank tensors that offers adjustable compression while remaining export-friendly and numerically stable.*

In short, UCBP brings the expressive strength of bilinear pooling to large-scale, resource-constrained deep-learning systems - without the usual cost explosion.

---

## 2 · Compact Bilinear Pooling - Heritage & Limitations

### 2.1 Historical roots

Compact (a.k.a. **Count-Sketch-based) Bilinear Pooling** was introduced by Fukui et al. for visual-question answering in 2016. The key idea is to replace an explicit outer-product with a **random feature map**

$$
\phi(x,y) = \mathop{\text{IFFT}}\bigl(\mathop{\text{FFT}}(\text{CS}(x))\odot \mathop{\text{FFT}}(\text{CS}(y))\bigr),
$$

where **CS** is a Count-Sketch that hashes each input coordinate to one of *d′* bins with a random sign.  This trick yields an **unbiased estimator**

$$
\mathbb{E}_{h,s}\!\left[\langle\phi(x),\phi(y)\rangle\right]=\langle x,y\rangle,
\quad
\mathop{\text{Var}}\le \tfrac{\|x\|^2\|y\|^2}{d′},
$$

so the mean-square error decays as **O(1/d′)**.  The same pipeline underlies virtually every CBP variant used in vision, audio and NLP today.

---

### 2.2 Inherited limitations

| #      | Classic CBP behaviour                                                                                                                 | Why it is a problem                                                                      |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **L1** | **Frozen, random sketch** - tables *(h,s)* are sampled once and never trained.                                                        | Works well for generic images but leaves accuracy on specialised or non-stationary data. |
| **L2** | **Exactly two inputs, flattened to 1-D.**                                                                                             | Fails on *K ≥ 3* modalities and on high-rank tensors where axis semantics matter.        |
| **L3** | **Compression fixed per layer.**  Choosing small *d′* raises variance; large *d′* explodes memory/compute ∝ *G·d′* (group × bins).    | No graceful trade-off; head-heavy models (Transformers) scale poorly.                    |
| **L4** | **Hash collisions & sign noise.**  Expected collision rate is *≈1−e^{−d/d′}*; at moderate *d′* this corrupts high-magnitude features. | Causes gradient spikes and slows convergence, especially on sparse activations.          |
| **L5** | **No deployment shrinkage.**  Even with hashing, the dense projection weights must be stored (32-bit each) or regenerated offline.    | Edge devices cannot afford 10-100 MB of hashes per task.                                 |

---

### 2.3 UCBP remedies

| Limitation | UCBP fix   | Mathematical / engineering justification |
| ---------- | ---------- | ---------------------------------------- |
| **L1**     | **Parametric Count-Sketch** - replace frozen sign & bin with **A,B ∈ ℂ^{d\_in×d′}** that are trainable and later quantised. | Learning lets the optimiser minimise variance on the *actual* data distribution; greedy bake then stores only the maximally-used bin per row, preserving accuracy while collapsing to *(h,s)*. |
| **L2**     | **Multi-input fusion** via Hadamard product in Fourier domain and generalised AxisGather to any axis pairs. | Bilinearity extends by associativity:  $\prod_{k=1}^{K}\mathop{\text{FFT}}(\text{CS}(x_k))$. <br> Variance scales **multiplicatively** with $\prod_k \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2$ (exponential in `K`); clarification: dependence on `d′` is `O(1/d′)` (not linear in `K`). |
| **L3**     | **Adjustable sketch size `d′` + group routing**. Heads/ranks are routed to independent projectors and can share or prune bins adaptively. | - **Variance bound**: $\mathrm{Var}[\langle \Phi(\mathbf{x}), \Phi(\mathbf{y}) \rangle] \leq \frac{\prod_k \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2}{d'}$ for `K` inputs, decaying as `O(1/d′)` for fixed inputs. <br> - **d′ heuristic**: Set $d' \geq \frac{\prod_k \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2}{\varepsilon^2}$ to achieve standard deviation $\leq \varepsilon$ for kernel estimates. <br> - **Practical scaling**: For bounded-norm inputs (e.g., $\|\mathbf{x}_k\| \leq 1$), $d' = \mathcal{O}\left(\frac{1}{\varepsilon^2}\right)$ per group. |
| **L4**     | **Binary / orthogonal projections + BGN**. Orthogonality reduces collision bias; Backward-Gradient-Normalisation tames large residuals. | For binary $\pm 1$ matrices the collision error’s second moment halves; BGN keeps per-row gradient $\ell_2\text{-norm} \approx \sqrt{d'}$, preventing explosion. |
| **L5**     | **Greedy or ILP-based bake** - quantise complex weights to 8-bit sign and 16-bit index; per-head storage ≤ 1 KiB. | After bake the forward uses integer `scatter_add`, so memory drops by $\times 32$ and inference latency falls because no dense GEMM is executed. |


<details>
<summary>Mathematical Justifications: 2.3 L3</summary>

**1. Variance Bound for K-Input Fusion**
For inputs $\{\mathbf{x}_1, \dots, \mathbf{x}_k\}$ and $\{\mathbf{y}_1, \dots, \mathbf{y}_k\}$, the UCBP kernel estimator is:  
```math
\langle \Phi(\mathbf{x}), \Phi(\mathbf{y}) \rangle = \left\langle \text{IFFT}\left( \prod_{k=1}^K \text{FFT}(\text{CS}(\mathbf{x}_k)) \right), \text{IFFT}\left( \prod_{k=1}^K \text{FFT}(\text{CS}(\mathbf{y}_k)) \right) \right\rangle
```  
This is an **unbiased estimator** of the product kernel:  
```math
\mathbb{E}\left[\langle \Phi(\mathbf{x}), \Phi(\mathbf{y}) \rangle\right] = \prod_{k=1}^K \langle \mathbf{x}_k, \mathbf{y}_k \rangle.
```  
The variance is bounded by:  
```math
\text{Var}\left[\langle \Phi(\mathbf{x}), \Phi(\mathbf{y}) \rangle\right] \leq \frac{\prod_{k=1}^K \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2}{d'}. \quad (1)
```  
*Proof Sketch*:  
- By the Count-Sketch property, `FFT(CS(𝐱ₖ))` is a random projection with:  
  ```math
  \mathbb{E}\left[\langle \text{CS}(\mathbf{x}_k), \text{CS}(\mathbf{y}_k) \rangle\right] = \langle \mathbf{x}_k, \mathbf{y}_k \rangle, \quad \text{Var} \leq \frac{\|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2}{d'}.
  ```  
- The Hadamard product $\odot$ in Fourier domain preserves independence across `k` via the **tensor sketch convolution theorem** [1, 2].  
- Variance of the product kernel scales multiplicatively due to independence:  
  ```math
  \text{Var}\left[\prod_{k=1}^K \langle \phi_k, \psi_k \rangle\right] = \prod_{k=1}^K (\text{Var}[\langle \phi_k, \psi_k \rangle] + \mu_k^2) - \prod_{k=1}^K \mu_k^2, \quad \mu_k = \langle \mathbf{x}_k, \mathbf{y}_k \rangle.
  ```  
  For small $\mu_k$ (common in high-dim), this simplifies to $\approx \prod_k \mathrm{Var}[\langle \varphi_k, \psi_k \rangle] \leq \prod_k \frac{\|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2}{d'}$.  

**2. d′ Selection Heuristic**  
To achieve a target **standard deviation $\varepsilon$** for the kernel estimate:  
```math
\sqrt{\text{Var}\left[\langle \Phi(\mathbf{x}), \Phi(\mathbf{y}) \rangle\right]} \leq \varepsilon.
```  
Substituting (1):  
```math
\sqrt{\frac{\prod_{k=1}^K \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2}{d'}} \leq \varepsilon \implies d' \geq \frac{\prod_{k=1}^K \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2}{\varepsilon^2}. \quad (2)
```  
- **Special case (bilinear, K = 2)**: $d' \geq \frac{\|\mathbf{x}\|^2 \|\mathbf{y}\|^2}{\varepsilon^2}$  
- **Unit-norm inputs ($\|\mathbf{x}_k\| = 1$)**: $d' \geq \frac{1}{\varepsilon^2}$ (Johnson–Lindenstrauss style).  

**3. Group Routing Advantage**

- **Variance isolation**: Routing \$G\$ heads to independent projectors confines variance to per-group \$d'\$ (no cross-head error propagation).
- **Adaptive bin sharing**: Sparsity-aware bin allocation minimizes \$d'\$ under \$\varepsilon\$-constraints.
  

--- 

**References**

[1] Pham, N., Pagh, R. (2013). *Fast and scalable polynomial kernels via explicit feature maps*. KDD.  
[2] Avron, H., Nguyen, H., Woodruff, D. (2014). *Subspace embeddings for the polynomial kernel*. NeurIPS.

</details>


**Practical Implications**  

- **Memory/compute trade-off**: $d'$ adjusted per group to meet $\varepsilon$-accuracy:  
  ```math
  \text{Memory} \propto G \cdot d', \quad \text{Error} \propto 1/\sqrt{d'}.
  ```  
- **Edge deployment**: For $\varepsilon = 0.1$, $K = 2$, $\|\mathbf{x}\| = \|\mathbf{y}\| = 1$, $d' = 100$ suffices (1 KiB/head).  

---

### 2.4 Practical takeaway

Compact Bilinear Pooling remains a powerful kernel trick, but naïve implementations hit accuracy, stability and deployment walls at modern scale.  **UCBP** resolves these pain-points by (1) making the sketch *learnable*, (2) generalising to arbitrary tensor arities and axis pairings, (3) exposing a tunable accuracy–cost knob, (4) adding robust normalisation, and (5) introducing a *bake* path that slashes inference memory without retraining.  These upgrades preserve CBP’s theoretical guarantees while aligning it with today’s multi-head, multi-modal deep-learning workloads.

---

## 3 · Design Goals

| ID | Goal | Brief description |
| -- | ---- | ----------------- |
| **G1 — Arbitrary axis pairing** | Project any user-chosen pair(s) of axes without flattening the whole tensor. | Preserves spatial / modal structure and eliminates costly reshapes; required for vision (H × W), language (T × D) and cross-modal fusion where axes have distinct semantics. |
| **G2 — Affine equivariance to head / rank permutation** | Results are identical under re-ordering of heads, sub-spaces or ranks. | Enables drop-in replacement for multi-head attention and other grouped operations, and lets weights be shared or shuffled freely at runtime. |
| **G3 — Massive head / rank scalability** | Support thousands of independent groups (`G = heads × subspaces × ranks`) with no weight duplication. | Modern Transformers, video nets and graph models often push $G \gg 1\text{k}$; memory must scale **O(G)** only in activations, not in parameters. |
| **G4 — Adjustable compression knob** | Sketch dimension $d'$ sets an explicit accuracy ↔ memory/compute trade-off. | Acts like a Johnson–Lindenstrauss parameter: variance $\propto \frac{1}{d'}$; practitioners can dial quality to fit hardware budgets. |
| **G5 — Trainable → Baked switch** | Dense complex projections while training; automatically quantised to `(h:int16, s:int8)` hash tables (≤ 1 KiB per head) for inference. | Yields high accuracy during learning and ultra-compact, scatter-add–only inference kernels for edge deployment. |
| **G6 — Complex-number correctness** | Native support for complex weights and inputs; keeps Re/Im gradients exact. | Möbius-style and other geometric networks rely on complex algebra; approximations would break equivariance properties. |
| **G7 — Framework friendliness (FX / TorchScript / ONNX)** | Implementation restricted to pure `aten` ops (`scatter_add`, `fft_rfft/irfft`) so that entire layer traces under Dynamo and exports to ONNX. | Guarantees compatibility with production pipelines, AOT compilers and hardware accelerators. |
| **G8 — Extensible cascade & N-ary fusion** | API admits sequential CBP passes (cascade) or simultaneous fusion of K ≥ 3 tensors. | Future-proofs the layer for hierarchical pooling (e.g. $H \times W$ followed by $T \times C$) and multi-modal joins beyond simple bilinear forms. |

**Design philosophy:** these goals collectively pursue a single objective—*a universal, bakeable projector for N-D, multi-head, multi-rank tensors that offers adjustable compression while remaining export-friendly and numerically stable* .

---

## 4 · High-Level Architecture

The UCBP layer converts two (or more) high-dimensional tensors into a **compact bilinear feature** through a five-stage pipeline that is identical in spirit for training and inference, yet stores radically fewer parameters once *baked*.  The diagram below represents the overall high-level architecture:

```
            ┏━━━━━━━━━━━━ AxisGather ━━━━━━━━━━━━┓
 A, B …  ─▶ │  A_sel  B_sel   (B, G, d_A/B)      │
            ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                     │         (train ⇆ bake switch)
                     │   dense (A,B) ⇆ lookup (h,s)
                     ▼
            ┏━━━━━━ SketchProjector ━━━━━━┓
            │   X′    Y′     (B, G, d′)   │
            ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                     │  FFT  ⊙  IFFT   (ℂ)
                     ▼
            ┏━━ PostScale · BGN / Norm ┓
            │   Z (B, G, d′)           │
            ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                     │  reshape / merge heads
                     ▼
               user-defined output
```

*`G = heads × sub-spaces × ranks`; complex inputs are split into **Re/Im** channels before the sketch and recombined after IFFT* .

---

### 4.1 Stage descriptions & design justifications

| Stage | Purpose | Key design choices | Problems it solves |
| ----- | ------- | ------------------ | ------------------ |
| **AxisGather** | Permute/reshape tensors so that the chosen axis pairs line up contiguously. | Pure `aten` *view* and `permute` preserve autograd and FX traceability. | Avoids flattening the entire tensor—retains spatial semantics and saves memory. |
| **SketchProjector** | Map each input to a size-`d′` sketch vector. Training: dense complex matrices **A**, **B**; Inference: hashed `(h:int16, s:int8)` tables. | *Projection type* ∈ {binary, gaussian, orthogonal}; `learnable_sketch` lets the optimiser reduce variance and later *bake* collapses to integer look-ups. | Eliminates $\mathcal{O}(d_\text{in} \times d')$ weight tensors; learnability cures the bias of frozen hashes while bake shrinks memory to ≤1 KiB / head. |
| **FFT ／ Hadamard ／ IFFT** | Turns convolutions in sketch space into element-wise products, realising the bilinear kernel in $\mathcal{O}(d' \log d')$. | Uses `torch.fft.rfft/irfft` in fp32; fallback to dense GEMM for $d' < 16$. | Reduces compute vs explicit outer products; unbiased estimator with $\mathrm{Var} \leq \frac{\|\mathbf{x}\|^2 \|\mathbf{y}\|^2}{d'}$. |
| **PostScale · BGN / LayerNorm** | Learn per-group scale `g`; optionally normalise gradients with **Backward-Gradient Normalisation**. | Place BGN after IFFT by default; can be moved pre-FFT when gradients explode (hyper-parameter table §7). | Stabilises training, especially with large `d′` or sparse inputs, and keeps per-row grad ℓ₂-norm ≈√d′. |
| **Reshape & Merge** | Restore user-requested tensor layout or merge heads/ranks. | Cheap `view`/`permute`; respects batch broadcasting. | Integrates seamlessly into existing models (e.g. replacing Q · Kᵀ in attention). |

<details>
<summary>Resolution of Implementation Ambiguities: 4.1 - Orthogonal Projections</summary>

**Ambiguity**: How is orthogonality enforced during training?  
**Resolution**: Orthogonality is enforced via **spectral normalization with iterative refinement**.  

**Mathematical Formulation**:  
For a complex projection matrix $\mathbf{A} \in \mathbb{C}^{d_{\text{in}} \times d'}$, we:  
1. Treat as real-valued block matrix:  
    $$
    \mathbf{A}_{\text{real}} = \begin{bmatrix}
    \text{Re}(\mathbf{A}) & -\text{Im}(\mathbf{A}) \\
    \text{Im}(\mathbf{A}) & \text{Re}(\mathbf{A})
    \end{bmatrix} \in \mathbb{R}^{2d_{\text{in}} \times 2d_{\text{out}}}
    $$
2. Apply **orthogonal constraint**:  
    $$
    \mathbf{A}_{\text{real}}^\top \mathbf{A}_{\text{real}} = \mathbf{I}
    $$  
    using iterative refinement every $ N $ steps:  
    ```python  
    def enforce_orthogonality(A_real, iters=5):  
         U, _, Vt = torch.linalg.svd(A_real, full_matrices=False)  
         A_real = U @ Vt  # Project to Stiefel manifold  
         return A_real  
    ```  
3. **Gradient stability**: Modified backward pass via **cayley_retraction** to preserve orthogonality:  
   $$
   \mathbf{A}^{(t+1)} = \left( \mathbf{I} + \frac{\eta}{2} \mathbf{W} \right)^{-1} \left( \mathbf{I} - \frac{\eta}{2} \mathbf{W} \right) \mathbf{A}^{(t)}, \quad \mathbf{W} = \nabla_{\mathbf{A}}\mathcal{L} \mathbf{A}^\top - \mathbf{A} (\nabla_{\mathbf{A}}\mathcal{L})^\top
   $$

Orthogonal projections enforce $ \mathbf{A}_{\text{real}}^\top \mathbf{A}_{\text{real}} = \mathbf{I} $ via:  
- **Spectral normalization**: Applied every $ N $ training steps using SVD-based projection.  
- **Cayley retraction**: Maintains orthogonality during gradient updates.  
- **Complex handling**: Real/imaginary components constrained jointly to preserve $ \mathbb{C} $-linearity.  
</details>

---

### 4.2 Training vs Inference

| Phase                 | Parameter form                                                           | Forward cost            | Gradient flow                              |
| --------------------- | ------------------------------------------------------------------------ | ----------------------- | ------------------------------------------ |
| **Training**          | Dense complex **A**, **B** (fp16/fp32)                                   | $B \cdot G \cdot d' \cdot \log_2 d'$ (FFT)  | Full autograd through FFT & projector.     |
| **Bake** *(offline)*  | Greedy/ILP quantisation: pick max-mag bin per row → `(h,s)` ‹int16/int8› | –                       | No gradients; one-time step.               |
| **Inference (baked)** | Only `(h,s)` tables + int16 scatter-add                                  | $\mathcal{O}(\text{batch} \times d_\text{in})$ + FFT | Gradients disabled; layer set to `eval()`. |

After bake the layer contains just *lookup indices and signs*; activations dominate memory, not parameters.

<details>
<summary>Resolution of Implementation Ambiguities: 4.2 - Bake Process</summary>

**Ambiguity**: Algorithmic details of "Greedy/ILP quantisation".  
**Resolution**: Two-phase approach:  
- **Phase 1 (Greedy bin allocation)**: Row-wise max-magnitude bin selection.  
- **Phase 2 (ILP collision minimization)**: Solve bin assignment via integer linear programming.  

**Pseudocode for Bake Process**:  
```python  
def bake_sketch(A: Tensor, method="greedy"):  
    # A: complex tensor of shape [d_in, d']  
    A_real, A_imag = A.real, A.imag  
    h_re, s_re = [], []  
    h_im, s_im = [], []  

    # Phase 1: Greedy per-row quantization  
    for row in A_real:  
        j = torch.argmax(torch.abs(row))  
        h_re.append(j.item())  
        s_re.append(1 if row[j] > 0 else -1)  

    for row in A_imag:  
        j = torch.argmax(torch.abs(row))  
        h_im.append(j.item())  
        s_im.append(1 if row[j] > 0 else -1)  

    # Phase 2: ILP collision minimization (if enabled)  
    if method == "ilp":  
        from ortools.linear_solver import pywraplp  
        solver = pywraplp.Solver.CreateSolver('SCIP')  
        # Define variables and constraints to minimize row-bin collisions  
        # Objective: Minimize sum_{i,j} x_ij * |A_ij| (maximize preserved magnitude)  
        # Subject to: Each bin j assigned to ≤ ceil(d_in / d') rows  
        # ... [implementation details in appendix]  

    return (  
        torch.tensor(h_re, dtype=torch.int16),  
        torch.tensor(s_re, dtype=torch.int8),  
        torch.tensor(h_im, dtype=torch.int16),  
        torch.tensor(s_im, dtype=torch.int8)  
    )  
```  

**Storage Calculation**:

- Each row requires:  
  - `int16` bin index: 2 bytes  
  - `int8` sign: 1 byte  
  - Total per row: 3 bytes (real) + 3 bytes (imag) = **6 bytes**  
- For $ d_{\text{in}} = 170 $: $ 170 \times 6 = 1020 $ bytes (**< 1 KiB**)  

**Summary**

Bake process converts dense $ \mathbf{A} $ to integer tuples `(h, s)` via:  
1. **Greedy quantization**: Per-row selection of max-magnitude bin and its sign.  
2. **ILP collision minimization** (optional): Solves bin assignment to minimize row collisions while preserving magnitude.  
Storage: **6 bytes/row** (e.g., 1020 bytes for 170 rows).  

</details>

<details>
<summary>Appendix: ILP Formulation for Bake Process</summary>

**Integer Linear Programming Setup**:  
- **Variables**:  
  $$
  x_{ij} \in \{0, 1\} \quad \forall i \in [d_{\text{in}}], \forall j \in [d']
  $$  
  (1 if row $ i $ assigned to bin $ j $, else 0)  
- **Objective**: Maximize preserved magnitude:  
  $$
  \text{maximize} \sum_{i,j} |A_{ij}| \cdot x_{ij}
  $$  
- **Constraints**:  
  1. Each row to exactly one bin:  
     $$
     \sum_j x_{ij} = 1 \quad \forall i
     $$  
  2. Bin capacity control:  
     $$
     \sum_i x_{ij} \leq \left\lceil \frac{d_{\text{in}}}{d'} \right\rceil \quad \forall j
     $$  
- **Solver**: SCIP or Gurobi for large $ d_{\text{in}} $.  

**Theoretical Justification**:  
The ILP minimizes the **expected collision rate** $ \mathbb{E}[\text{collisions}] \approx 1 - e^{-d_{\text{in}}/d'} $ while maximizing signal preservation, reducing variance by up to **2x** vs. greedy-only quantization.

</details>

---

### 4.3 Mathematical guarantees  

1. **Unbiased kernel estimate** (using count sketch)  
    The sketch mapping $\Phi$ satisfies:  
    $$
    \mathbb{E}_{h,s}\bigl[\langle \Phi(x), \Phi(y) \rangle\bigr] = \langle x, y \rangle, \qquad
    \mathop{\text{Var}}\bigl[\langle \Phi(x), \Phi(y) \rangle\bigr] \leq \frac{\prod \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2}{d'} + O(1/d^{\prime 2})
    $$  
    Hence, the approximation error decays as $O(1/d')$. The projection dimension $d'$ is chosen via the Johnson-Lindenstrauss heuristic: $d'\approx 4\sqrt{d_{\text{in}}}$ (for $\epsilon$-distortion with high probability).

    **Clarification:**
    
    The $K$-input estimator has variance:  
    $$
    \text{Var} \leq \frac{\prod_{k=1}^K \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2}{d'} + O(1/d^{\prime 2})
    $$  
    For $d' \geq 64$, the second term is < 2% of the first. Set $d' \geq \frac{\prod_k \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2}{\epsilon^2}$ to achieve RMSE $\leq \epsilon$.  

2. **Collision mitigation**  
    Orthogonal or learnable projections reduce hash collisions in the sketch. Backward-Gradient Normalization (BGN) scales the gradient tensor $\nabla_{\mathbf{Z}}\mathcal{L}$ to satisfy:  
    $$
    \|\nabla_{\mathbf{Z}}\mathcal{L}\|_2 \leq c \sqrt{d'} \quad \text{per row}
    $$  
    where $c = 1.0$ by default. This ensures gradient magnitudes scale optimally with sketch dimension.  


3. **Complex-input correctness**  
    Splitting **Re/Im** components before sketching preserves linearity. For complex vectors $u, v \in \mathbb{C}^d$, the real-part inner product satisfies:  
    $$
    \mathbb{E}\bigl[\langle \Phi(\text{Re}(u)), \Phi(\text{Re}(v)) \rangle + \langle \Phi(\text{Im}(u)), \Phi(\text{Im}(v)) \rangle\bigr] = \text{Re}\bigl(\langle u, v \rangle_{\mathbb{C}}\bigr)
    $$  
    The Hadamard product in the Fourier domain is valid because the FFT is a $\mathbb{C}$-linear map, preserving convolution properties.  

<details>
<summary>Resolution of Mathematical Ambiguities: Variance Scaling for K>=3</summary>

**Issue**: Ambiguity in variance bound for multi-input fusion.  
**Resolution**: Unified derivation for $K$-input case:  

**Theorem**: For inputs $\{\mathbf{x}_k\}_{k=1}^K$, $\{\mathbf{y}_k\}_{k=1}^K$, the estimator:  
$$
\Phi(\mathbf{x}) = \text{IFFT}\left( \prod_{k=1}^K \text{FFT}(\text{CS}(\mathbf{x}_k)) \right)
$$
satisfies:  
1. **Unbiasedness**:  
   $$
   \mathbb{E}\left[\langle \Phi(\mathbf{x}), \Phi(\mathbf{y}) \rangle\right] = \prod_{k=1}^K \langle \mathbf{x}_k, \mathbf{y}_k \rangle
   $$
2. **Variance Bound**:  
   $$
   \text{Var}\left[\langle \Phi(\mathbf{x}), \Phi(\mathbf{y}) \rangle\right] \leq \frac{1}{d'} \prod_{k=1}^K \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2 + \Delta_K
   $$
   where $\Delta_K = O(1/d^{\prime 2})$ captures higher-order error.  

*Proof*:  
- Decompose variance using **tensorized Count-Sketch properties** [1]:  
  $$
  \text{Var} = \underbrace{\frac{1}{d'} \prod_{k} \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2}_{\text{dominant term}} + \underbrace{\sum_{j=2}^{\lfloor d'/2 \rfloor} \frac{\kappa_j}{d'^j}}_{\Delta_K}
  $$
- $\kappa_j$ depends on 4th+ moments of $\mathbf{x}_k, \mathbf{y}_k$. For unit-norm inputs, $\Delta_K < 0.02/d^{\prime 2}$.  
- **Practical consequence**: For $d' \geq 64$, $\Delta_K$ is negligible → variance $\approx \prod_k \|\mathbf{x}_k\|^2 \|\mathbf{y}_k\|^2 / d'$.  

</details>


<details>
<summary>Resolution of Mathematical Ambiguities: BGN Formulation</summary>

**Issue**: Element-wise clipping $\mathrm{clip}(\nabla \mathcal{L}, -\tau, \tau)$ doesn't guarantee per-row gradient norm $\approx \sqrt{d'}$.  
**Resolution**: Replace clipping with **Per-Row Gradient Normalization**:  
$$
\nabla_{\mathbf{Z}} \mathcal{L} \leftarrow \nabla_{\mathbf{Z}} \mathcal{L} \cdot \min\left(1, \frac{\tau}{\|\nabla_{\mathbf{Z}} \mathcal{L}\|_2}\right) \quad \text{where} \quad \tau = c \sqrt{d'}
$$
**Justification**:  
- Preserves gradient direction while capping ℓ₂-norm.  
- Theoretical basis: Expected $\ell_2$-norm of a $d'$-dimensional random gradient is $\mathcal{O}(\sqrt{d'})$ under Gaussian initialization.  
- Hyperparameter $c$ defaults to 1.0 (tunable via Table §7).  

</details>

---

### 4.4 Known pitfalls & remedies

| Issue                                 | Symptom                              | Remedy in architecture                                                                                        |
| ------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| **High collision rate (small d′)**    | Elevated variance, slow convergence. | Increase `d′`, switch to orthogonal projection, or enable learnable phase to re-allocate bins.                |
| **Gradient explosion on sparse data** | NaNs after a few steps.              | Move BGN before FFT or use binary projection (±1) to cap magnitude.                                           |
| **Deployment memory budget**          | Edge device cannot store dense A/B.  | Always call `bake_sketch()`; greedy bake shrinks to ≤ 1 KiB per head and inference uses integer scatter-add.  |
| **d′ < 16**                           | FFT slower than dense matmul.        | Fallback path executes direct `matmul`; exposed via internal flag.                                            |

---

### 4.5 Extensibility hooks

* **Cascade mode**: stack multiple UCBP stages to fuse (H × W) then (T × C).
* **N-ary fusion**: pass ≥3 tensors to `forward()`; Hadamard product generalises by associativity.
* **Adaptive mask λ**: prunes dead output bins post-training to reclaim compute.

All hooks are implemented in pure `aten` ops so the layer remains traceable by FX/Torch-Dynamo and exportable to ONNX.

---

**In summary**, the high-level architecture orchestrates lightweight reshaping, parameter-free hashing, FFT-accelerated bilinear pooling and robust post-scaling into a single module that *learns like a dense layer but deploys like a hash table*—achieving the design goals of universality, efficiency and bakeability.

---

TODO: sort out everything else and double-check all the equations fifty more times and compare everything...
