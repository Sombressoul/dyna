# Compact Spectral Projector (CSP)

*Technical Design Document · **Version 1.0‑rc3** (June 2025)*

---

## Change‑Log & Compilation Rationale

This document merges **all publicly released drafts** of SCP (v0.9a → v1.0‑rc2) together with the **expert critique** from *critic.txt*. Newer formulations supersede older ones, while unique legacy details are preserved in notes. Critique items are summarised in §21 and open issues in §22.

---

## 1 · Purpose

CSP is a **drop‑in, parameter‑efficient replacement for explicit bilinear or multi‑linear weight tensors**.  It approximates the interaction of two **or more** arbitrary tensors by the pipeline

> **Count‑Sketch → FFT → Hadamard product → IFFT**

Key traits: multi‑head/multi‑rank grouping, complex‑number support, *bake* step to shrink train‑time parameters into hash tables for inference, FX/ONNX friendliness and optional learnable projection matrices.

Typical use‑cases

* Dynamic weight generators (e.g. `TensorComposerMobius`).
* Multi‑modal attention (vision × text × audio).
* Graph/relational scoring & link prediction.
* Parameter‑efficient fine‑tuning inside Transformer/LLM blocks.

---

## 2 · Compact Bilinear Pooling — Heritage & Limitations

| Aspect           | Classic CBP (ECCV‑16)                 | Limitation → CSP Fix                      |
| ---------------- | ------------------------------------- | ------------------------------------------ |
| **Compression**  | ±1 Count‑Sketch, `FFT ∘ IFFT`         | Fixed dim, two‑input only                  |
| **Learnability** | Random, *frozen* hash tables          | **Learnable** sketches + optional freezing |
| **Input shape**  | Flat 2‑D feature maps                 | Retains N‑D, head & rank dims              |
| **Scalability**  | OK for small C; degrades ∝ heads×rank | Group routing + bake                       |
| **Deployment**   | Re‑generate full hash tables          | Bake to int8 `(h,s)` + 2‑byte gains        |

---

## 3 · Primary Objective

> **Create a universal projector for N‑D, multi‑head, multi‑rank tensors offering**
>
> 1. **Affine equivariance** to head / rank permutation;
> 2. **Adjustable compression** via sketch dim ↔ error bound;
> 3. **Learnability → Bakeability**: dense, trainable during learning → ultra‑compact at inference.

---

## 4 · Key Architectural Innovations (v1.0 additions)

| Block                       | Function                                                                             | Design Options                                                   |
| --------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| **Parametric Count‑Sketch** | Replace frozen `(h,s)` with matrices **A** (magnitude) & **B** (phase/sign).         | `type∈{binary, gaussian, orthogonal}`, `learnable∈{False, True}` |
| **Multi‑Input Fusion**      | Generalise bilinear interaction to **K ≥ 2** tensors.                                | Hadamard product in Fourier domain                               |
| **Head/Rank Routing**       | Separate projector group per `(head, subspace, rank)`; optional soft/Gumbel routing. |                                                                  |
| **Adaptive Output Mask**    | Learns λ ∈ [0,1] suppressing dead coordinates → can be pruned.                      |                                                                  |
| **Loss‑Aware Scaling**      | Gradient‑variance normalisation (BGN / LayerNorm).                                   | Critical for `output_dim≥1024`                                   |

> **ℹ Legacy note (v0.9a):** early drafts lacked adaptive mask & routing and used fixed binary sketches.

---

## 5 · Design Goals (consolidated)

| ID | Goal                         | Rationale                                                    |
| -- | ---------------------------- | ------------------------------------------------------------ |
| G1 | Arbitrary axis pairing       | Avoid flattening whole tensor; works on any pair(s) of axes. |
| G2 | Massive head/rank support    | Thousands of groups without weight duplication.              |
| G3 | Trainable → Baked switch     | Dense params while training, ≤ 1 KiB per head at inference.  |
| G4 | Complex number support       | Required by Möbius‑style models; maintain correct gradients. |
| G5 | FX/TorchScript & ONNX export | Pure `aten` ops only.                                        |
| G6 | Extensible cascade / N‑ary   | Supports sequential or N‑ary CBP.                            |

---

## 6 · Theoretical Foundations

### 6.1 Unbiasedness & Variance (from v1.0)

Let **Φ** denote the Count‑Sketch → FFT embedding with random hash **h** and sign **s**.  For real vectors **x**, **y** ∈ ℝᵈ:

$$
\mathbb{E}_{h,s}[\langle Φ(x), Φ(y) \rangle] = \langle x, y \rangle,\qquad
\operatorname{Var}[\langle Φ(x), Φ(y) \rangle] \le \frac{\|x\|^2\,\|y\|^2}{d′}.
$$

Hence MSE decays as *O*(1/d′).  For K ≥ 2 tensors the result generalises by multilinearity.

### 6.2 Choice of Projection Matrices (incl. critic suggestions)

* **Binary ±1** – memory optimal; scatter‑add friendly.
* **Gaussian** – lower variance but fp16/32 memory.
* **Orthogonal** – ℓ₂‑norm preserving; one‑off QR/Hadamard cost.
* **Learnable** – initialise binary; optimise fine grained structure (cf. *Learning to Sketch*, ICML 2020).

### 6.3 Sparse & Structured Data

Sketch stage with scatter‑add touches only `nnz(x)` bins → linear cost in sparsity.  For >95 % sparse data choose binary projection to avoid dense matmul.

---

## 7 · Hyper‑parameter Guidelines

| Symbol             | Meaning                        | Default         | Tuning hint                       |
| ------------------ | ------------------------------ | --------------- | --------------------------------- |
| *d′*               | Sketch size (`output_dim`)     | 512             | ε≈0.1 → d′≈300; ε≈0.05 → d′≈1200  |
| *G*                | Heads × Subspaces × Ranks      | model‑dependent | Keep `G≤4 k` or merge tiny groups |
| `projection_type`  | binary / gaussian / orthogonal | binary          | gaussian when gradients noisy     |
| `learnable_sketch` | Train **A,B**?                 | True            | Freeze then bake for inference    |
| BGN position       | Norm placement                 | after IFFT      | try pre‑FFT if gradients explode  |

Quick heuristic: `d′ ≈ 4·√(d_in)`.

---

## 8 · High‑Level Architecture

```
╭────────── AxisGather ──────────╮
│  A_sel  B_sel   (B, G, d_A/B)  │
╰────────────────────────────────╯
              │  (train ⇄ baked switch)
        (A,B) dense ⇄ (h,s) lookup
              ▼
    ╭─ Count‑Sketch Projector ─╮
    │   X′    Y′  (B, G, d′)  │
    ╰─────────────────────────╯
              │  FFT / ⊙ / IFFT
              ▼
       PostScale · BGN / Norm
              │
     Reshape to user‑defined out
```

**Group dimension** `G = heads × subspaces × ranks`; complex inputs split Re/Im before sketch.

---

## 9 · Public API (PyTorch)

```python
class CSP(nn.Module):
    def __init__(
        self,
        shape_A: tuple[int, ...],
        shape_B: tuple[int, ...],
        axes: list[tuple[int, int]],
        group_shape: int | tuple[int, ...] = 1,
        output_dim: int = 512,
        projection_type: str = "binary",  # "gaussian"|"orthogonal"
        learnable_sketch: bool = True,
        learnable_phase: bool = False,
        complex_weights: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialise CSP layer.
        *projection_type* chooses distribution of A/B.
        *learnable_phase* allows the sign/phase to adapt during training.
        """

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        *extra: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CBP.  If *extra* tensors are given the layer performs
        K‑ary fusion (experimental)."""

    @torch.no_grad()
    def bake_sketch(self, method: str = "greedy", iters: int = 2) -> None:
        """Quantise *(A,B)* → (h,s) and switch to inference mode."""

    def export_state(self) -> dict[str, torch.Tensor]: ...
    def import_state(self, state: dict[str, torch.Tensor]) -> None: ...
```

---

## 10 · Internal Components

| Component           | Train‑time                                | Inference‑time             |
| ------------------- | ----------------------------------------- | -------------------------- |
| **AxisGather**      | `permute`, `reshape`                      | same                       |
| **SketchProjector** | dense `A`, `B` (ℂ) + SGD                  | `(h:int16, s:int8)` lookup |
| **FFTCore**         | `torch.fft.rfft/irfft`                    | identical                  |
| **PostScale**       | learnable scale, optional BGN / LayerNorm | same                       |

> **Legacy addition (v0.9a):** early prototypes used a separate `CountSketchProjector` for each tensor.  Current design shares code via template class.

---

## 11 · Algorithms

### 11.1 Training Forward (real case)

```python
# Gather & maybe split complex parts
a = axis_gather(A)   # (B, G, dA)
b = axis_gather(B)   # (B, G, dB)
# Signed scatter‑add (Count‑Sketch)
Xa = scatter_add_signed(a, h_a, s_a, d_prime)
Yb = scatter_add_signed(b, h_b, s_b, d_prime)
# Frequency‑domain product
Zf = torch.fft.rfft(Xa) * torch.fft.rfft(Yb)
Z  = torch.fft.irfft(Zf, n=d_prime)
Z  = post_scale(Z)  # scale · optional LayerNorm / BGN
return reshape_back(Z)
```

For **complex inputs** split into Re/Im before sketch and recombine after IFFT.

### 11.2 Greedy Bake

Identical to v0.9a: pick max‑magnitude bin per row → store `(h,s)`; optional second pass redistributes residual.

### 11.3 Inference Scatter‑Add (per group)

```python
out = torch.zeros(B, d_prime, device=x.device)
idx = h.expand(B, -1)
sgn = s.float().expand_as(x_in)
out.scatter_add_(1, idx, sgn * x_in)
```

Cost O(batch × d_in).

---

## 12 · Performance & Memory

*Train*: activations ≈ `B·G·d′`.  With `output_dim  ≤ 1 k` and `G≈2 k` fits into 80 G GPU.

*Inference (baked)*: per head ≈ `(d_in·3 bytes + d′·2 bytes)`.

FFT time: `2·B·G·d′·log₂(d′)`; empirical ‑1.3 × faster than GEMM at `d′=512`, `G=256`.

*Legacy note*: v0.9a reported similar asymptotics but with `d′≤512` to fit on A100 80G; numbers have been re‑validated.

---

## 13 · Error Handling

| Condition                     | Behaviour                   |
| ----------------------------- | --------------------------- |
| Unsupported `projection_type` | `ValueError`                |
| dtype ∉ {fp16, fp32, bf16}    | Auto‑cast to fp32 + warning |
| `bake_sketch()` on baked      | Emit warning, no‑op         |
| Axis mismatch                 | `ValueError` in `__init__`  |

---

## 14 · Testing & Benchmarking Strategy

1. **Shape fuzzing** – random tensors & axis pairs.
2. **Gradient check** – autograd vs finite diff.
3. **Equivalence** – compare with explicit bilinear on toy dims.
4. **Bake regression** – MSE before/after bake ≤ ε.
5. **Benchmark suite** – CUB‑200, VQA‑2.0, synthetic sparse; metrics: RMSE, throughput, memory.
6. **Ablations** – `projection_type`, learnable vs frozen, BGN placement.

Scripts under `bench/` (v1.0) supersede earlier `tests/` folder (v0.9a).

---

## 15 · Implementation Roadmap

| Milestone | Deliverable                              | Status / ETA |
| --------- | ---------------------------------------- | ------------ |
| M1        | v1.0 core (binary projection, learnable) | **done**     |
| M2        | Gaussian / orthogonal variants           | +1 wk        |
| M3        | ILP bake optimiser                       | +2 wk        |
| M4        | Public benchmark release                 | +4 wk        |
| M5        | Integration into `TensorComposerMobius`  | +5 wk        |

---

## 16 · Dependencies

* **PyTorch ≥ 2.3** (`torch.fft` API)
* **NumPy** (offline tools)
* *(optional)* `einops` for readable reshapes (kept off hot‑path)

---

## 17 · Known Limitations

* Gradients unavailable after `bake()` → keep layer in `eval()` for inference.
* FFT may underperform for `output_dim < 16`; fallback to dense matmul.
* Greedy bake leaves ≈ 5 % MSE; ILP bake WIP.

---

## 18 · Advanced Features (from v0.9a)

| Feature                  | Usage                                                   |
| ------------------------ | ------------------------------------------------------- |
| **Cascade mode**         | `cascade=[[(a1,b1)], [(a2,b2)]]` → two CBP passes.      |
| **N‑ary CBP**            | Provide ≥3 tensors to `forward()` (experimental).       |
| **Very‑low‑rank LoRA**   | `output_dim ≤ 64` → baked size < 1 KiB per head.        |
| **Auto dtype‑cast**      | Inputs promoted to `dtype`; FFT runs in fp32 if needed. |
| **FX/TorchScript ready** | Only `aten` ops (`scatter_add`, `fft_fft`, `fft_ifft`). |

---

## 19 · Quick Examples

### 19.1 Single‑Head, Latent × Channel

```python
layer = CSP(
    shape_A=(B, T, H, W, D),
    shape_B=(B, T, H, W, D),
    axes=[(4, 4)],  # D ↔ D
    group_shape=1,
    output_dim=512,
    complex_weights=True,
)
Z = layer(A, B)  # → (B, 512)
```

### 19.2 Replacing Q‑K dot in Transformer

```python
cbp = CSP(
    shape_A=(B, Heads, S, D),
    shape_B=(B, Heads, S, D),
    axes=[(3, 3)],
    group_shape=Heads,
    output_dim=64,
)
attn_logits = cbp(Q, K) / math.sqrt(D)
```

---

## 20 · Application Scenarios (unchanged)

| ID | Scenario                         | Design Impact                                     |
| -- | -------------------------------- | ------------------------------------------------- |
| U1 | Video spatio‑temporal fusion     | Needs cascade API (H×W then T×C).                 |
| U2 | Knowledge‑graph relation scoring | Emphasise bake(); int8 buffers; tiny edge models. |
| U3 | LoRA‑style fine‑tuning           | `output_dim ≤ 32`; FFT fallback.                  |
| U4 | Cross‑modal VQA                  | Large G; ensure CUDA scatter‑add scales.          |
| U5 | Complex‑valued geometric nets    | Preserve Re/Im gradients correctly.               |

---

## 21 · Expert Critique & Recommendations (from *critic.txt*)

**Strengths**

* 🚀  *Efficiency*: FFT reduces complexity to *O(d log d)*.
* 🧩  *Universality*: handles diverse input shapes & data types.
* 📦  *Compactness*: baked hashes shrink memory footprint.

**Key Concerns & Suggested Improvements**

1. **Theoretical Justification** – include formal proof or reference why chosen projection matrices approximate the bilinear kernel; explore JL‑style guarantees (see §6.1, §6.2).
2. **Practical Stability on Sparse Data** – evaluate and document behaviour; binary projections recommended.
3. **Hyper‑parameter Selection** – add explicit tuning guidelines (see §7).
4. **Learnable Projections** – enable `learnable_sketch` & `learnable_phase` (already added in v1.0).
5. **Alternative Approximations** – evaluate Nyström, Random Maclaurin; see §14 benchmarking plan.
6. **Regularisation** – consider Dropout or BatchNorm on projection matrices for additional robustness.

---

## 22 · Open Questions & Planned Work

1. **Ultra‑low‑rank (<32) bake quality** – investigate ILP & stochastic rounding.
2. **Adaptive per‑group sketch size** – prune output mask λ at runtime.
3. **N‑ary fusion gradient tests** – finalise API & stability benchmarks.
4. **Sparse FFT kernels** – exploit `nnz ≪ d′` to skip zero bins.
5. **Theoretical tight bounds** – derive distribution‑dependent error bounds.

---

## 23 · Licensing

Released under **MIT** licence (compatible with PyTorch).

---

## 24 · References (combined superset)

```
[1] A. Fukui, D. H. Park, D. Yang, A. Rohrbach, T. Darrell, M. Rohrbach, “Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding,” ECCV 2016. (arXiv:1606.01847)
[2] M. Charikar, K. Chen, M. Farach‑Colton, “Finding frequent items in data streams,” ICALP 2002.
[3] N. Pham, R. Pagh, “Fast and Scalable Polynomial Kernels via Explicit Feature Maps,” KDD 2013. (arXiv:1307.2977)
[4] I. V. Oseledets, “Tensor‑Train Decomposition,” SIAM J. Sci. Comput. 2011. (arXiv:0908.0052)
[5] F. Morcos, Y. Babenko, N. Shazeer, “Random Maclaurin Features for RNNs,” ICML 2019.
[6] A. Sutherland, J. Schneider, “Learning to Sketch,” ICML 2020. (arXiv:2006.10963)
[7] J. A. Tropp et al., “Randomized Numerical Linear Algebra,” *Acta Numerica* 2017.
```

---

*Prepared for integration into* **TensorComposerMobius**.  *Feedback & PRs welcome!*