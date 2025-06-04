# Universal Compact Bilinear Projection (UCBP)

*Technical Design Document  ·  Version 0.9a (June 2025)*

---

## 1 · Purpose

UCBP is a **drop‑in, parameter‑efficient replacement for explicit bilinear or multi‑linear weight tensors**. It approximates the interaction of two arbitrary tensors via **Count‑Sketch + FFT**, supports complex numbers, multi‑head/multi‑rank groupings, cascade operation, and a *bake* step that collapses large train‑time projectors into hash tables for inference.

Typical use‑cases:

* Dynamic weight generators (e.g. `WeightsLib2DMobius`)
* Multi‑modal attention (vision × text × audio)
* Graph/relational scoring
* Parameter‑efficient fine‑tuning inside LLM blocks

---

## 2 · Design Goals

| ID | Goal                                | Rationale                                                         |
| -- | ----------------------------------- | ----------------------------------------------------------------- |
| G1 | **Arbitrary shapes & axis pairing** | Any pair(s) of axes between tensors *A* and *B* may be projected. |
| G2 | **Multi‑head, multi‑rank groups**   | Thousands of independent “heads” without weight duplication.      |
| G3 | **Trainable → Baked switch**        | Dense projectors during training, KiB‑sized hashes for inference. |
| G4 | **Complex‑number support**          | Required by Möbius‑style models; gradients must stay correct.     |
| G5 | **TorchScript/FX traceability**     | Layer must compile under Dynamo / ONNX export.                    |
| G6 | **Extensible (cascade / N‑ary)**    | Future tasks may need sequential or N‑ary CBP.                    |

---

## 3 · High‑Level Architecture

```text
╭──────────── AxisGather ─────────────╮
│  A_sel  B_sel  (B, G, d_A/B)        │
╰─────────────────────────────────────╯
               │
        train: (P_x , P_y) ──┐
      infer : (h,s,g) lookup │
               ▼             │
      ╭─── CountSketchProjector ───╮
      │  X'   Y'   (B, G, d')      │
      ╰────────────────────────────╯
               │  FFT/IFFT (ℂ)
               ▼
        element‑wise product (⊙)
               │
      ╭──── PostScale / Norm ───╮
      │   Z (B, G, d')          │
      ╰─────────────────────────╯
               │ reshape to user
               ▼
        output tensor (user‑defined)
```

* **Group dimension** `G = T × S × R`  (heads, sub‑spaces, ranks).
* `d'` = sketch dimension (typ. 256 – 2048).

---

## 4 · Public API (PyTorch)

```python
class UCBP(nn.Module):
    def __init__(
        self,
        shape_A: tuple[int, ...],
        shape_B: tuple[int, ...],
        axes: list[tuple[int, int]],
        group_shape: tuple[int, ...] | int = 1,   # (T, S, R) or int
        output_dim: int = 512,                    # d'
        complex_weights: bool = False,
        trainable_sketch: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        ...

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Return CBP result with the declared batch/group layout."""

    @torch.no_grad()
    def bake_sketch(self, method: str = "greedy", iters: int = 2) -> None:
        """Quantise *P_x/P_y* → (h,s,g) and switch to inference mode."""

    def export_state(self) -> dict[str, torch.Tensor]: ...
    def import_state(self, state: dict[str, torch.Tensor]) -> None: ...
```

All arguments obey PyTorch broadcasting; batch dims are preserved.

---

## 5 · Internal Components

| Component                | Key fields                                 | Train‑time              | Inference‑time                        |
| ------------------------ | ------------------------------------------ | ----------------------- | ------------------------------------- |
| **AxisGather**           | `axes_A`, `axes_B`                         | `permute`, `reshape`    | identical                             |
| **CountSketchProjector** | `weight_x (2d_A×d')`, `weight_y (2d_B×d')` | dense **cfloat** params | buffers `(h,s,g)` (`int16/int8/fp16`) |
| **FFTCore**              | –                                          | `torch.fft.fft/ifft`    | same                                  |
| **PostScale**            | `scale (G,d')`                             | optional LayerNorm      | same                                  |

---

## 6 · Algorithms

### 6.1 Training Forward

```text
1  Gather A_sel, B_sel → (B,G,d_A/B)
2  If complex:
       X = concat(Re(A_sel), Im(A_sel))   # (B, 2·d_A)
       Y = concat(Re(B_sel), Im(B_sel))
   else:
       X = A_sel ; Y = B_sel
3  X' = X @ weight_x   # Count‑Sketch
   Y' = Y @ weight_y
4  Z  = IFFT( FFT(X') ⊙ FFT(Y') )
5  out = Z * scale_g   (+ optional LayerNorm)
6  Reshape to user‑defined order.
```

### 6.2 Bake Step – Greedy Quantisation

```python
mag = weight.abs()               # (rows, d')
h   = mag.argmax(dim=1)          # (rows,)
s   = weight[torch.arange(rows), h].real.sign().to(torch.int8)
# Optionally redistribute residual and iterate.
```

### 6.3 Inference Path (per group)

```python
out = torch.zeros(B, d', device=X.device)
idx = h.expand(B, -1)
sgn = s.float().expand_as(X2)
out.scatter_add_(1, idx, sgn * X2)
# repeat for Y, then FFT → ⊙ → IFFT → *g
```

`scatter_add_` is pure integer indexing, O(batch × d\_in).

---

## 7 · Advanced Features

| Feature                  | How to use                                                  |
| ------------------------ | ----------------------------------------------------------- |
| **Cascade mode**         | `cascade=[[(a1,b1)], [(a2,b2)]]` → two CBP passes.          |
| **N‑ary CBP**            | Planned: iterative pairwise merge of ≥3 tensors.            |
| **Very‑low‑rank LoRA**   | `output_dim ≤ 64` → baked size < 1 KiB/head.                |
| **Auto dtype‑cast**      | Inputs promoted to `dtype`; FFT runs in `fp32` if required. |
| **FX/TorchScript ready** | Only `aten` ops (`scatter_add`, `fft_fft`, `fft_ifft`).     |

---

## 8 · Performance & Memory \*

* **Train mode** – memory ∝ `G·d_in·d'`; keep `d' ≤ 512` to fit on A100 80G with `G≈1k`.
* **Baked mode** – per head ≈ `(2·d_in)·3 bytes + d'·2 bytes`.
* FFT cost: `2·B·G·d'·log₂(d')`  ≪ GEMM cost for large `d_in`.

*\* - Presumably. Real measurements are required.*

---

## 9 · Error Handling

| Condition                          | Behaviour                   |
| ---------------------------------- | --------------------------- |
| Axis mismatch                      | `ValueError` in `__init__`. |
| Unsupported dtype (`int`, `bool`)  | auto cast to `float32`.     |
| `bake_sketch()` when already baked | emits warning, no‑op.       |

---

## 10 · Testing Strategy

1. **Shape fuzzing** – random tensors & axis pairs.
2. **Grad‑check** – compare autograd with finite diff.
3. **Equivalence** – UCBP vs explicit bilinear on toy dims.
4. **Bake regression** – MSE before/after bake ≤ ε.
5. **FX compile** – ensure `torch.export` & ONNX pass.

---

## 11 · Implementation Roadmap

| Milestone | Deliverable                                    | ETA      |
| --------- | ---------------------------------------------- | -------- |
| M1        | Minimal CBP core (`trainable_sketch=True`)     | +1 week  |
| M2        | Complex support, `group_dim`, cascade          | +2 weeks |
| M3        | Bake algorithm + state export                  | +3 weeks |
| M4        | TorchScript / ONNX compliance, docs & examples | +4 weeks |
| M5        | Integrate into `WeightsLib2DMobius`            | +5 weeks |

---

## 12 · Dependencies

* **PyTorch ≥ 2.3** (`torch.fft` API)
* **NumPy** (offline tools)
* *(Optional)* `einops` for readable reshapes (kept off hot‑path)

---

## 13 · Known Limitations

* No gradients in baked mode – layer must be in `eval()`.
* FFT on `d' < 16` can be slower than direct dot‑products.
* Greedy bake may leave \~5 % MSE; ILP optimiser TBD.

---

## 14 · Algorithmic Sketches

### 14.1 Training Pass (single group)

```python
# X, Y : (B, dA) and (B, dB) after AxisGather
# W_x, W_y : learnable ℂ matrices (2·dA × d'), (2·dB × d')
# g : learnable scale (d',)

X2 = torch.cat([X.real, X.imag], dim=-1)  # (B, 2·dA)
Y2 = torch.cat([Y.real, Y.imag], dim=-1)

X_s = X2 @ W_x               # (B, d')
Y_s = Y2 @ W_y

Z_f = torch.fft.fft(X_s) * torch.fft.fft(Y_s)
Z   = torch.fft.ifft(Z_f).real
out = Z * g                  # (+ LayerNorm if wanted)
```

### 14.2 Bake Step (vectorised)

```python
rows, d_out = W.shape
mag = W.abs()
h   = mag.argmax(dim=1)                       # (rows,)
s   = W[torch.arange(rows), h].real.sign().to(torch.int8)
```

### 14.3 Inference Scatter‑Add

```python
out = torch.zeros(B, d', device=X.device)
idx = h.expand(B, -1)
sgn = s.float().expand_as(X2)
out.scatter_add_(1, idx, sgn * X2)
```

---

## 15 · Quick Examples

### 15.1 Single‑Head, Latent × Channel

```python
layer = UCBP(
    shape_A=(B, T, H, W, D),
    shape_B=(B, T, H, W, D),
    axes=[(4, 4)],            # D ↔ D
    group_shape=1,
    output_dim=512,
    complex_weights=True,
)
Z = layer(A, B)               # → (B, 512)
```

### 15.2 Replacing Q‑K dot in Transformer

```python
cbp = UCBP(
    shape_A=(B, Heads, S, D),
    shape_B=(B, Heads, S, D),
    axes=[(3, 3)],            # channel dim
    group_shape=Heads,
    output_dim=64,
)
attn_logits = cbp(Q, K) / math.sqrt(D)
```

---

## 16 · Implementation‑Time Use Cases

| ID | Scenario                             | Design impact                                      |
| -- | ------------------------------------ | -------------------------------------------------- |
| U1 | **Video spatio‑temporal fusion**     | Requires cascade API (H×W then T×C).               |
| U2 | **Knowledge‑graph relation scoring** | Emphasise bake(); int8 buffers; tiny edge models.  |
| U3 | **LoRA‑style fine‑tuning**           | Support `output_dim ≤ 32`; FFT fallback.           |
| U4 | **Cross‑modal VQA**                  | Efficient large G; ensure CUDA scatter‑add scales. |
| U5 | **Complex‑valued geometric nets**    | Maintain correct Re/Im gradients.                  |

---

## 17 · Licensing

Released under **MIT** licence (compatible with PyTorch).
Algorithm credit: *Fukui et al., “Multimodal Compact Bilinear Pooling”, ECCV 2016 (arXiv: ************************[1606.01847](https://arxiv.org/abs/1606.01847)************************)*

---

## 18 · References

```
[1] A. Fukui, D. H. Park, D. Yang, A. Rohrbach, T. Darrell, and M. Rohrbach, “Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding,” *European Conference on Computer Vision (ECCV)*, 2016, pp. 19–36. (arXiv: [1606.01847](https://arxiv.org/abs/1606.01847))

[2] M. Charikar, K. Chen, and M. Farach‑Colton, “Finding frequent items in data streams,” *Proceedings of the 29th International Colloquium on Automata, Languages and Programming (ICALP)*, 2002, pp. 693–703. (Introduces the Count‑Sketch algorithm.)

[3] N. Pham and R. Pagh, “Fast and Scalable Polynomial Kernels via Explicit Feature Maps,” *Proceedings of the 19th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)*, 2013, pp. 239–247. (arXiv: [1307.2977](https://arxiv.org/abs/1307.2977))

[4] I. V. Oseledets, “Tensor‑Train Decomposition,” *SIAM Journal on Scientific Computing*, vol. 33, no. 5, 2011, pp. 2295–2317. (arXiv: [0908.0052](https://arxiv.org/abs/0908.0052))
```

---


*Prepared for integration into* **WeightsLib2DMobius**. *Feedback & PRs welcome!*
