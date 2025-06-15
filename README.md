**Important note**

This repository contains the work‑in‑progress research and implementation of the
biologically inspired signal propagation framework **DyNA** (Dynamic Neural Architecture).

---

## Philosophy

DyNA explores alternatives to conventional sequence modelling with the guiding
principle:

> *Any linear or non‑linear transformation should be derived from the data
> semantics itself.*

The library focuses on dynamic weight generation, stable signal compression and
gradient control techniques.

---

## Project structure

```
docs/             # Various documentation about DyNA project
dyna/
├── functional/   # Stand‑alone differentiable functions
├── lib/          # Low‑level building blocks
└── module/       # High‑level neural network layers
```

---

## Implemented components

### `dyna.functional`

- **backward_gradient_normalization** – normalizes gradients during the backward
  pass to prevent exploding or vanishing updates. Implements the method
  described in the paper “Backward Gradient Normalization in Deep Neural
  Networks” by Alejandro Cabana and Luis F. Lago‑Fernández
  ([arXiv:2106.09475](https://arxiv.org/abs/2106.09475)).
- **log_proportional_error** – computes a logarithmic error term with custom
  gradients, useful near zero.
- **noisein** / **noiseover** – injects element-wise or global noise for regularization.
- **siglog** and **siglog_parametric** – signed logarithmic mappings with custom
  gradient shaping.

### `dyna.lib`

- **TensorComposerDelta** – generates banks of 2‑D weights via rank‑weighted
  modulation and diversity penalties.
- **TensorComposerMobius** – constructs spatial filters using Mobius‑like
  complex transformations and learned projections.

### `dyna.module`

- **DynamicConv2DDelta** – convolutional layer that draws its kernels from
  `TensorComposerDelta` conditioned on context vectors.
- **DynamicConv2DMobius** – convolutional layer based on
  `TensorComposerMobius` with optional dynamic bias, padding and offsets.
- **SignalStabilizationCompressor** – non‑linear block combining gating,
  logarithmic compression and inverse RMS scaling for stable activations.

---

## Documentation

### Holographic Projection Memory (HPM)

All HPM-related documentation has been moved to: https://github.com/sombressoul/CogniRay

---

### TensorComposerMobius (TCM)

**[TensorComposerMobius (TCM)](/docs/TCM/)** is a dynamic composition module that generates high-dimensional tensor structures through complex-phase transformations over learnable subspaces. By leveraging Möbius-like modulation, spectral routing, and adaptive rank mixing, TCM supports self-modifying representations and nonlinear composition - enabling compositional generalization, runtime reconfiguration, and modular control in DyNA-style cognitive architectures.

|            | Theory     | Implementation  | Verified  | Test code |
| ---------- | ---------- | --------------- | --------- | --------- |
| **Status** | TODO       | COMPLETED       | YES       | YES       |

---

### Compact Spectral Multiplier (CSM)

**[Compact Spectral Multiplier (CSM)](/docs/Compact-Spectral-Multiplier-(CSM)/Theoretical-Foundations-of-the-CSM.md)** is a randomized kernel estimator that approximates high-dimensional multilinear interactions via CountSketch and FFT. It enables fast, low-memory inner product estimation between structured tensors while preserving unbiasedness. CSM is particularly effective in highly correlated regimes.

|            | Theory     | Implementation  | Verified  | Test code |
| ---------- | ---------- | --------------- | --------- | --------- |
| **Status** | COMPLETED  | COMPLETED       | YES       | YES       |
