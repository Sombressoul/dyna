**Important note**

This repository contains the work‑in‑progress research and implementation of the
biologically inspired signal propagation framework **DyNA** (Dynamic Neural Architecture).

## Philosophy

DyNA explores alternatives to conventional sequence modelling with the guiding
principle:

> *Any linear or non‑linear transformation should be derived from the data
> semantics itself.*

The library focuses on dynamic weight generation, stable signal compression and
gradient control techniques.

## Project structure

```
dyna/
├── functional/  # Stand‑alone differentiable functions
├── lib/         # Low‑level dynamic weight builders
└── module/      # High‑level neural network layers
```


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

- **WeightsLib2DDelta** – generates banks of 2‑D weights via rank‑weighted
  modulation and diversity penalties.
- **WeightsLib2DMobius** – constructs spatial filters using Mobius‑like
  complex transformations and learned projections.

### `dyna.module`

- **DynamicConv2DDelta** – convolutional layer that draws its kernels from
  `WeightsLib2DDelta` conditioned on context vectors.
- **DynamicConv2DMobius** – convolutional layer based on
  `WeightsLib2DMobius` with optional dynamic bias, padding and offsets.
- **SignalStabilizationCompressor** – non‑linear block combining gating,
  logarithmic compression and inverse RMS scaling for stable activations.
