import torch
import math

from typing import Union
from enum import Enum

import dyna


class SignalStabilizationCompressorMode(Enum):
    GATING = "gating"
    DAMPENING = "dampening"


class SignalStabilizationCompressor(torch.nn.Module):
    """
    SignalStabilizationCompressor
    -----------------------------

    A nonlinear signal conditioning layer designed to reshape, stabilize, and compress feature distributions,
    while preserving directional expressiveness and maintaining controllable gradient flow.

    This module is useful in situations where signal energy, gradient propagation, or value ranges
    tend to become unstable — for example, in deep or dynamic architectures with highly nonlinear blocks.

    Functional highlights:
    - Sigmoid-based gating (signed) or dampening (absolute) for soft control of signal pass-through
    - Logarithmic warping (`siglog`) to flatten high-amplitude ranges
    - Residual leak injection (fixed or trainable via `softplus`) to prevent vanishing dynamics
    - Optional backward gradient normalization at input, mid, and output stages
    - Inverse RMS-based rescaling to ensure bounded amplitude growth

    Modes:
        - "gating": allows signed leak passthrough, asymmetric and selective
        - "dampening": uses absolute leak, enforcing smooth positive stabilization

    Args:
        bgn_input (bool): Whether to apply BGN before any transformation.
        bgn_mid (bool): Whether to apply BGN after nonlinear composition (before RMS).
        bgn_output (bool): Whether to apply BGN after RMS normalization.
        mode (str or SignalStabilizationCompressorMode): Mode for leak injection (gating or dampening).
        trainable (bool): If True, leak scaling becomes a learnable softplus-transformed parameter.
        leak (float): Scalar factor for residual leak injection (default: 1e-3).
        eps (float): Small epsilon added to prevent division-by-zero in inverse RMS (default: 1e-12).

    Input:
        x (Tensor): Input tensor of shape [..., D], where D is the feature dimension.

    Returns:
        Tensor: Regularized and compressed tensor of the same shape.
    """

    def __init__(
        self,
        bgn_input: bool = False,
        bgn_mid: bool = False,
        bgn_output: bool = False,
        mode: Union[SignalStabilizationCompressorMode, str] = "gating",
        trainable: bool = False,
        leak: float = 1.0e-3,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.input_bgn = bgn_input
        self.bgn_mid = bgn_mid
        self.bgn_output = bgn_output
        self.mode = SignalStabilizationCompressorMode(mode)
        self.trainable = trainable
        self.leak = leak
        self.eps = eps

        if self.trainable:
            self.leak_scale = torch.nn.Parameter(
                data=torch.tensor(
                    [1.0],
                    dtype=torch.float32,
                    requires_grad=True,
                ),
            )

        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        precision_pass = x.dtype in [torch.bfloat16, torch.float32]
        eps = self.eps if precision_pass else torch.finfo(x.dtype).smallest_normal
        leak = self.leak if precision_pass else math.sqrt(eps)

        # Normalize backward flow to prevent unstable gradient propagation if needed.
        if self.input_bgn:
            x = dyna.functional.backward_gradient_normalization(x)

        # Inject a small residual of the original signal to prevent vanishing.
        if self.trainable:
            x_leak = x * self.leak * torch.nn.functional.softplus(self.leak_scale.to(x.dtype))
        else:
            x_leak = x * leak

        if self.mode == SignalStabilizationCompressorMode.GATING:
            x_leak_sigmoid = x_leak
        elif self.mode == SignalStabilizationCompressorMode.DAMPENING:
            x_leak_sigmoid = x_leak.abs()
        else:
            raise ValueError(f"Unknown SignalStabilizationCompressorMode: {self.mode}")
        
        x_a = torch.sigmoid(x) + x_leak_sigmoid
        x_b = dyna.functional.siglog(x) + x_leak
        x = x_a * x_b

        # Prevent unstable gradient scaling introduced by subsequent RMS normalization if needed.
        if self.bgn_mid:
            x = dyna.functional.backward_gradient_normalization(x)

        x = x * x.abs().mean(dim=-1, keepdim=True).add(eps).rsqrt()

        # Ensures uniform backward sensitivity after amplitude normalization.
        if self.bgn_output:
            x = dyna.functional.backward_gradient_normalization(x)

        return x
