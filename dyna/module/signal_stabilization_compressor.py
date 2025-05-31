import torch
import math

from dyna.functional import siglog, backward_gradient_normalization


class SignalStabilizationCompressor(torch.nn.Module):
    """
    SignalStabilizationCompressor
    -----------------------------

    A nonlinear signal regularization module designed to:
    - suppress uncontrolled amplitude growth
    - reshape and compress signal distribution
    - preserve directional expressiveness
    - stabilize gradients through repeated normalization

    Core mechanisms:
    - Sigmoid-based gating to softly attenuate extreme values
    - Logarithmic remapping via custom `siglog` transformation
    - Recurrent backward gradient normalization (BGN) to maintain uniform sensitivity
    - Inverse RMS-based scaling to compress and normalize signal energy

    This module is suitable as a preprocessing or intermediate layer in deep architectures,
    particularly when input signals or features tend to exhibit heavy-tailed or unstable distributions.

    Args:
        leak (float): Small coefficient for residual passthrough of original signal (default: 1e-3).
        eps (float): Epsilon to prevent division by zero and stabilize inverse operations (default: 1e-12).

    Input:
        x (Tensor): Input tensor of shape [..., D], where the last dimension represents feature vectors.

    Returns:
        Tensor: Transformed tensor of the same shape, with regularized amplitude and stabilized gradients.
    """

    def __init__(
        self,
        leak: float = 1.0e-3,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.leak = leak
        self.eps = eps
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        precision_pass = x.dtype in [torch.bfloat16, torch.float32]
        eps = self.eps if precision_pass else torch.finfo(x.dtype).smallest_normal
        leak = self.leak if precision_pass else math.sqrt(eps)

        x = backward_gradient_normalization(x) # Normalizes backward flow to prevent unstable gradient propagation.
        x_leak = x * leak # Injects a small residual of the original signal to prevent vanishing.
        x_a = torch.sigmoid(x) + x_leak
        x_b = siglog(x) + x_leak
        x = x_a * x_b
        x = backward_gradient_normalization(x) # Prevents unstable gradient scaling introduced by subsequent RMS normalization.
        x = x * x.abs().mean(dim=-1, keepdim=True).add(eps).rsqrt()
        x = backward_gradient_normalization(x) # On backward: stabilizes input ranges for differentiation.
        return x
