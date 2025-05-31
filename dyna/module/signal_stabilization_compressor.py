import torch
import math

from dyna.functional import siglog, backward_gradient_normalization


class SignalStabilizationCompressor(torch.nn.Module):
    """
    SignalStabilizationCompressor
    -----------------------------

    A dynamic vector-field transformation module designed to:
    - preserve directional expressiveness
    - suppress uncontrolled amplitude growth
    - stabilize gradients through repeated normalization

    Core mechanisms:
    - Sigmoid-based masking to softly suppress large activations
    - Custom nonlinear transformation (siglog) to reshape signal distribution
    - Repeated backward gradient normalization (BGN) to ensure uniform update dynamics
    - Soft amplitude normalization via inverse root-mean-square scaling

    Inputs:
    - x: Tensor of shape [..., vector_dim], where the last dimension(s) represent the feature vector

    Returns:
    - Tensor of the same shape, dynamically regularized and gradient-stabilized
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
