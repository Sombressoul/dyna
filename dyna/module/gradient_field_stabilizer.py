import torch

from dyna.functional import siglog, backward_gradient_normalization


class GradientFieldStabilizer(torch.nn.Module):
    """
    GradientFieldStabilizer
    ------------------------
    A dynamic vector-field transformation module designed to:
    - preserve directional expressiveness
    - suppress amplitude explosion
    - stabilize gradients during backpropagation

    Core mechanisms:
    - sigmoid pre-squashing to limit magnitude escalation
    - custom nonlinear warping (siglog)
    - repeated backward gradient normalization (BGN)
    - soft amplitude balancing via inverse RMS

    Inputs:
    - x: Tensor of arbitrary shape

    Returns:
    - Regularized tensor of same shape
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = backward_gradient_normalization(x)
        x = torch.sigmoid(x) * siglog(x)
        x = backward_gradient_normalization(x)
        x = x * x.abs().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = backward_gradient_normalization(x)
        return x
