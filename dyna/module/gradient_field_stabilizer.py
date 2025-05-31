import torch

from dyna.functional import siglog, backward_gradient_normalization


class GradientFieldStabilizer(torch.nn.Module):
    """
    GradientFieldStabilizer
    ------------------------

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
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.eps = eps
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = backward_gradient_normalization(x)
        x = torch.sigmoid(x) * siglog(x)
        x = x * x.abs().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = backward_gradient_normalization(x)
        return x
