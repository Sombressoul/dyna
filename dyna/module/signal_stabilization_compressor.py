import torch

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
        x = backward_gradient_normalization(x) # Normalizes grad for further backward computations.
        x_leak = x * self.leak # Small leak of shrinked original signal.
        x_a = torch.sigmoid(x) + x_leak
        x_b = siglog(x) + x_leak
        x = x_a * x_b
        x = backward_gradient_normalization(x) # Prevents grad explosions after deriving 1/(2 * sqrt(x))
        x = x * x.abs().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = backward_gradient_normalization(x) # On backward: stabilizes input ranges for differentiation.
        return x
