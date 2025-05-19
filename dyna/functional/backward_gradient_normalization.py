import torch

from typing import Optional

class BackwardGradientNormalization(torch.autograd.Function):
    eps: float = 1.0e-6

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        eps: Optional[float] = None,
    ) -> torch.Tensor:
        eps = torch.tensor([eps if eps is not None else BackwardGradientNormalization.eps], dtype=x.dtype, device=x.device)
        ctx.save_for_backward(eps)
        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> list[torch.Tensor]:
        (eps,) = ctx.saved_tensors
        norm_dims = tuple(range(1, grad_output.dim()))
        grad_norm = grad_output.norm(p=2, dim=norm_dims, keepdim=True)
        grad_norm = grad_norm + eps
        scaling_factor = grad_output[0].numel() ** 0.5
        grad_output = (grad_output / grad_norm) * scaling_factor
        return grad_output, None


def backward_gradient_normalization(
        x: torch.Tensor, 
        eps: Optional[float] = None,
    ) -> torch.Tensor:
    return BackwardGradientNormalization.apply(x, eps)
