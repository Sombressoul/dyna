import torch


class SigLog(torch.autograd.Function):
    """
    SigLog
    ------

    Autograd-compatible nonlinear function for signal compression with asymmetric gradient shaping.

    Use via `siglog(x)` wrapper function.
    """
    eps: float = 1.0e-4

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(input)
        x_sign = torch.where(input > 0.0, +1.0, -1.0).to(
            dtype=input.dtype,
            device=input.device,
        )
        x = (torch.log(input.abs() + torch.e + SigLog.eps) - 1.0) * x_sign
        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        x_abs = input.abs()
        x_pow = 1.0 + torch.where(x_abs < 1.0, x_abs, 0.0).to(
            dtype=input.dtype,
            device=input.device,
        ).sqrt()
        dx = 1.0 / (x_abs**x_pow + 1)
        return grad_output * dx


def siglog(x: torch.Tensor) -> torch.Tensor:
    """
    siglog(x)
    ---------

    Nonlinear activation function that combines logarithmic compression with sign preservation.
    Smoothly suppresses large magnitudes while preserving small-scale signal variation.

    Forward:
        y = sign(x) * (log(|x| + e + eps) - 1)

    Backward:
        Custom gradient with smooth decay for small values:
        dy/dx = 1 / (|x|^p + 1), where p = 1 + sqrt(|x|) if |x| < 1, else p = 1

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Transformed tensor with same shape as input
    """    
    return SigLog.apply(x)
