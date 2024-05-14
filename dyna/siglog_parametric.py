import torch

from typing import Optional


class SigLogParametric(torch.autograd.Function):
    alpha: float = 0.25
    smoothing: float = 0.1
    smooth_grad: bool = False

    @staticmethod
    def _real_forward(
        x: torch.Tensor,
        mod: torch.Tensor,
    ) -> torch.Tensor:
        x_sign = torch.where(x > 0.0, +1.0, -1.0).to(
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.log(x.abs() + mod)
        x = (x - torch.log(mod)) * x_sign
        return x

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        alpha: float = None,
        smoothing: float = None,
        smooth_grad: bool = None,
    ) -> torch.Tensor:
        alpha = alpha if alpha is not None else SigLogParametric.alpha
        alpha = torch.tensor(alpha).to(dtype=input.dtype, device=input.device)
        mod = torch.e * alpha

        assert mod > 0, f"alpha cannot be <=0. Got: {alpha}"

        smoothing = smoothing if smoothing is not None else SigLogParametric.smoothing
        smoothing = torch.tensor(smoothing).to(dtype=input.dtype, device=input.device)

        smooth_grad = (
            smooth_grad if smooth_grad is not None else SigLogParametric.smooth_grad
        )
        smooth_grad = torch.tensor(smooth_grad).to(dtype=torch.bool, device=input.device)

        ctx.save_for_backward(input, mod, smoothing, smooth_grad)

        with torch.no_grad():
            x = SigLogParametric._real_forward(input, mod)

        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        (input, mod, smoothing, smooth_grad) = ctx.saved_tensors

        with torch.no_grad():
            if smooth_grad.item():
                # Smooth symmetric derivative:
                # f'(x) = (f(x+h) - f(x-h))/2h
                x_1 = SigLogParametric._real_forward(input + smoothing, mod)
                x_2 = SigLogParametric._real_forward(input - smoothing, mod)
                dx = (x_1 - x_2) / (2 * smoothing)
            else:
                # Exact derivative:
                # f'(x) = 1.0 / (|x| + e*alpha)
                dx = 1.0 / (input.abs() + mod)

        return grad_output * dx, None, None, None


def siglog_parametric(
    x: torch.Tensor,
    alpha: Optional[float] = SigLogParametric.alpha,
    smoothing: Optional[float] = SigLogParametric.smoothing,
    smooth_grad: Optional[bool] = SigLogParametric.smooth_grad,
) -> torch.Tensor:
    return SigLogParametric.apply(x, alpha, smoothing, smooth_grad)
