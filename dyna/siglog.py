import torch


class SigLog(torch.autograd.Function):
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
        )
        dx = 1.0 / (x_abs**x_pow + 1)
        return grad_output * dx


def siglog(x: torch.Tensor) -> torch.Tensor:
    return SigLog.apply(x)
