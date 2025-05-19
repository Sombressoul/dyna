import torch


class MeanProportionalError(torch.autograd.Function):
    # Unstable on small diffs and near-zero values. Should be used only w/ gradient clipping.

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        y: torch.Tensor,
        eps: torch.Tensor = torch.tensor(1.0e-4),
    ) -> torch.Tensor:
        x_shape = torch.tensor([*x.shape], dtype=torch.int32, device=x.device)
        y_shape = torch.tensor([*y.shape], dtype=torch.int32, device=y.device)
        ctx.save_for_backward(x, x_shape, y, y_shape, eps)
        x, y = x.flatten(0).unsqueeze(0), y.flatten(0).unsqueeze(0)
        g_min = torch.cat([x, y], dim=0).min() - 1.0
        x, y = x - g_min, y - g_min
        err = ((x - y) / y).abs().mean()
        return err

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        (x, x_shape, y, y_shape, eps) = ctx.saved_tensors
        x, y = x.flatten(), y.flatten()

        # # Exact solution leads to grad explosion on small differences or near-zero values:
        # N = x.numel()
        # sign = torch.sign((x - y) / (y + eps))
        # dx = sign / (y + eps) / N
        # dy = -sign * x / (y + eps)**2 / N

        # Some kind of workaround...
        N = x.numel()
        diff = x - y
        safe_sign = diff / torch.sqrt(diff**2 + eps)
        dx = safe_sign / (y + eps) / N
        dy = -safe_sign * x / ((y + eps)**2 + eps) / N

        sign = lambda sign_x: torch.where(sign_x > 0.0, +1.0, -1.0).to(
            dtype=sign_x.dtype,
            device=sign_x.device,
        )
        dx = (torch.log(dx.abs() + torch.e + eps) - 1.0) * sign(dx)
        dy = (torch.log(dy.abs() + torch.e + eps) - 1.0) * sign(dy)

        dx = dx.reshape(x_shape.tolist())
        dy = dy.reshape(y_shape.tolist())
        
        return grad_output * dx, grad_output * dy

def mean_proportional_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return MeanProportionalError.apply(x, y)
