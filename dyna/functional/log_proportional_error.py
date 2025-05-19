import torch

class LogProportionalError(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        g = torch.stack([x, y], dim=0)
        g_min = g.min()
        g_shifted = g - g_min + torch.e
        log_x = g_shifted[0].log()
        log_y = g_shifted[1].log()
        err = ((log_x - log_y) / log_y).abs()

        ctx.save_for_backward(log_x, log_y, g_shifted)
        return err

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> list[torch.Tensor]:
        (log_x, log_y, g_shifted) = ctx.saved_tensors
        
        u = log_x - log_y
        v = log_y
        sign = torch.sign(u / v)

        # dx
        d_logx_dx = 1 / g_shifted[0]
        d_err_dx = sign / v * d_logx_dx

        # dy
        d_logy_dy = 1 / g_shifted[1]
        d_err_dy = sign * ((-1 / v + u / v**2) * d_logy_dy)

        return grad_output * d_err_dx, grad_output * d_err_dy


def log_proportional_error(
        x: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
    r"""
    Computes the log-proportional error:
        err = |log(x_shifted) - log(y_shifted)| / |log(y_shifted)|

    Where:
        x_shifted = x - min(x, y) + e

    This loss is useful when relative errors are important but needs to be smoothed to avoid gradient explosion near zero.
    """
    reduction_types = ["mean", "sum", "none"]

    assert reduction in reduction_types, f"reduction must be one of {reduction_types=}"

    err = LogProportionalError.apply(x, y)

    if reduction == 'none':
        return err
    elif reduction == 'mean':
        return err.mean()
    elif reduction == 'sum':
        return err.sum()
    else:
        raise ValueError(f"Unknown reduction mode passed: {reduction}")
