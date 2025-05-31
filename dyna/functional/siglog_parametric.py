import torch

from typing import Optional, Union


class SigLogParametric(torch.autograd.Function):
    """
    Autograd implementation of the parametric sigmoid-logarithmic transform.

    This function implements a signed logarithmic mapping:
        f(x) = sign(x) * (log(|x| + mod) - log(mod)),
    where mod = e * alpha.

    The class supports two gradient computation modes:
    - Exact gradient for input x: df/dx = 1 / (|x| + mod)
    - Finite-difference approximation for df/dalpha (always),
      and optionally for df/dx when `smooth_grad=True`.

    Class Attributes
    ----------------
    alpha : float
        Default curvature parameter (1 / e).
    smoothing : float
        Default delta value for numerical differentiation (1e-3).
    smooth_grad : bool
        Whether to use finite-difference gradient estimation.
    """    
    alpha: float = 1.0 / torch.e
    smoothing: float = 0.001
    smooth_grad: bool = False

    @staticmethod
    def _real_forward(
        x: torch.Tensor,
        mod: torch.Tensor,
    ) -> torch.Tensor:
        """
        Core transformation: computes signed log-scaled activation.

        This is the actual mathematical function:
            f(x) = sign(x) * (log(|x| + mod) - log(mod))

        It is used in both forward and backward passes — including
        finite-difference gradient approximations — to ensure that
        all transformations remain consistent and differentiable.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        mod : torch.Tensor
            Additive offset inside the logarithm. Usually `e * alpha`.
        """
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
        alpha: Union[float, torch.Tensor] = None,
        smoothing: float = None,
        smooth_grad: bool = None,
    ) -> torch.Tensor:
        alpha = alpha if alpha is not None else SigLogParametric.alpha
        alpha = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)
        alpha = alpha.to(dtype=input.dtype, device=input.device)
        mod = torch.e * alpha

        assert mod > 0, f"alpha cannot be <=0. Got: {alpha.item()}"

        smoothing = smoothing if smoothing is not None else SigLogParametric.smoothing
        smoothing = torch.tensor(smoothing).to(dtype=input.dtype, device=input.device)

        smooth_grad = (
            smooth_grad if smooth_grad is not None else SigLogParametric.smooth_grad
        )
        smooth_grad = torch.tensor(smooth_grad).to(
            dtype=torch.bool, device=input.device
        )

        ctx.save_for_backward(input, alpha, smoothing, smooth_grad)

        with torch.no_grad():
            x = SigLogParametric._real_forward(input, mod)

        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        (input, alpha, smoothing, smooth_grad) = ctx.saved_tensors

        get_mod = lambda alpha, smoothing=0.0: torch.e * (alpha + smoothing)

        with torch.no_grad():
            if smooth_grad.item():
                # Smooth symmetric derivative for both: x and alpha.
                # f'(x) = (f(x+h) - f(x-h))/2h
                x_1 = SigLogParametric._real_forward(input + smoothing, get_mod(alpha))
                x_2 = SigLogParametric._real_forward(input - smoothing, get_mod(alpha))
                dx = (x_1 - x_2) / (2 * smoothing)
                mod_1 = SigLogParametric._real_forward(
                    input,
                    get_mod(alpha, +(alpha * smoothing)),
                )
                mod_2 = SigLogParametric._real_forward(
                    input,
                    get_mod(alpha, -(alpha * smoothing)),
                )
                dmod = (mod_1 - mod_2) / (2 * smoothing)
            else:
                # Exact derivative for x:
                # f'(x) = 1.0 / (|x| + e*alpha)
                dx = 1.0 / (input.abs() + get_mod(alpha))
                # Smooth symmetric derivative for alpha.
                mod_1 = SigLogParametric._real_forward(
                    input,
                    get_mod(alpha, +(alpha * SigLogParametric.smoothing)),
                )
                mod_2 = SigLogParametric._real_forward(
                    input,
                    get_mod(alpha, -(alpha * SigLogParametric.smoothing)),
                )
                dmod = (mod_1 - mod_2) / (2 * SigLogParametric.smoothing)

        return grad_output * dx, grad_output * dmod, None, None


def siglog_parametric(
    x: torch.Tensor,
    alpha: Optional[Union[float, torch.Tensor]] = SigLogParametric.alpha,
    smoothing: Optional[float] = SigLogParametric.smoothing,
    smooth_grad: Optional[bool] = SigLogParametric.smooth_grad,
) -> torch.Tensor:
    """
    Parametric sigmoid-logarithmic transformation with optional gradient smoothing.

    Applies a scaled logarithmic mapping to the input tensor:
        y = sign(x) * (log(|x| + mod) - log(mod)),
    where `mod = e * alpha`.

    This transformation behaves similarly to a logarithmic squashing function,
    with tunable curvature controlled by `alpha`. It supports smooth symmetric
    numerical gradient estimation via central differences when `smooth_grad=True`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to transform.
    alpha : float or torch.Tensor, optional
        Curvature scale factor. Controls sensitivity near zero.
        Larger alpha makes the function flatter; smaller alpha sharpens it.
        Default is 1 / e.
    smoothing : float, optional
        Small positive constant used as delta in finite difference
        approximations for smooth gradients. Ignored if `smooth_grad=False`.
        Default is 1e-3.
    smooth_grad : bool, optional
        Whether to enable smooth symmetric gradient estimation (for both x and alpha).
        If False, uses exact gradient for x and smooth approximation only for alpha.
        Default is False.

    Returns
    -------
    torch.Tensor
        Transformed tensor with the same shape as `x`.

    Notes
    -----
    - The function is strictly monotonic and zero-centered.
    - The transformation is safe around x=0 due to added modulus.
    - Gradients are well-behaved and controllable, even in low-precision setups.
    - If `alpha <= 0`, the function will raise a runtime error.

    See Also
    --------
    SigLogParametric : Internal autograd implementation.
    """    
    return SigLogParametric.apply(x, alpha, smoothing, smooth_grad)
