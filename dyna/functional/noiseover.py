import torch


def noiseover(
    x: torch.Tensor,
    rate: float,
) -> torch.Tensor:
    """
    Adds global additive noise to the entire tensor.

    The function samples Gaussian noise from a normal distribution
    matching the mean and standard deviation of `x`, then scales and
    adds it proportionally to all elements of the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to perturb.
    rate : float
        Relative noise intensity. If zero or negative, input is returned unchanged.

    Returns
    -------
    torch.Tensor
        Tensor with globally added Gaussian noise.

    Notes
    -----
    - Noise has the same shape, dtype, and device as the input.
    - The resulting noise follows Normal(x.mean(), x.std()).
    - For `rate<=0.0`, the function is a no-op.
    """    
    if rate <= 0.0:
        return x

    x_noise = torch.nn.init.normal_(
        tensor=torch.empty_like(x),
        mean=x.mean().item(),
        std=x.std().item(),
    )
    x = x + (x_noise * rate)

    return x
