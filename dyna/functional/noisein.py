import torch


def noisein(
    x: torch.Tensor,
    rate: float,
) -> torch.Tensor:
    """
    Injects noise into randomly selected elements of the input tensor.

    For each sample in the batch, a subset of elements is selected
    and replaced by noisy values constructed as:
        -x[i] + noise(mean=x.mean, std=x.std)

    All other elements remain unchanged. The injection rate controls
    the fraction of perturbed elements.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [B, ...]. The first dimension is treated as batch.
    rate : float
        Fraction of features per sample to be replaced with noise (0.0 to 1.0).

    Returns
    -------
    torch.Tensor
        Tensor with partial injection of noise into randomly chosen positions.

    Notes
    -----
    - Noise is generated independently for each sample.
    - Only a subset of features is affected per instance.
    - For `rate=0.0`, the function is a no-op.
    """    
    if rate <= 0.0:
        return x

    x_shape = x.shape
    x = x.flatten(1)
    x_numel = x.shape[1]
    x_noise_numel = int(x_numel * rate)
    x_noise = torch.nn.init.normal_(
        tensor=torch.empty([x.shape[0], x_noise_numel], device=x.device),
        mean=x.mean().item(),
        std=x.std().item(),
    )
    target_indices = torch.randint(0, x_numel, [x.shape[0], x_noise_numel])
    target_indices = target_indices + torch.arange(0, torch.prod(torch.tensor(x.shape)), x_numel).unsqueeze(1)
    target_indices = target_indices.flatten(0)
    x = x.flatten(0)
    x_negative = torch.zeros_like(x)
    x_negative[target_indices] = (x[target_indices] * -1.0) + x_noise.flatten(0)
    x = x + x_negative
    x = x.reshape([*x_shape])

    return x
