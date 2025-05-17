import torch

def noisein(
    x: torch.Tensor,
    rate: float,
) -> torch.Tensor:
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
