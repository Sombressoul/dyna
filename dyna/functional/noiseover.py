import torch

def noiseover(
    x: torch.Tensor,
    rate: float,
) -> torch.Tensor:
    if rate <= 0.0:
        return x

    x_noise = torch.nn.init.normal_(
        tensor=torch.empty_like(x),
        mean=x.mean().item(),
        std=x.std().item(),
    )
    x = x + (x_noise * rate)

    return x

