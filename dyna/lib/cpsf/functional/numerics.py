import torch


def cholesky_spd(
    Sigma: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError


def tri_solve_norm_sq(
    L: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError
