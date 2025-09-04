import torch

from dyna.lib.cpsf.errors import NumericalError


def hermitianize(
    Sigma: torch.Tensor,
) -> torch.Tensor:
    return (
        0.5 * (Sigma + Sigma.mH)
        if torch.is_complex(Sigma)
        else 0.5 * (Sigma + Sigma.transpose(-2, -1))
    )


def cholesky_spd(
    Sigma: torch.Tensor,
) -> torch.Tensor:
    if Sigma.dim() < 2 or Sigma.shape[-1] != Sigma.shape[-2]:
        raise NumericalError(
            f"cholesky_spd: expected [..., n, n], got {tuple(Sigma.shape)}"
        )

    Sigma_h = hermitianize(Sigma)

    if not torch.isfinite(Sigma_h).all():
        raise NumericalError("Cholesky: non-finite entries in input")

    try:
        return torch.linalg.cholesky(Sigma_h)
    except RuntimeError as e:
        finfo = (
            torch.finfo(Sigma_h.real.dtype)
            if torch.is_complex(Sigma_h)
            else torch.finfo(Sigma_h.dtype)
        )
        eps = torch.sqrt(finfo.eps)
        scale = (
            Sigma_h.diagonal(dim1=-2, dim2=-1)
            .abs()
            .mean(dim=-1, keepdim=True)
            .clamp(min=1.0)
        )
        eye = torch.eye(
            Sigma_h.shape[-1],
            device=Sigma_h.device,
            dtype=Sigma_h.real.dtype if torch.is_complex(Sigma_h) else Sigma_h.dtype,
        )
        Sigma_h_jittered = Sigma_h + (
            eps
            * scale[..., None]
            * (eye if not torch.is_complex(Sigma_h) else eye.type_as(Sigma_h))
        )
        try:
            return torch.linalg.cholesky(Sigma_h_jittered)
        except RuntimeError as e2:
            raise NumericalError(f"Cholesky failed even with jitter: {e2}") from e2


def tri_solve_norm_sq(
    L: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    if L.dim() < 2 or L.shape[-1] != L.shape[-2]:
        raise NumericalError(
            f"tri_solve_norm_sq: expected L as [..., n, n], got {tuple(L.shape)}"
        )

    rhs = w.unsqueeze(-1) if w.dim() == L.dim() - 1 else w
    rhs = rhs.to(dtype=L.dtype)
    n = L.shape[-1]

    if not (torch.isfinite(L).all() and torch.isfinite(rhs).all()):
        raise NumericalError("non-finite in inputs")

    if rhs.shape[-2] != n:
        raise NumericalError(
            f"tri_solve_norm_sq: mismatched shapes: L:[...,{n},{n}] vs w:[...,{rhs.shape[-2]},*]"
        )

    if hasattr(torch.linalg, "solve_triangular"):
        y = torch.linalg.solve_triangular(L, rhs, upper=False, left=True)
    else:
        y = torch.triangular_solve(rhs, L, upper=False).solution

    if torch.is_complex(y):
        norm_sq = (y.conj() * y).real.sum(dim=-2)
    else:
        norm_sq = (y * y).sum(dim=-2)

    norm_sq = torch.clamp(norm_sq, min=0)

    return norm_sq.squeeze(-1) if w.dim() == L.dim() - 1 else norm_sq
