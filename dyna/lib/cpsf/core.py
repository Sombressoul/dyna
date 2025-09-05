import torch

from typing import Optional

from dyna.lib.cpsf.functional.numerics import (
    cholesky_spd,
    tri_solve_norm_sq,
)
from dyna.lib.cpsf.context import CPSFContext


class CPSFCore:
    def __init__(
        self,
        context: CPSFContext,
    ):
        self.ctx = context

    def R(
        self,
        d: torch.Tensor,
        eps: float = 1.0e-3,
    ) -> torch.Tensor:
        *B, N = d.shape
        dtype = d.dtype
        device = d.device

        if d.dim() < 1:
            raise ValueError(f"R(d): expected [..., N], got {tuple(d.shape)}")

        E = torch.eye(N, dtype=dtype, device=device).expand(*B, N, N).clone()
        M = E * (1.0 + eps)
        M[..., :, 0] = d
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vh

        return R

    def R_ext(
        self,
        R: torch.Tensor,
    ) -> torch.Tensor:
        *B, N, _ = R.shape
        twoN = 2 * N

        if R.dim() < 2 or R.shape[-1] != R.shape[-2]:
            raise ValueError(f"R_ext(R): expected [..., N, N], got {tuple(R.shape)}")

        R_ext = torch.zeros(*B, twoN, twoN, dtype=R.dtype, device=R.device)

        R_ext[..., :N, :N] = R
        R_ext[..., N:, N:] = R

        return R_ext

    def build_sigma(
        self,
        R_ext: torch.Tensor,
        sigma_par: torch.Tensor,
        sigma_perp: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def cholesky_L(
        self,
        Sigma: torch.Tensor,
    ) -> torch.Tensor:
        return cholesky_spd(Sigma)

    def delta_vec_d(
        self,
        d_q: torch.Tensor,
        d_j: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def lift(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def iota(
        self,
        dz: torch.Tensor,
        delta_d: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def tri_solve_norm_sq(
        self,
        L: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        return tri_solve_norm_sq(L, w)

    def rho_q(
        self,
        q: torch.Tensor,
    ) -> torch.Tensor:
        return torch.exp(-torch.pi * torch.clamp(q, max=self.ctx.exp_clip_q_max))

    def resolvent_delta_T_hat(
        self,
        alpha_j: torch.Tensor,
        v_j: torch.Tensor,
        A: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
