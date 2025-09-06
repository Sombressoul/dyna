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
        if d.dim() < 1:
            raise ValueError(f"R(d): expected [..., N], got {tuple(d.shape)}")

        *B, N = d.shape
        dtype = d.dtype
        device = d.device

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
        if R.dim() < 2 or R.shape[-1] != R.shape[-2]:
            raise ValueError(f"R_ext(R): expected [..., N, N], got {tuple(R.shape)}")

        *B, N, _ = R.shape
        twoN = 2 * N

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
        if R_ext.dim() < 2 or R_ext.shape[-1] != R_ext.shape[-2]:
            raise ValueError(
                f"build_sigma: expected R_ext as [..., 2N, 2N], got {tuple(R_ext.shape)}"
            )

        *B, twoN, _ = R_ext.shape
        if twoN % 2 != 0:
            raise ValueError(f"build_sigma: last dims must be even, got {twoN}")
        N = twoN // 2

        device = R_ext.device
        dt_real = R_ext.real.dtype

        sigma_par = torch.as_tensor(sigma_par, device=device, dtype=dt_real)
        sigma_perp = torch.as_tensor(sigma_perp, device=device, dtype=dt_real)
        par = sigma_par.reshape(*sigma_par.shape, 1) + torch.zeros(
            *B, 1, device=device, dtype=dt_real
        )
        perp = sigma_perp.reshape(*sigma_perp.shape, 1) + torch.zeros(
            *B, 1, device=device, dtype=dt_real
        )

        if not (torch.all(par > 0) and torch.all(perp > 0)):
            raise ValueError("build_sigma: sigma_par and sigma_perp must be positive")

        base_mask = torch.zeros(twoN, dtype=torch.bool, device=device)
        base_mask[0] = True
        base_mask[N] = True
        mask = base_mask.view(*([1] * len(B)), twoN).expand(*B, twoN)

        diag_vals = torch.where(mask, par, perp)

        D = torch.diag_embed(diag_vals).type_as(R_ext)

        R_h = R_ext.mH if torch.is_complex(R_ext) else R_ext.transpose(-2, -1)
        Sigma = R_h @ (D @ R_ext)

        return Sigma

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
        if d_q.shape != d_j.shape or d_q.dim() < 1:
            raise ValueError(
                f"delta_vec_d: expected matching [..., N], got {tuple(d_q.shape)} vs {tuple(d_j.shape)}"
            )

        inner = torch.sum(torch.conj(d_j) * d_q, dim=-1)
        tangent = d_q - inner.unsqueeze(-1) * d_j

        sin_theta = torch.linalg.vector_norm(tangent, dim=-1)

        cos_theta = torch.clamp(torch.abs(inner), max=1.0)

        theta = torch.acos(cos_theta)

        finfo = torch.finfo(sin_theta.dtype)
        eps = torch.sqrt(finfo.eps)
        scale_safe = theta / torch.clamp(sin_theta, min=eps)
        scale = torch.where(sin_theta < eps, torch.ones_like(scale_safe), scale_safe)

        delta = scale.unsqueeze(-1) * tangent
        return delta

    def lift(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if torch.is_complex(z):
            return torch.cat([z.real, z.imag], dim=-1)

        if z.dim() >= 1 and (z.shape[-1] % 2 == 0):
            return z
        raise ValueError(
            f"lift: expected complex [..., N] or real [..., 2N], got shape={tuple(z.shape)}, dtype={z.dtype}"
        )

    def iota(
        self,
        dz: torch.Tensor,
        delta_d: torch.Tensor,
    ) -> torch.Tensor:
        w_z = self.lift(dz)
        w_d = self.lift(delta_d)

        try:
            w = w_z + w_d
        except RuntimeError as e:
            raise ValueError(
                f"iota: broadcasting mismatch for shapes {tuple(w_z.shape)} and {tuple(w_d.shape)}"
            ) from e
        return w

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
