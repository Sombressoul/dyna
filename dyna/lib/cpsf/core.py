import torch

from typing import Optional

from dyna.lib.cpsf.errors import NumericalError
from dyna.lib.cpsf.context import CPSFContext


class CPSFCore:
    def __init__(
        self,
        context: CPSFContext,
    ):
        self.ctx = context

    def _hermitianize(
        self,
        S: torch.Tensor,
    ) -> torch.Tensor:
        return (
            0.5 * (S + S.mH) if torch.is_complex(S) else 0.5 * (S + S.transpose(-2, -1))
        )

    def _cholesky_spd(
        self,
        S: torch.Tensor,
    ) -> torch.Tensor:
        if S.dim() < 2 or S.shape[-1] != S.shape[-2]:
            raise NumericalError(
                f"_cholesky_spd: expected [..., n, n], got {tuple(S.shape)}"
            )

        S_h = self._hermitianize(S)

        if not torch.isfinite(S_h).all():
            raise NumericalError("Cholesky: non-finite entries in input")

        try:
            return torch.linalg.cholesky(S_h)
        except RuntimeError as e:
            finfo = (
                torch.finfo(S_h.real.dtype)
                if torch.is_complex(S_h)
                else torch.finfo(S_h.dtype)
            )
            eps = torch.sqrt(finfo.eps)
            scale = (
                S_h.diagonal(dim1=-2, dim2=-1)
                .abs()
                .mean(dim=-1, keepdim=True)
                .clamp(min=1.0)
            )
            eye = torch.eye(
                S_h.shape[-1],
                device=S_h.device,
                dtype=S_h.real.dtype if torch.is_complex(S_h) else S_h.dtype,
            )
            S_h_jittered = S_h + (
                eps
                * scale[..., None]
                * (eye if not torch.is_complex(S_h) else eye.type_as(S_h))
            )
            try:
                return torch.linalg.cholesky(S_h_jittered)
            except RuntimeError as e2:
                raise NumericalError(f"Cholesky failed even with jitter: {e2}") from e2

    def _tri_solve_norm_sq(
        self,
        L: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        if L.dim() < 2 or L.shape[-1] != L.shape[-2]:
            raise NumericalError(
                f"_tri_solve_norm_sq: expected L as [..., n, n], got {tuple(L.shape)}"
            )

        rhs = w.unsqueeze(-1) if w.dim() == L.dim() - 1 else w
        rhs = rhs.to(dtype=L.dtype)
        n = L.shape[-1]

        if not (torch.isfinite(L).all() and torch.isfinite(rhs).all()):
            raise NumericalError("non-finite in inputs")

        if rhs.shape[-2] != n:
            raise NumericalError(
                f"_tri_solve_norm_sq: mismatched shapes: L:[...,{n},{n}] vs w:[...,{rhs.shape[-2]},*]"
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

        dn = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
        finfo = (
            torch.finfo(d.real.dtype) if torch.is_complex(d) else torch.finfo(d.dtype)
        )
        d_unit = d / torch.clamp(dn, min=torch.sqrt(finfo.eps))

        I = torch.eye(N, dtype=dtype, device=device).expand(*B, N, N)
        M = (1.0 + eps) * I.clone()
        M[..., :, 0] = d_unit
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        R0 = U @ Vh

        P = I - d_unit.unsqueeze(-1) @ torch.conj(d_unit).unsqueeze(-2)
        Bcomp = P @ R0[..., :, 1:]

        if N > 1:
            Qc, _ = torch.linalg.qr(Bcomp, mode="reduced")
            R_out = torch.cat([d_unit.unsqueeze(-1), Qc], dim=-1)
        else:
            R_out = d_unit.unsqueeze(-1)

        return R_out

    def R_ext(
        self,
        R: torch.Tensor,
    ) -> torch.Tensor:
        if R.dim() < 2 or R.shape[-1] != R.shape[-2]:
            raise ValueError(f"R_ext(R): expected [..., N, N], got {tuple(R.shape)}")

        *B, N, _ = R.shape
        Z = torch.zeros(*B, N, N, dtype=R.dtype, device=R.device)

        top = torch.cat([R, Z], dim=-1)
        bottom = torch.cat([Z, R], dim=-1)
        R_ext = torch.cat([top, bottom], dim=-2)

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

    def lift(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(z):
            raise ValueError(
                f"lift: expected complex [..., N], got {z.dtype} {tuple(z.shape)}"
            )
        return z

    def iota(self, dz: torch.Tensor, delta_d: torch.Tensor) -> torch.Tensor:
        u = self.lift(dz)
        v = self.lift(delta_d)
        try:
            return torch.cat([u, v], dim=-1)
        except RuntimeError as e:
            raise ValueError(
                f"iota: concat failed for shapes {tuple(u.shape)} and {tuple(v.shape)}"
            ) from e

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
        dtype = v_j.dtype
        device = v_j.device

        def _unsqueeze_to(alpha: torch.Tensor, target_ndim: int) -> torch.Tensor:
            a = torch.as_tensor(alpha, dtype=dtype, device=device)
            while a.dim() < target_ndim:
                a = a.unsqueeze(-1)
            return a

        if A is None:
            a = _unsqueeze_to(alpha_j, v_j.dim())
            finfo = (
                torch.finfo(v_j.real.dtype)
                if torch.is_complex(v_j)
                else torch.finfo(v_j.dtype)
            )
            eps = torch.sqrt(finfo.eps)
            denom = torch.clamp(a, min=eps)
            return v_j / denom

        if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
            raise ValueError(
                f"resolvent_delta_T_hat: A must be [..., M, M], got {tuple(A.shape)}"
            )
        M = A.shape[-1]
        if v_j.shape[-1] != M:
            raise ValueError(
                f"resolvent_delta_T_hat: trailing dim mismatch: v_j:[..., {v_j.shape[-1]}] vs A:[..., {M}, {M}]"
            )

        I = torch.eye(M, dtype=dtype, device=device).expand(*A.shape[:-2], M, M)
        a_mm = _unsqueeze_to(alpha_j, A.dim())
        K = A.to(dtype=dtype, device=device) + a_mm * I

        L = self._cholesky_spd(K)

        rhs = v_j.unsqueeze(-1)
        if hasattr(torch.linalg, "solve_triangular"):
            y = torch.linalg.solve_triangular(L, rhs, upper=False, left=True)
            Lt = L.mH if torch.is_complex(L) else L.transpose(-2, -1)
            x = torch.linalg.solve_triangular(Lt, y, upper=True, left=True)
        else:
            y = torch.triangular_solve(rhs, L, upper=False).solution
            Lt = L.mH if torch.is_complex(L) else L.transpose(-2, -1)
            x = torch.triangular_solve(y, Lt, upper=True).solution

        return x.squeeze(-1)
