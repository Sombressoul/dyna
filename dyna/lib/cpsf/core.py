import torch

from typing import Optional

from dyna.lib.cpsf.errors import NumericalError
from dyna.lib.cpsf.context import CPSFContext
from dyna.lib.cpsf.functional.core_math import (
    # CPSF core math
    delta_vec_d,
    iota,
    lift,
    q,
    rho,
    R,
    R_ext,
    Sigma,
    # Math helpers
    hermitianize,
)


class CPSFCore:
    def __init__(
        self,
        context: CPSFContext,
    ):
        self.ctx = context

    def _cholesky_spd(
        self,
        S: torch.Tensor,
        eps: Optional[float],
    ) -> torch.Tensor:
        if S.dim() < 2 or S.shape[-1] != S.shape[-2]:
            raise NumericalError(
                f"_cholesky_spd: expected [..., n, n], got {tuple(S.shape)}"
            )

        S_h = hermitianize(A=S)

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
            eps = torch.sqrt(finfo.eps) if eps is None else eps
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

    def R(
        self,
        vec_d: torch.Tensor,
    ) -> torch.Tensor:
        return R(
            vec_d=vec_d,
        )

    def R_ext(
        self,
        R: torch.Tensor,
    ) -> torch.Tensor:
        return R_ext(
            R=R,
        )

    def Sigma(
        self,
        R_ext: torch.Tensor,
        sigma_par: torch.Tensor,
        sigma_perp: torch.Tensor,
    ) -> torch.Tensor:
        return Sigma(
            R_ext=R_ext,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
        )

    def q(
        self,
        w: torch.Tensor,
        R_ext: torch.Tensor,
        sigma_par: torch.Tensor,
        sigma_perp: torch.Tensor,
    ) -> torch.Tensor:
        return q(
            w=w,
            R_ext=R_ext,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
        )

    def delta_vec_d(
        self,
        vec_d: torch.Tensor,
        vec_d_j: torch.Tensor,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        return delta_vec_d(
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            eps=eps,
        )

    def iota(
        self,
        delta_z: torch.Tensor,
        delta_vec_d: torch.Tensor,
    ) -> torch.Tensor:
        return iota(
            delta_z=delta_z,
            delta_vec_d=delta_vec_d,
        )

    def lift(self, z: torch.Tensor) -> torch.Tensor:
        return lift(
            z=z,
        )

    def rho(
        self,
        q: torch.Tensor,
    ) -> torch.Tensor:
        return rho(
            q=q,
            q_max=self.ctx.exp_clip_q_max,
        )

    def resolvent_delta_T_hat(
        self,
        alpha_j: torch.Tensor,
        v_j: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        eps: float = 1.0e-6,
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
            denom = a + eps
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
        eps_t = torch.as_tensor(eps, dtype=dtype, device=device)
        K = a_mm * I + eps_t * A.to(dtype=dtype, device=device)
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
