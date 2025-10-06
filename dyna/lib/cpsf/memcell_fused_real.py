import torch
import torch.nn as nn


from dataclasses import dataclass
from typing import Tuple, List, Optional, Union


def T_Zero_Fused_Real(
    *,
    z: torch.Tensor,  # [B, N] (real)
    z_j: torch.Tensor,  # [M, N] (real)
    vec_d_j: torch.Tensor,  # [M, N] (real)
    T_hat_j: torch.Tensor,  # [M, S] (real)
    alpha_j: torch.Tensor,  # [M] (real)
    sigma_par: torch.Tensor,  # [M] (real, >0)
    sigma_perp: torch.Tensor,  # [M] (real, >0)
    eps: float = 1e-6,
    max_q: float = 25.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device, dtype = z.device, z.dtype

    tiny = torch.finfo(sigma_par.dtype).eps
    w_par = (1.0 / (sigma_par.clamp_min(tiny) ** 2)).to(dtype)  # [M]
    w_perp = (1.0 / (sigma_perp.clamp_min(tiny) ** 2)).to(dtype)  # [M]
    w_diff = w_par - w_perp  # [M]

    dz = z.unsqueeze(1) - z_j.unsqueeze(0)  # [B,M,N]
    dz_norm_sq = (dz * dz).sum(dim=-1)  # [B,M]

    d_norm = vec_d_j.norm(dim=-1, keepdim=True)  # [M,1]
    use_proj_mask = (d_norm > eps).squeeze(-1).to(dtype)  # [M]
    b = torch.where(d_norm > eps, vec_d_j / d_norm, torch.zeros_like(vec_d_j))  # [M,N]
    proj = (dz * b.unsqueeze(0)).sum(dim=-1) * use_proj_mask.unsqueeze(0)  # [B,M]

    q_pos_a = w_perp.unsqueeze(0) * dz_norm_sq  # [B,M]
    q_pos_b = w_diff.unsqueeze(0) * (proj * proj)  # [B,M]
    q_pos = q_pos_a + q_pos_b  # [B,M]
    q_cap = torch.tensor(max_q, dtype=dtype, device=device)
    q_pos = torch.minimum(q_pos, q_cap)  # clamp max
    A_pos = torch.exp(-torch.pi * q_pos)  # [B,M]

    gain = alpha_j.unsqueeze(0) * A_pos  # [B,M]
    T = (gain.unsqueeze(-1) * T_hat_j.unsqueeze(0)).sum(dim=1)  # [B,S]

    return T, gain


@dataclass
class CPSFContributionStoreFusedRealPack:
    z_j: torch.Tensor  # [M,N]
    vec_d_j: torch.Tensor  # [M,N]
    T_hat_j: torch.Tensor  # [M,S]
    alpha_j: torch.Tensor  # [M]
    sigma_par: torch.Tensor  # [M]
    sigma_perp: torch.Tensor  # [M]


class CPSFContributionStoreFusedReal(nn.Module):
    def __init__(
        self,
        *,
        N: int,
        M: int,
        S: int,
        init_range_z: Tuple[float, float] = (-1.0e-3, +1.0e-3),
        init_range_vec_d: Tuple[float, float] = (-1.0e-3, +1.0e-3),
        init_range_T: Tuple[float, float] = (-1.0e-3, +1.0e-3),
        init_range_alpha_j: Tuple[float, float] = (0.9, 1.1),
        init_range_sigma_par: Tuple[float, float] = (0.9, 1.5),
        init_range_sigma_perp: Tuple[float, float] = (0.1, 0.8),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.N = int(N)
        self.M = int(M)
        self.S = int(S)
        self.dtype = dtype

        self.z_j = torch.nn.Parameter(
            data=(
                self._init_param(
                    shape=[self.M, self.N],
                    min=init_range_z[0],
                    max=init_range_z[1],
                )
            ),
            requires_grad=True,
        )
        self.vec_d_j = torch.nn.Parameter(
            data=(
                self._init_param(
                    shape=[self.M, self.N],
                    min=init_range_vec_d[0],
                    max=init_range_vec_d[1],
                )
            ),
            requires_grad=True,
        )
        self.T_hat_j = torch.nn.Parameter(
            data=self._init_param(
                shape=[self.M, self.S],
                min=init_range_T[0],
                max=init_range_T[1],
            ),
            requires_grad=True,
        )
        self.T_hat_j_delta = torch.nn.Parameter(
            data=torch.zeros([self.M, self.S], dtype=self.dtype),
            requires_grad=False,
        )
        self.alpha_j = torch.nn.Parameter(
            data=self._init_param(
                shape=[self.M],
                min=init_range_alpha_j[0],
                max=init_range_alpha_j[1],
            ),
            requires_grad=True,
        )
        self.sigma_par = torch.nn.Parameter(
            data=self._init_param(
                shape=[self.M],
                min=init_range_sigma_par[0],
                max=init_range_sigma_par[1],
            ),
            requires_grad=True,
        )
        self.sigma_perp = torch.nn.Parameter(
            data=self._init_param(
                shape=[self.M],
                min=init_range_sigma_perp[0],
                max=init_range_sigma_perp[1],
            ),
            requires_grad=True,
        )

    def _init_param(
        self,
        shape: List[int],
        min: float = -1.0,
        max: float = +1.0,
    ) -> torch.Tensor:
        p = torch.empty(shape, dtype=self.dtype).uniform_(min, max)

        return p

    def read(
        self,
    ) -> CPSFContributionStoreFusedRealPack:
        T_hat_j_eff = self.T_hat_j + self.T_hat_j_delta.detach()

        return CPSFContributionStoreFusedRealPack(
            z_j=self.z_j,
            vec_d_j=self.vec_d_j,
            T_hat_j=T_hat_j_eff,
            alpha_j=self.alpha_j,
            sigma_par=self.sigma_par,
            sigma_perp=self.sigma_perp,
        )

    @torch.no_grad()
    def update(
        self,
        *,
        T_hat_j_delta: torch.Tensor,
    ) -> None:
        dT = T_hat_j_delta.detach()
        self.T_hat_j_delta.add_(dT)

    @torch.no_grad()
    def consolidate(
        self,
    ) -> None:
        self.T_hat_j.add_(self.T_hat_j_delta)
        self.T_hat_j_delta.zero_()


class CPSFMemcellFusedReal(nn.Module):
    def __init__(
        self,
        *,
        N: int,
        M: int,
        S: int,
        init_range_z: Tuple[float, float] = (-1.0e-3, +1.0e-3),
        init_range_vec_d: Tuple[float, float] = (-1.0e-3, +1.0e-3),
        init_range_T: Tuple[float, float] = (-1.0e-3, +1.0e-3),
        init_range_alpha_j: Tuple[float, float] = (0.9, 1.1),
        init_range_sigma_par: Tuple[float, float] = (0.9, 1.5),
        init_range_sigma_perp: Tuple[float, float] = (0.1, 0.8),
        initial_alpha: float = 1.0e-9,
        delta_T_hat_j_cap: float = 1.0,
        max_q: float = 25.0,
        eps: float = 1.0e-6,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.N = int(N)
        self.M = int(M)
        self.S = int(S)
        self.initial_alpha = float(initial_alpha)
        self.delta_T_hat_j_cap = float(delta_T_hat_j_cap)
        self.max_q = float(max_q)
        self.eps = float(eps)
        self.dtype = dtype
        self.store = CPSFContributionStoreFusedReal(
            N=self.N,
            M=self.M,
            S=self.S,
            init_range_z=init_range_z,
            init_range_vec_d=init_range_vec_d,
            init_range_T=init_range_T,
            init_range_alpha_j=init_range_alpha_j,
            init_range_sigma_par=init_range_sigma_par,
            init_range_sigma_perp=init_range_sigma_perp,
            dtype=self.dtype,
        )

        self.alpha = torch.nn.Parameter(
            data=torch.logit(
                torch.as_tensor(float(initial_alpha), dtype=dtype),
                eps=torch.finfo(self.dtype).eps,
            ),
            requires_grad=True,
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        raise RuntimeError("The module is not intended to be called directly.")

    def recall(
        self,
        *,
        z: torch.Tensor,
        T_star: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = self.store.read()

        T_base, gain = T_Zero_Fused_Real(
            z=z,
            z_j=data.z_j,
            vec_d_j=data.vec_d_j,
            T_hat_j=data.T_hat_j,
            alpha_j=data.alpha_j,
            sigma_par=data.sigma_par,
            sigma_perp=data.sigma_perp,
            max_q=self.max_q,
            eps=self.eps,
        )

        if T_star is None:
            return T_base

        tiny = torch.finfo(z.dtype).tiny
        alpha = torch.sigmoid(self.alpha)

        gain_det = gain.detach()
        E_det = (T_base - T_star).detach()

        grad_T_hat_j = gain_det.transpose(0, 1) @ E_det
        T_hat_j_delta = -alpha * grad_T_hat_j

        with torch.no_grad():
            n = torch.linalg.norm(T_hat_j_delta, ord="fro")
            s = torch.clamp(self.delta_T_hat_j_cap / (n + tiny), max=1.0)
        T_hat_j_delta = T_hat_j_delta * s

        T_hat_base = self.store.T_hat_j
        old_delta = self.store.T_hat_j_delta.clone().detach()

        T_from_local = gain.unsqueeze(-1) * (T_hat_base + T_hat_j_delta).unsqueeze(0)
        T_from_local = T_from_local.sum(dim=1)
        T_from_store = gain.unsqueeze(-1) * old_delta.unsqueeze(0)
        T_from_store = T_from_store.sum(dim=1)
        T = T_from_local + T_from_store

        self.store.update(T_hat_j_delta=T_hat_j_delta)

        return T
