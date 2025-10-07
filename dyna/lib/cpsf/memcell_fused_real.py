import math
import torch
import torch.nn as nn


from dataclasses import dataclass
from enum import Enum, auto as enum_auto
from typing import Tuple, Optional


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
    q_pos = max_q - torch.nn.functional.softplus(max_q - q_pos)  # soft cap
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
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.N = int(N)
        self.M = int(M)
        self.S = int(S)
        self.dtype = dtype

        self.z_j = torch.nn.Parameter(
            data=self._init_z_j(),
            requires_grad=True,
        )
        self.vec_d_j = torch.nn.Parameter(
            data=self._init_vec_d_j(),
            requires_grad=True,
        )
        self.T_hat_j = torch.nn.Parameter(
            data=self._init_T_hat_j(),
            requires_grad=True,
        )
        self.T_hat_j_delta = torch.nn.Parameter(
            data=torch.zeros([self.M, self.S], dtype=self.dtype),
            requires_grad=False,
        )
        self.alpha_j = torch.nn.Parameter(
            data=torch.empty([self.M], dtype=self.dtype).uniform_(0.5, 1.5),
            requires_grad=True,
        )
        sigma_par, sigma_perp = self._init_sigmas()
        self.sigma_par = torch.nn.Parameter(
            data=sigma_par,
            requires_grad=True,
        )
        self.sigma_perp = torch.nn.Parameter(
            data=sigma_perp,
            requires_grad=True,
        )

    def _init_z_j(
        self,
        scale: float = 1.0,
    ) -> torch.Tensor:
        # Pseudo Latin Hypercube
        intervals = torch.arange(self.M, dtype=self.dtype) / self.M
        samples = torch.zeros(self.M, self.N, dtype=self.dtype)

        for dim in range(self.N):
            perm = torch.randperm(self.M)
            samples[:, dim] = (
                intervals[perm] + torch.rand(self.M, dtype=self.dtype) / self.M
            )

        samples = 2 * samples - 1
        samples = samples * scale

        return samples

    def _init_vec_d_j(
        self,
    ) -> torch.Tensor:
        # Random directions on the sphere
        directions = torch.randn(self.M, self.N, dtype=self.dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        return directions

    def _init_T_hat_j(
        self,
        scale: float = 1.0e-3,
    ) -> torch.Tensor:
        # Orthogonal vectors for M < S
        vec = torch.randn([self.M, self.S], dtype=self.dtype)
        if self.M < self.S:
            vec, _ = torch.linalg.qr(vec.T)
            vec = vec.T * scale
        return vec

    def _init_sigmas(
        self,
        anisotropy: float = 2.0,
        coverage_factor: float = 0.5,
        min_sigma: float = 1e-3,
    ) -> torch.Tensor:
        # Uniform coverage
        top_k = min(int(self.M**0.5), self.M - 1)
        distances = torch.cdist(self.z_j, self.z_j, p=2)  # [M, M]
        distances = distances + torch.eye(self.M) * 1.0e9  # ignore self
        k_nearest = distances.topk(k=top_k, largest=False).values  # [M, top_k]
        median_dist = k_nearest.median(dim=-1).values.to(self.dtype)  # [M]

        sigma_perp = median_dist * coverage_factor
        sigma_par = sigma_perp * (1.0 + anisotropy)

        sigma_perp = torch.clamp(sigma_perp, min=min_sigma)
        sigma_par = torch.clamp(sigma_par, min=min_sigma)

        return sigma_par, sigma_perp

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


class CPSFMemcellFusedRealGradMode(Enum):
    FULL = enum_auto()
    MIXED = enum_auto()
    SAFE = enum_auto()


class CPSFMemcellFusedReal(nn.Module):
    def __init__(
        self,
        *,
        N: int,
        M: int,
        S: int,
        initial_alpha: float = 1.0e-2,
        delta_T_hat_j_cap: Optional[float] = 1.0,
        max_q: float = 25.0,
        eps: float = 1.0e-6,
        grad_mode: CPSFMemcellFusedRealGradMode = CPSFMemcellFusedRealGradMode.MIXED,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.N = N
        self.M = M
        self.S = S
        self.initial_alpha = initial_alpha
        self.delta_T_hat_j_cap = delta_T_hat_j_cap
        self.max_q = max_q
        self.eps = eps
        self.grad_mode = grad_mode
        self.dtype = dtype

        self.store = CPSFContributionStoreFusedReal(
            N=self.N,
            M=self.M,
            S=self.S,
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
        *agrs,
        **kwargs,
    ) -> torch.Tensor:
        raise RuntimeError("The module is not intended to be called directly.")

    # -----------------------------------------------------------------------------
    # CPSFMemcellFusedReal.recall — Gradient Flow Cheat Sheet
    # -----------------------------------------------------------------------------
    # This call performs a one-step local adaptation inside forward:
    #   1) Compute base prediction T_base and gain(z, z_j, vec_d_j, sigma_*, alpha_j)
    #   2) Build a local update ΔT_hat_j = -sigmoid(alpha) * (gain_eff^T @ E_eff)
    #   3) Trust-region scale ΔT_hat_j by ||Δ||_F (no_grad)
    #   4) Assemble final T using LIVE gain:
    #        T = sum_j gain * (T_hat_base + ΔT_hat_j) + sum_j gain * old_delta(detached)
    #   5) Accumulate ΔT_hat_j into the store buffer (detached) for subsequent reads
    #
    # Gradient destinations (common to all modes unless explicitly detached):
    #   — Request geometry: z
    #   — Contribution geometry: z_j, vec_d_j, sigma_par, sigma_perp
    #   — Contribution weight: alpha_j
    #   — Contribution data (base): T_hat_j
    #   — Memory-cell LR: self.alpha (always via ΔT_hat_j)
    #   — Reference field: T_star (depends on mode)
    #   — Store buffer (old deltas): NO grad (always detached)
    #
    # Modes control what participates in the local step ΔT_hat_j via (gain_eff, E_eff):
    #
    #   FULL mode:
    #     gain_eff = gain
    #     E_eff    = T_base - T_star
    #     Gradients:
    #       - Geometry: YES (two paths: via final T and via ΔT_hat_j)
    #       - T_hat_j (base): YES (via final T and via Δ through T_base)
    #       - alpha: YES
    #       - T_star: YES
    #
    #   MIXED mode:
    #     gain_eff = gain.detach()
    #     E_eff    = T_base.detach() - T_star
    #     Gradients:
    #       - Geometry: YES (only via final T; Δ path is blocked)
    #       - T_hat_j (base): YES (via final T; Δ path is blocked)
    #       - alpha: YES (Δ depends on alpha)
    #       - T_star: YES (appears in E_eff)
    #
    #   SAFE mode:
    #     gain_eff = gain.detach()
    #     E_eff    = (T_base - T_star).detach()
    #     Gradients:
    #       - Geometry: YES (only via final T; Δ path is blocked)
    #       - T_hat_j (base): YES (via final T)
    #       - alpha: YES (Δ depends on alpha)
    #       - T_star: NO
    #
    # Special cases & stability:
    #   — If T_star is None: returns T_base (pure read), no Δ computation, no store update.
    #   — Trust-region scaling of ΔT_hat_j is done under no_grad; it rescales magnitude
    #     without creating extra graph branches.
    #   — Clamping q <= max_q can shrink gain strongly; small gain -> small grads through geometry.
    #   — Keep the order: build T first, THEN store.update(Δ). This avoids double-counting the new Δ.
    #   — alpha is passed through sigmoid; if you want a small effective step, use:
    #         alpha = alpha_max * sigmoid(raw_alpha)
    #     and initialize raw_alpha = logit(alpha_init / alpha_max).
    #   — To ensure alpha receives gradients, compute your loss on the output of `recall(...)`,
    #     not on a separate read path.
    # -----------------------------------------------------------------------------
    def recall(
        self,
        *,
        z: torch.Tensor,
        T_star: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = self.store.read()

        sigma_eps = torch.finfo(data.sigma_par.dtype).eps
        sigma_par = torch.nn.functional.softplus(data.sigma_par) + sigma_eps
        sigma_perp = torch.nn.functional.softplus(data.sigma_perp) + sigma_eps

        T_base, gain = T_Zero_Fused_Real(
            z=z,
            z_j=data.z_j,
            vec_d_j=data.vec_d_j,
            T_hat_j=data.T_hat_j,
            alpha_j=data.alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            max_q=self.max_q,
            eps=self.eps,
        )

        if T_star is None:
            return T_base

        if self.grad_mode is CPSFMemcellFusedRealGradMode.FULL:
            gain_eff = gain
            E_eff = T_base - T_star
        elif self.grad_mode is CPSFMemcellFusedRealGradMode.MIXED:
            gain_eff = gain.detach()
            E_eff = T_base.detach() - T_star
        elif self.grad_mode is CPSFMemcellFusedRealGradMode.SAFE:
            gain_eff = gain.detach()
            E_eff = (T_base - T_star).detach()
        else:
            raise ValueError(f"Unknown grad mode: '{self.grad_mode}'")

        tiny = torch.finfo(z.dtype).tiny
        alpha = torch.sigmoid(self.alpha)
        grad_T_hat_j = (gain_eff.transpose(0, 1) @ E_eff) / max(z.shape[0], 1)
        T_hat_j_delta_new = -alpha * grad_T_hat_j

        s = 1.0
        if self.delta_T_hat_j_cap is not None:
            with torch.no_grad():
                n = torch.linalg.norm(T_hat_j_delta_new, ord="fro")
                s = torch.clamp(self.delta_T_hat_j_cap / (n + tiny), max=1.0)

        T_hat_j_delta_new = T_hat_j_delta_new * s
        T_hat_j_delta_old = self.store.T_hat_j_delta.detach()
        T_hat_j_delta_eff = T_hat_j_delta_old + T_hat_j_delta_new

        T = gain.unsqueeze(-1) * (self.store.T_hat_j + T_hat_j_delta_eff).unsqueeze(0)
        T = T.sum(dim=1)

        self.store.update(T_hat_j_delta=T_hat_j_delta_new)

        return T
