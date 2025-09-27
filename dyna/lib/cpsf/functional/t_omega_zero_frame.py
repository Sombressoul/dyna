import torch

from dyna.lib.cpsf.functional.core_math import (
    delta_vec_d,
)


def _t_omega_zero_frame(
    *,
    z: torch.Tensor,  # [B, M, N] (complex)
    z_j: torch.Tensor,  # [B, M, N] (complex)
    vec_d: torch.Tensor,  # [B, M, N] (complex)
    vec_d_j: torch.Tensor,  # [B, M, N] (complex)
    T_hat_j: torch.Tensor,  # [B, M, S] (complex)
    alpha_j: torch.Tensor,  # [B, M] (real)
    sigma_par: torch.Tensor,  # [B, M] (real > 0) - complex convention (no squaring)
    sigma_perp: torch.Tensor,  # [B, M] (real > 0) - complex convention (no squaring)
) -> torch.Tensor:
    # 0-frame positional delta: NO wrapping (W=0 means single zero offset)
    x = z - z_j  # [B, M, N] (complex)

    # rank-1 metric coefficients (complex convention: 1/sigma, no squaring)
    inv_sp = torch.reciprocal(sigma_par)  # [B, M]  (1/sigma_par)
    inv_sq = torch.reciprocal(sigma_perp)  # [B, M]  (1/sigma_perp)
    a = inv_sq
    b = inv_sp - inv_sq

    # q_pos = a*||x||^2 + b*|<u,x>|^2  with u = vec_d_j (assumed unit-norm)
    norm2_x = (x.conj() * x).real.sum(dim=-1)  # [B, M]
    inner = (vec_d_j.conj() * x).sum(dim=-1)  # [B, M] complex
    proj_abs2 = (inner.conj() * inner).real  # |<u,x>|^2
    q_pos = a * norm2_x + b * proj_abs2  # [B, M]
    A_pos = torch.exp(-torch.pi * q_pos)  # [B, M]

    # directional factor (isotropic with sigma_perp; no periodization)
    dv = delta_vec_d(vec_d, vec_d_j)  # [B, M, N]
    norm2_dv = (dv.conj() * dv).real.sum(dim=-1)  # [B, M]
    q_dir = inv_sq * norm2_dv
    A_dir = torch.exp(-torch.pi * q_dir)  # [B, M]

    gain = (alpha_j * A_pos * A_dir).unsqueeze(-1)  # [B, M, 1]
    zero_frame = (gain.to(T_hat_j.dtype) * T_hat_j).sum(dim=1)  # [B, S] (complex)
    return zero_frame


def T_Omega_ZF(
    *,
    z: torch.Tensor,  # [B, N] (complex)
    z_j: torch.Tensor,  # [B, M, N] (complex)
    vec_d: torch.Tensor,  # [B, N] (complex)
    vec_d_j: torch.Tensor,  # [B, M, N] (complex)
    T_hat_j: torch.Tensor,  # [B, M, S] (complex)
    alpha_j: torch.Tensor,  # [B, M] (real)
    sigma_par: torch.Tensor,  # [B, M] (real > 0) - complex convention (no squaring)
    sigma_perp: torch.Tensor,  # [B, M] (real > 0) - complex convention (no squaring)
) -> torch.Tensor:

    # Broadcast
    B, M, N = vec_d_j.shape
    z = z.unsqueeze(1).expand(B, M, N)
    vec_d = vec_d.unsqueeze(1).expand(B, M, N)

    # Get 0-frame
    zero_frame = _t_omega_zero_frame(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
    )

    return zero_frame
