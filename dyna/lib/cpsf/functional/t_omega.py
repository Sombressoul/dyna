import torch

from enum import Enum, auto as enum_auto

from dyna.lib.cpsf.functional.core_math import delta_vec_d
from dyna.lib.cpsf.functional.t_omega_math import _t_omega_jv


class T_Omega_Components(Enum):
    ZERO = enum_auto()
    TAIL = enum_auto()
    BOTH = enum_auto()
    UNION = enum_auto()


def T_Omega(
    z: torch.Tensor,  # [B,N] (complex)
    z_j: torch.Tensor,  # [B,M,N] (complex)
    vec_d: torch.Tensor,  # vec_d: [B,N] (complex)
    vec_d_j: torch.Tensor,  # vec_d_j: [B,M,N] (complex)
    T_hat_j: torch.Tensor,  # T_hat_j: [B,M,S] (complex)
    alpha_j: torch.Tensor,  # alpha_j: [B,M] (real)
    sigma_par: torch.Tensor,  # sigma_par: [B,M] (real)
    sigma_perp: torch.Tensor,  # sigma_perp: [B,M] (real)
    return_components: T_Omega_Components = T_Omega_Components.UNION,
) -> torch.Tensor:
    # ============================================================
    # BASE
    # ============================================================
    device = z.device
    dtype_r = z.real.dtype
    tiny = torch.finfo(dtype_r).tiny

    # ============================================================
    # MAIN
    # ============================================================
    # Broadcast
    B, M, N = vec_d_j.shape
    z = z.unsqueeze(1).expand(B, M, N)
    vec_d = vec_d.unsqueeze(1).expand(B, M, N)

    # Constants
    C = torch.tensor(float(N), dtype=dtype_r, device=device)
    NU = torch.tensor(float(N - 1), dtype=dtype_r, device=device)
    PI = torch.tensor(torch.pi, dtype=dtype_r, device=device)
    LOG2 = torch.tensor(2.0, dtype=dtype_r, device=device).log()
    FOUR_PI = 4.0 * PI
    LOG_PI = PI.log()

    # Common
    x = z - z_j  # [B,M,N] complex
    precision_perp = torch.reciprocal(sigma_perp)  # [B,M]
    precision_par = torch.reciprocal(sigma_par)  # [B,M]
    precision_excess_par = precision_par - precision_perp  # [B,M]
    precision_perp_clamped = torch.clamp(precision_perp, min=tiny)  # [B,M]
    precision_par_clamped = torch.clamp(precision_par,  min=tiny)  # [B,M]

    # ============================================================
    # ZERO-FRAME
    # ============================================================
    # q_pos: [B,M]
    x_norm_sq = (x.real * x.real + x.imag * x.imag).sum(dim=-1)  # [B,M]
    inner_re = (vec_d_j.real * x.real + vec_d_j.imag * x.imag).sum(dim=-1)  # [B,M]
    inner_im = (vec_d_j.real * x.imag - vec_d_j.imag * x.real).sum(dim=-1)  # [B,M]
    inner_abs_sq = inner_re * inner_re + inner_im * inner_im  # [B,M]
    q_pos = precision_perp * x_norm_sq + precision_excess_par * inner_abs_sq  # [B,M]
    A_pos = torch.exp(-PI * q_pos)  # [B,M]

    # A_dir: [B,M]
    delta_d = delta_vec_d(vec_d, vec_d_j)  # [B,M,N] complex
    delta_d_norm_sq = (delta_d.real * delta_d.real + delta_d.imag * delta_d.imag).sum(dim=-1)  # [B,M]
    A_dir = torch.exp(-PI * precision_perp_clamped * delta_d_norm_sq)  # [B,M]

    # Gain
    gain_zero = alpha_j * A_pos * A_dir  # [B,M]
    T_zero = (gain_zero.unsqueeze(-1) * T_hat_j).sum(dim=1)  # [B,S]

    if return_components == T_Omega_Components.ZERO:
        return T_zero

    # ============================================================
    # DERIVATIVES
    # ============================================================
    vec_d_j_norm_sq = (vec_d_j.real * vec_d_j.real + vec_d_j.imag * vec_d_j.imag).sum(dim=-1)  # [B,M]
    vec_d_j_norm_inv = torch.rsqrt(torch.clamp(vec_d_j_norm_sq, min=tiny))  # [B,M]
    u_re = vec_d_j.real * vec_d_j_norm_inv.unsqueeze(-1)  # [B,M,N]
    u_im = vec_d_j.imag * vec_d_j_norm_inv.unsqueeze(-1)  # [B,M,N]

    inner_ux_re = (u_re * x.real + u_im * x.imag).sum(dim=-1)  # [B,M]
    inner_ux_im = (u_re * x.imag - u_im * x.real).sum(dim=-1)  # [B,M]

    # ============================================================
    # WHITENING
    # ============================================================
    inner_ux_abs_sq = inner_ux_re * inner_ux_re + inner_ux_im * inner_ux_im  # [B,M]
    
    x_perp_re = x.real - (inner_ux_re.unsqueeze(-1) * u_re - inner_ux_im.unsqueeze(-1) * u_im)  # [B,M,N]
    x_perp_im = x.imag - (inner_ux_re.unsqueeze(-1) * u_im + inner_ux_im.unsqueeze(-1) * u_re)  # [B,M,N]
    x_perp_norm_sq = (x_perp_re * x_perp_re + x_perp_im * x_perp_im).sum(dim=-1)  # [B,M]

    xprime_norm_sq = precision_perp_clamped * x_perp_norm_sq + precision_par_clamped * inner_ux_abs_sq  # [B,M]

    # ============================================================
    # J_v
    # ============================================================
    beta = torch.sqrt(torch.clamp(FOUR_PI * xprime_norm_sq, min=tiny))  # [B,M]
    log_beta = torch.log(torch.clamp(beta, min=tiny))  # [B,M]
    J_nu = _t_omega_jv(v=NU, z=beta, dtype=dtype_r, device=device)  # [B,M]

    log_RJ = torch.log(torch.clamp(J_nu.abs(), min=tiny)) + torch.lgamma(C) + (1.0 - C) * log_beta  # [B,M]
    log_Cj = -torch.log(precision_par_clamped) - (C - 1.0) * torch.log(precision_perp_clamped)  # [B,M]
    log_A_dir = torch.log(torch.clamp(A_dir,  min=tiny))  # [B,M]
    log_alpha = torch.log(torch.clamp(alpha_j, min=tiny))  # [B,M]
    log_Cang = C * LOG_PI - torch.lgamma(C)  # [], scalar
    log_Kp = (C - 1.0) * LOG2  # [], scalar

    log_gain_jv = log_RJ + log_Cj + log_A_dir + log_alpha + log_Cang + log_Kp  # [B,M]

    print("\n" + "\n".join(
        [
            f"xprime_norm_sq: {xprime_norm_sq.mean().item()}",
            f"-------------------------------------------",
            f"log_RJ mean   : {log_RJ.mean().item()}",
            f"log_Cj mean   : {log_Cj.mean().item()}",
            f"log_A_dir mean: {log_A_dir.mean().item()}",
            f"log_alpha mean: {log_alpha.mean().item()}",
        ]
    ))

    gain_tail = torch.exp(log_gain_jv)  # [B,M]

    # ============================================================
    # Assembly T_tail
    # ============================================================
    T_tail = (gain_tail.unsqueeze(-1) * T_hat_j).sum(dim=1)  # [B,S]

    if return_components == T_Omega_Components.TAIL:
        return T_tail
    elif return_components == T_Omega_Components.BOTH:
        return T_zero, T_tail
    elif return_components == T_Omega_Components.UNION:
        return T_zero + T_tail
    else:
        raise ValueError(f"Unknown mode: '{return_components=}'")
