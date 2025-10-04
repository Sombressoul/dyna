import torch

from enum import Enum, auto as enum_auto

from dyna.lib.cpsf.functional.core_math import delta_vec_d


class T_Omega_Components(Enum):
    ZERO = enum_auto()
    TAIL = enum_auto()
    BOTH = enum_auto()
    UNION = enum_auto()


def T_Omega(
    z: torch.Tensor,  # [B,N] (complex)
    z_j: torch.Tensor,  # [B,M,N] (complex)
    vec_d: torch.Tensor,  # vec_d: [B,N] (complex); unit
    vec_d_j: torch.Tensor,  # vec_d_j: [B,M,N] (complex); unit
    T_hat_j: torch.Tensor,  # T_hat_j: [B,M,S] (complex)
    alpha_j: torch.Tensor,  # alpha_j: [B,M] (real)
    sigma_par: torch.Tensor,  # sigma_par: [B,M] (real)
    sigma_perp: torch.Tensor,  # sigma_perp: [B,M] (real)
    return_components: T_Omega_Components = T_Omega_Components.UNION,
    guards: bool = True, # Use numerical guards (True/False -> stability/speed)
) -> torch.Tensor:
    # ============================================================
    # GUARDS
    # ============================================================
    if bool(guards):
        assert (alpha_j > 0).all(), "CPSF/T_Omega requires alpha_j > 0."
        assert (sigma_par > 0).all(), "CPSF/T_Omega requires sigma_par > 0."
        assert (sigma_perp > 0).all(), "CPSF/T_Omega requires sigma_perp > 0."

    # ============================================================
    # VARIABLES
    # ============================================================
    # For future use.

    # ============================================================
    # BASE
    # ============================================================
    device = z.device
    dtype_r = z.real.dtype
    tiny = torch.finfo(dtype_r).tiny
    eps = torch.finfo(dtype_r).eps

    # ============================================================
    # MAIN
    #
    # Note: precision = 1/sigma (not 1/sigma^2); sigma_ are scale
    #   parameters.
    # ============================================================
    # Broadcast
    B, M, N = vec_d_j.shape
    z = z.unsqueeze(1).expand(B, M, N)
    vec_d = vec_d.unsqueeze(1).expand(B, M, N)

    # Fast checks
    assert N >= 2, "CPSF/T_Omega requires N >= 2 (complex N)."

    # Constants
    C = torch.tensor(float(N), dtype=dtype_r, device=device)
    PI = torch.tensor(torch.pi, dtype=dtype_r, device=device)

    # Common
    x = z - z_j  # [B,M,N] complex

    sigma_par_clamped = torch.clamp(sigma_par,  min=tiny)  # [B,M]
    sigma_perp_clamped = torch.clamp(sigma_perp, min=tiny)  # [B,M]
    precision_par = torch.reciprocal(sigma_par_clamped)   # 1/sigma_par
    precision_perp = torch.reciprocal(sigma_perp_clamped)  # 1/sigma_perp
    precision_par_clamped = torch.clamp(precision_par,  min=tiny)
    precision_perp_clamped = torch.clamp(precision_perp, min=tiny)
    precision_excess_par_clamped = precision_par_clamped - precision_perp_clamped

    # ============================================================
    # ZERO-FRAME
    # 
    # Note: non-periodized, real space.
    # ============================================================
    # q_pos: [B,M]
    x_norm_sq = (x.real * x.real + x.imag * x.imag).sum(dim=-1)  # [B,M]
    inner_re = (vec_d_j.real * x.real + vec_d_j.imag * x.imag).sum(dim=-1)  # [B,M]
    inner_im = (vec_d_j.real * x.imag - vec_d_j.imag * x.real).sum(dim=-1)  # [B,M]
    inner_abs_sq = inner_re * inner_re + inner_im * inner_im  # [B,M]
    q_pos = precision_perp_clamped * x_norm_sq + precision_excess_par_clamped * inner_abs_sq  # [B,M]
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
    # TAIL VARIABLES
    #
    # Note: vec_d, vec_d_j — unit by default.
    # Note: tail *is* periodized, thus use x_frac.
    # ============================================================
    x_frac_re = torch.remainder((z - z_j).real + 0.5, 1.0) - 0.5
    x_frac_im = torch.remainder((z - z_j).imag + 0.5, 1.0) - 0.5
    x_frac = torch.complex(x_frac_re, x_frac_im)

    anisotropy_ratio = (precision_par_clamped / precision_perp_clamped) - 1.0  # [B,M]

    # ============================================================
    # TAIL
    # ============================================================
    tail_weights = ...

    gain_tail = alpha_j * A_dir * tail_weights

    # ============================================================
    #                Assembly gain_tail and T_tail
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
