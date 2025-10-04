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
    # Note: 1/sigma (not 1/sigma^2); sigma_* are scale parameters.
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
    sigma_par_inv = torch.reciprocal(sigma_par_clamped)   # 1/sigma_par
    sigma_perp_inv = torch.reciprocal(sigma_perp_clamped)  # 1/sigma_perp
    sigma_par_inv_clamped = torch.clamp(sigma_par_inv,  min=tiny)
    sigma_perp_inv_clamped = torch.clamp(sigma_perp_inv, min=tiny)
    sigma_excess_par_clamped = sigma_par_inv_clamped - sigma_perp_inv_clamped

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
    q_pos = sigma_perp_inv_clamped * x_norm_sq + sigma_excess_par_clamped * inner_abs_sq  # [B,M]
    A_pos = torch.exp(-PI * q_pos)  # [B,M]

    # A_dir: [B,M]
    delta_d = delta_vec_d(vec_d, vec_d_j)  # [B,M,N] complex
    delta_d_norm_sq = (delta_d.real * delta_d.real + delta_d.imag * delta_d.imag).sum(dim=-1)  # [B,M]
    A_dir = torch.exp(-PI * sigma_perp_inv_clamped * delta_d_norm_sq)  # [B,M]

    # Gain
    gain_zero = alpha_j * A_pos * A_dir  # [B,M]
    T_zero = (gain_zero.unsqueeze(-1) * T_hat_j).sum(dim=1)  # [B,S]

    if return_components == T_Omega_Components.ZERO:
        return T_zero

    # ============================================================
    # PREPARE GEOMETRY (1D-HS core in R^{2N})
    # ============================================================
    # Periodized displacement in R^{2N}: wrap BOTH Re/Im to [-1/2, 1/2]
    x_wrapped_re = torch.remainder(x.real + 0.5, 1.0) - 0.5  # [B,M,N]
    x_wrapped_im = torch.remainder(x.imag + 0.5, 1.0) - 0.5  # [B,M,N]
    x_wrapped_R2N = torch.cat([x_wrapped_re, x_wrapped_im], dim=-1)  # [B,M,2*N]

    # Orientation u in R^{2N} (unit)
    u_re = vec_d_j.real  # [B,M,N]
    u_im = vec_d_j.imag  # [B,M,N]
    u_raw = torch.cat([u_re, u_im], dim=-1)  # [B,M,2*N]
    u_norm = torch.linalg.norm(u_raw, dim=-1, keepdim=True).clamp(min=tiny)  # [B,M,1]
    u_R2N = u_raw / u_norm  # [B,M,2*N]

    # Anisotropy params
    delta_sigma = sigma_par_clamped - sigma_perp_clamped  # [B,M]
    abs_delta_sigma = delta_sigma.abs()  # [B,M]
    sign_delta = torch.sign(delta_sigma).to(dtype_r)  # [B,M] (+1 osc, -1 hyp)

    # Theta base (dual isotropic scale)
    q = torch.exp(-PI * sigma_perp_clamped)  # [B,M]

    # HS scaling
    xi_scale = (1.0 / (2.0 * PI)).to(dtype_r)  # scalar
    s_scale = torch.sqrt(4.0 * PI * abs_delta_sigma).clamp_min(tiny)  # [B,M]

    # Flags
    is_iso = (abs_delta_sigma <= eps)  # [B,M]  (delta sigma ~= 0 -> isotropic limit)
    is_osc = (delta_sigma >= 0)  # [B,M]  (HS branch: osc vs hyperbolic)

    # ============================================================
    # ASSEMBLY TAIL
    # ============================================================
    gain_tail = ...
    T_tail = (gain_tail.unsqueeze(-1) * T_hat_j).sum(dim=1)  # [B,S]

    if return_components == T_Omega_Components.TAIL:
        return T_tail
    elif return_components == T_Omega_Components.BOTH:
        return T_zero, T_tail
    elif return_components == T_Omega_Components.UNION:
        return T_zero + T_tail
    else:
        raise ValueError(f"Unknown mode: '{return_components=}'")
