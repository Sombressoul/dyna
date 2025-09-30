import torch

from enum import Enum, auto as enum_auto

from dyna.lib.cpsf.functional.core_math import delta_vec_d
from dyna.lib.cpsf.functional.t_omega_math import _t_omega_jv, _t_omega_roots_jacobi, _t_omega_roots_gen_laguerre


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
) -> torch.Tensor:
    # ============================================================
    # VARIABLES
    # ============================================================
    Q_THETA = 24
    Q_RAD = 128

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

    # Common
    x = z - z_j  # [B,M,N] complex
    sigma_par_clamped = torch.clamp(sigma_par, min=tiny)
    sigma_perp_clamped = torch.clamp(sigma_perp, min=tiny)
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
    # Note: vec_d, vec_d_j — unit by default.
    # ============================================================
    vec_d_j_norm_sq = (vec_d_j.real * vec_d_j.real + vec_d_j.imag * vec_d_j.imag).sum(dim=-1)  # [B,M]
    vec_d_j_norm_inv = torch.rsqrt(torch.clamp(vec_d_j_norm_sq, min=tiny))  # [B,M]
    u_re = vec_d_j.real * vec_d_j_norm_inv.unsqueeze(-1)  # [B,M,N]
    u_im = vec_d_j.imag * vec_d_j_norm_inv.unsqueeze(-1)  # [B,M,N]

    inner_ux_re = (u_re * x.real + u_im * x.imag).sum(dim=-1)  # [B,M]
    inner_ux_im = (u_re * x.imag - u_im * x.real).sum(dim=-1)  # [B,M]

    anisotropy_ratio = precision_excess_par / torch.clamp(precision_perp, min=tiny)  # [B,M]

    metric_mix_re = precision_perp.unsqueeze(-1) * x.real + precision_excess_par.unsqueeze(-1) * (inner_ux_re.unsqueeze(-1) * u_re - inner_ux_im.unsqueeze(-1) * u_im)  # [B,M,N]
    metric_mix_im = precision_perp.unsqueeze(-1) * x.imag + precision_excess_par.unsqueeze(-1) * (inner_ux_re.unsqueeze(-1) * u_im + inner_ux_im.unsqueeze(-1) * u_re)  # [B,M,N]
    metric_mix_norm_sq = (metric_mix_re * metric_mix_re + metric_mix_im * metric_mix_im).sum(dim=-1)  # [B,M]
    gamma_sq = torch.clamp(metric_mix_norm_sq / torch.clamp(precision_perp, min=tiny), min=0.0)  # [B,M]

    # ============================================================
    # JACOBI
    # ============================================================
    x_jac, w_jac = _t_omega_roots_jacobi(
        N=Q_THETA,
        alpha=-0.5,
        beta=NU - 0.5,
        normalize=True,
        return_weights=True,
        dtype=dtype_r,
        device=device,
    )

    t_theta_bm = x_jac.view(1, 1, -1)  # [1,1,Q]
    w_theta_bm = w_jac.view(1, 1, -1)  # [1,1,Q]

    lam_theta = 1.0 + anisotropy_ratio[..., None] * (1.0 - t_theta_bm)  # [B,M,Q]
    lam_theta = torch.clamp(lam_theta, min=tiny)

    # ============================================================
    # LAGUERRE
    # ============================================================
    x_rad, w_rad = _t_omega_roots_gen_laguerre(
        N=Q_RAD,
        alpha=NU,  # NU = N-1
        normalize=True,
        return_weights=True,
        dtype=dtype_r,
        device=device,
    )

    # ============================================================
    # WHITENING
    # ============================================================
    inner_ux_abs_sq = inner_ux_re * inner_ux_re + inner_ux_im * inner_ux_im  # [B,M]

    x_perp_re = x.real - (inner_ux_re.unsqueeze(-1) * u_re - inner_ux_im.unsqueeze(-1) * u_im)  # [B,M,N]
    x_perp_im = x.imag - (inner_ux_re.unsqueeze(-1) * u_im + inner_ux_im.unsqueeze(-1) * u_re)  # [B,M,N]
    x_perp_norm_sq = (x_perp_re * x_perp_re + x_perp_im * x_perp_im).sum(dim=-1)  # [B,M]

    xprime_norm_sq = precision_perp_clamped * x_perp_norm_sq + precision_par_clamped * inner_ux_abs_sq  # [B,M]

    # ============================================================
    # TAIL
    # ============================================================
    t = torch.clamp(x_rad, min=tiny)  # [Qr]
    bessel_arg = 2.0 * torch.sqrt(
        (gamma_sq[..., None, None] / torch.clamp(lam_theta[..., None], min=tiny))  # [B,M,Qθ,1]
        * t.view(1, 1, 1, -1)  # [1,1,1,Qr]
    )  # [B,M,Qθ,Qr]

    # Bessel J_{NU}(arg), (custom)
    Jv = _t_omega_jv(
        v=NU,
        z=bessel_arg,
        device=device,
        dtype=dtype_r,
    )  # [B,M,Q_THETA,Q_RAD]

    w_rad_r = w_rad.view(1, 1, 1, -1)  # [1,1,1,Q_RAD]
    I_rad = (w_rad_r * Jv).sum(dim=-1)  # [B,M,Q_THETA]

    w_theta_r = w_theta_bm.expand_as(I_rad)  # [B,M,Q_THETA]
    I_theta = (w_theta_r * I_rad).sum(dim=-1)  # [B,M]

    gauss_dim_prefactor = sigma_par_clamped * torch.pow(sigma_perp_clamped, C - 1.0)  # [B,M]
    tail_base = gauss_dim_prefactor * torch.exp(-PI * xprime_norm_sq)  # [B,M]
    tail_integral = torch.clamp(I_theta, min=tiny)  # [B,M]

    gain_tail = alpha_j * A_dir * tail_base * tail_integral

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
