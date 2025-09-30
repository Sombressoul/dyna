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
    guards: bool = True, # Use numerical guards (True/False -> stability/speed)
    q_theta: int = 24,  # Jacobi nodes
    q_rad: int = 128,  # Laguerre nodes
) -> torch.Tensor:
    # ============================================================
    # GUARDS
    # ============================================================
    if bool(guards):
        assert (alpha_j > 0).all(), "CPSF/T_Omega requires alpha_j > 0."
        assert (sigma_par > 0).all(), "CPSF/T_Omega requires sigma_par > 0."
        assert (sigma_perp > 0).all(), "CPSF/T_Omega requires sigma_perp > 0."
        assert int(q_theta) > 0, "CPSF/T_Omega requires q_theta > 0."
        assert int(q_rad) > 0, "CPSF/T_Omega requires q_rad > 0."

    # ============================================================
    # VARIABLES
    # ============================================================
    Q_THETA = int(q_theta)
    Q_RAD = int(q_rad)

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
    NU = torch.tensor(float(N - 1), dtype=dtype_r, device=device)
    PI = torch.tensor(torch.pi, dtype=dtype_r, device=device)

    # Common
    x = z - z_j  # [B,M,N] complex
    x_frac_re = torch.remainder((z - z_j).real + 0.5, 1.0) - 0.5
    x_frac_im = torch.remainder((z - z_j).imag + 0.5, 1.0) - 0.5
    x_frac = torch.complex(x_frac_re, x_frac_im)
    sigma_par_clamped = torch.clamp(sigma_par, min=tiny)
    sigma_perp_clamped = torch.clamp(sigma_perp, min=tiny)
    precision_perp = torch.reciprocal(sigma_perp)  # [B,M]
    precision_par = torch.reciprocal(sigma_par)  # [B,M]
    precision_perp_clamped = torch.clamp(precision_perp, min=tiny)  # [B,M]
    precision_par_clamped = torch.clamp(precision_par,  min=tiny)  # [B,M]
    precision_excess_par = precision_par - precision_perp  # [B,M]
    precision_excess_par_clamped = precision_par_clamped - precision_perp_clamped

    # ============================================================
    # ZERO-FRAME
    # 
    # Note: non-periodized.
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
    # DERIVATIVES
    #
    # Note: vec_d, vec_d_j — unit by default.
    # Note: tail *is* periodized, thus use x_frac.
    # ============================================================
    u_re = vec_d_j.real
    u_im = vec_d_j.imag

    inner_ux_re = (u_re * x_frac.real + u_im * x_frac.imag).sum(dim=-1)  # [B,M]
    inner_ux_im = (u_re * x_frac.imag - u_im * x_frac.real).sum(dim=-1)  # [B,M]

    anisotropy_ratio = precision_excess_par / precision_perp_clamped  # [B,M]

    metric_mix_re = precision_perp_clamped.unsqueeze(-1) * x_frac.real + precision_excess_par.unsqueeze(-1) * (inner_ux_re.unsqueeze(-1) * u_re - inner_ux_im.unsqueeze(-1) * u_im)  # [B,M,N]
    metric_mix_im = precision_perp_clamped.unsqueeze(-1) * x_frac.imag + precision_excess_par.unsqueeze(-1) * (inner_ux_re.unsqueeze(-1) * u_im + inner_ux_im.unsqueeze(-1) * u_re)  # [B,M,N]
    metric_mix_norm_sq = (metric_mix_re * metric_mix_re + metric_mix_im * metric_mix_im).sum(dim=-1)  # [B,M]
    gamma_sq = torch.clamp(metric_mix_norm_sq / precision_perp_clamped, min=0.0)  # [B,M]

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
    lam_theta = torch.clamp(lam_theta, min=eps)

    # ============================================================
    # LAGUERRE
    #
    # Note: Weights sum to 1 (divide by mu0); nodes are returned as t = x/(x+1) in [0,1);
    #   to use the Gamma(a+1, 1) measure with density ~ x^a * exp(-x) on [0,∞),
    #   first recover x via x = t/(1 - t).
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

    x_perp_re = x_frac.real - (inner_ux_re.unsqueeze(-1) * u_re - inner_ux_im.unsqueeze(-1) * u_im)  # [B,M,N]
    x_perp_im = x_frac.imag - (inner_ux_re.unsqueeze(-1) * u_im + inner_ux_im.unsqueeze(-1) * u_re)  # [B,M,N]
    x_perp_norm_sq = (x_perp_re * x_perp_re + x_perp_im * x_perp_im).sum(dim=-1)  # [B,M]

    xprime_norm_sq = precision_perp_clamped * x_perp_norm_sq + precision_par_clamped * inner_ux_abs_sq  # [B,M]

    # ============================================================
    # TAIL
    #
    # Note: constants pi and 2*pi are consistent with T_PD (see: t_pd.py).
    # ============================================================
    t_std = torch.clamp(x_rad.clamp(max=1.0 - eps) / (1.0 - x_rad), min=tiny)  # [Q_RAD]
    bessel_arg = 2.0 * PI * torch.sqrt(
        (gamma_sq[..., None, None] / lam_theta[..., None])  # [B,M,Q_THETA,1]
        * (t_std.view(1, 1, 1, -1) / PI)  # [1,1,1,Q_RAD]
    )  # [B,M,Q_THETA,Q_RAD]

    # Bessel J_{NU}(arg), (custom)
    Jv = _t_omega_jv(
        v=NU,
        z=bessel_arg,
        device=device,
        dtype=dtype_r,
    )  # [B,M,Q_THETA,Q_RAD]

    w_rad_r = (w_rad / PI.pow(NU + 1)).view(1, 1, 1, -1)  # [1,1,1,Q_RAD]; pi^{nu + 1} by T_PD
    I_rad = (w_rad_r * Jv).sum(dim=-1)  # [B,M,Q_THETA]

    w_theta_r = w_theta_bm.expand_as(I_rad)  # [B,M,Q_THETA]
    I_theta = (w_theta_r * I_rad).sum(dim=-1)  # [B,M]

    gauss_dim_prefactor = sigma_par_clamped * torch.pow(sigma_perp_clamped, C - 1.0)  # [B,M]
    tail_base = gauss_dim_prefactor * torch.exp(-PI * xprime_norm_sq)  # [B,M]
    tail_integral = torch.clamp(I_theta, min=0.0)  # [B,M]

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
