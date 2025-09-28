import math
import torch

from enum import Enum, auto as enum_auto

import numpy as np
from scipy.special import roots_genlaguerre, ive, gammaln, logsumexp, roots_hermite, jv

from dyna.lib.cpsf.functional.core_math import (
    delta_vec_d,
)
from dyna.lib.cpsf.functional.t_omega_math import (
    _t_omega_roots_jacobi,
)


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
    #                      VARIABLES
    # ============================================================
    Q_THETA = 24

    # ============================================================
    #                      BASE
    # ============================================================
    device = z.device
    dtype_c = z.dtype
    dtype_r = z.real.dtype
    tiny = torch.finfo(dtype_r).tiny

    # ============================================================
    #                      MAIN
    # ============================================================
    # Broadcast
    B, M, N = vec_d_j.shape
    z = z.unsqueeze(1).expand(B, M, N)
    vec_d = vec_d.unsqueeze(1).expand(B, M, N)

    # Constants
    D = 2 * N
    C = float(N)
    NU = float(N - 1)
    PI = torch.tensor(torch.pi, dtype=dtype_r, device=device)
    PI2_SQRT = 2.0 * PI.sqrt()

    # Common
    x = z - z_j  # [B,M,N] complex
    precision_perp = torch.reciprocal(sigma_perp)  # [B,M]
    precision_par  = torch.reciprocal(sigma_par)  # [B,M]
    precision_excess_par = precision_par - precision_perp  # [B,M]

    # ============================================================
    #                      ZERO-FRAME
    # ============================================================
    # q_pos: [B,M]
    x_norm_sq = (x.real * x.real + x.imag * x.imag).sum(dim=-1)  # [B,M,N]
    inner_re = (vec_d_j.real * x.real + vec_d_j.imag * x.imag).sum(dim=-1)  # [B,M]
    inner_im = (vec_d_j.real * x.imag - vec_d_j.imag * x.real).sum(dim=-1)  # [B,M]
    inner_abs_sq = inner_re * inner_re + inner_im * inner_im  # [B,M]
    q_pos = precision_perp * x_norm_sq + precision_excess_par * inner_abs_sq  # [B,M]
    A_pos = torch.exp(-PI * q_pos)  # [B,M]

    # A_dir: [B,M]
    delta_d = delta_vec_d(vec_d, vec_d_j)  # [B,M,N] complex
    delta_d_norm_sq = (delta_d.real * delta_d.real + delta_d.imag * delta_d.imag).sum(dim=-1)  # [B,M]
    A_dir = torch.exp(-PI * precision_perp * delta_d_norm_sq)  # [B,M]

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

    anisotropy_ratio = precision_excess_par / torch.clamp(precision_perp, min=tiny)  # [B,M]

    metric_mix_re = precision_perp.unsqueeze(-1) * x.real + precision_excess_par.unsqueeze(-1) * (inner_ux_re.unsqueeze(-1) * u_re - inner_ux_im.unsqueeze(-1) * u_im)  # [B,M,N]
    metric_mix_im = precision_perp.unsqueeze(-1) * x.imag + precision_excess_par.unsqueeze(-1) * (inner_ux_re.unsqueeze(-1) * u_im + inner_ux_im.unsqueeze(-1) * u_re)  # [B,M,N]
    metric_mix_norm_sq = (metric_mix_re * metric_mix_re + metric_mix_im * metric_mix_im).sum(dim=-1)  # [B,M]
    gamma_sq = torch.clamp(metric_mix_norm_sq / torch.clamp(precision_perp, min=tiny), min=0.0)  # [B,M]

    gauss_dim_prefactor = (2.0 ** NU) * torch.pow(torch.clamp(precision_perp, min=tiny), -C)  # [B,M]
    bessel_arg = PI2_SQRT * torch.sqrt(gamma_sq)  # [B,M]

    # ============================================================
    # JACOBI
    # ============================================================
    x_jac, w_jac = _t_omega_roots_jacobi(
        N=Q_THETA,
        alpha=-0.5,
        beta=NU - 0.5,
        normalize=True,
        return_weights=True,
        dtype=dtype_c,
        device=device,
    )

    t_theta_bm = x_jac.view(1, 1, -1)  # [1,1,Q]
    w_theta_bm = w_jac.view(1, 1, -1)  # [1,1,Q]

    lam_theta = 1.0 + anisotropy_ratio.to(dtype_r)[..., None] * (1.0 - t_theta_bm)  # [B,M,Q]
    lam_theta = torch.clamp(lam_theta, min=tiny)
    beta_theta = bessel_arg.to(dtype_r)[..., None] / torch.sqrt(lam_theta)  # [B,M,Q]

    # ============================================================
    #         Masks
    # ============================================================
    tiny64 = np.finfo(np.float64).tiny

    lam_np    = lam_theta.detach().cpu().numpy()                         # [B,M,Q]
    beta_t_np = beta_theta.detach().cpu().numpy()                        # [B,M,Q]
    KD_np     = gauss_dim_prefactor.detach().to(torch.float64).cpu().numpy()             # [B,M]
    wth_np    = w_theta_bm.detach().cpu().numpy()                        # [1,1,Q]
    Adir_np   = A_dir.detach().to(torch.float64).cpu().numpy()           # [B,M]
    alpha_np  = alpha_j.detach().to(torch.float64).cpu().numpy()         # [B,M]
    beta_np   = bessel_arg.detach().to(torch.float64).cpu().numpy()            # [B,M]
    qpos_np   = q_pos.detach().to(torch.float64).cpu().numpy()           # [B,M]
    gamma2_np = gamma_sq.detach().to(torch.float64).cpu().numpy()          # [B,M]

    Delta_np  = PI.cpu().numpy() * (gamma2_np[..., None] / np.clip(lam_np, tiny64, None) - qpos_np[..., None])  # [B,M,Q]
    mask_J    = (Delta_np > 0.0)
    mask_I    = ~mask_J

    # ============================================================
    #                 Branch I_ν : Gauss–Laguerre (log-domein)
    # ============================================================
    Q_RAD = 128

    alpha_L = 0.5 * NU
    u_nodes, w_nodes = roots_genlaguerre(Q_RAD, alpha_L)                 # [Qr]
    U_L   = u_nodes.reshape(1, 1, 1, Q_RAD)                               # [1,1,1,Qr]
    W_L   = w_nodes.reshape(1, 1, 1, Q_RAD)                               # [1,1,1,Qr]
    logW_L = np.log(np.clip(W_L, tiny64, None))                           # [1,1,1,Qr]

    Z_I = np.maximum(beta_t_np[..., None] * np.sqrt(U_L), tiny64)         # [B,M,Q,Qr]
    z_small = (Z_I <= 1e-12)
    log_I = np.empty_like(Z_I)
    ive_small = np.clip(ive(NU, np.maximum(Z_I[z_small], tiny64)), tiny64, None) if np.any(z_small) else None
    ive_big   = np.clip(ive(NU, np.maximum(Z_I[~z_small], tiny64)), tiny64, None) if np.any(~z_small) else None
    if np.any(z_small):
        log_I[z_small] = np.log(ive_small) + Z_I[z_small]
    if np.any(~z_small):
        log_I[~z_small] = np.log(ive_big) + Z_I[~z_small]

    beta2_over_4 = 0.25 * (beta_t_np ** 2)                                # [B,M,Q]
    log_phi_I   = log_I - beta2_over_4[..., None]                         # [B,M,Q,Qr]
    log_sum_u_I = logsumexp(logW_L + log_phi_I, axis=-1)                  # [B,M,Q]

    lam_cl   = np.clip(lam_np, tiny64, None)
    log_lam  = np.log(lam_cl)                                             # [B,M,Q]
    log_pref_I = math.log(0.5) - (NU / 2.0 + 1.0) * log_lam           # [B,M,Q]

    log_Ck = float(gammaln(NU + 1.0) - gammaln(NU + 0.5) - gammaln(0.5)) # Константа Куммера
    log_G_I = (log_Ck
               + np.log(np.clip(KD_np, tiny64, None))[..., None]
               + log_sum_u_I + log_pref_I
               + np.log(np.clip(wth_np, tiny64, None))
               + np.log(np.clip(Adir_np, tiny64, None))[..., None]
               + np.log(np.clip(alpha_np, tiny64, None))[..., None]
               )                                                          # [B,M,Q]
    # Mask I
    neg_inf = -1e300
    log_G_I_masked = np.where(mask_I, log_G_I, neg_inf)
    log_gain_I = logsumexp(log_G_I_masked, axis=-1)                        # [B,M]
    gain_I_np  = np.exp(np.clip(log_gain_I, a_min=np.log(tiny64), a_max=None))

    # ============================================================
    #                 Branch J_ν
    # ============================================================
    lamJ = lam_cl                                                          # [B,M,Q]
    beta2_over_4_abs = (beta_np[..., None] ** 2) / (4.0 * lamJ)            # [B,M,Q]
    coeffJ = (beta_np[..., None] ** NU) * np.power(2.0 * lamJ, -(NU + 1.0))  # [B,M,Q]
    R_J = coeffJ * np.exp(-np.clip(beta2_over_4_abs, 0.0, 7.0e2))          # [B,M,Q]

    log_Ck = float(gammaln(NU + 1.0) - gammaln(NU + 0.5) - gammaln(0.5))
    Ck = float(np.exp(log_Ck))

    G_J = (Ck * KD_np[..., None] * wth_np *
           Adir_np[..., None] * alpha_np[..., None] * R_J)                 # [B,M,Q]
    G_J_masked = np.where(mask_J, G_J, 0.0)
    gain_J_np  = np.sum(G_J_masked, axis=-1)                               # [B,M]

    # ============================================================
    #                Assembly gain_tail and T_tail
    # ============================================================
    gain_tail_np = gain_I_np + gain_J_np                                   # [B,M]
    gain_tail = torch.from_numpy(gain_tail_np).to(device=device, dtype=dtype_r)
    T_tail = (gain_tail.unsqueeze(-1) * T_hat_j).sum(dim=1)                # [B,S]

    if return_components == T_Omega_Components.TAIL:
        return T_tail
    elif return_components == T_Omega_Components.BOTH:
        return T_zero, T_tail
    elif return_components == T_Omega_Components.UNION:
        return T_zero + T_tail
    else:
        raise ValueError(f"Unknown mode: '{return_components=}'")
