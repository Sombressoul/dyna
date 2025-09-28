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
    #                      BASE
    # ============================================================
    device = z.device
    dtype_c = z.dtype
    dtype_r = z.real.dtype

    # ============================================================
    #                      MAIN
    # ============================================================
    # Broadcast
    B, M, N = vec_d_j.shape
    z = z.unsqueeze(1).expand(B, M, N)
    vec_d = vec_d.unsqueeze(1).expand(B, M, N)

    # Common
    x = z - z_j  # [B,M,N] complex
    xr, xi = x.real, x.imag  # [B,M,N]
    a = torch.reciprocal(sigma_perp)  # [B,M]
    b = torch.reciprocal(sigma_par) - a  # [B,M]

    # ============================================================
    #                      ZERO-FRAME
    # ============================================================
    # q_pos: [B,M]
    dr, di = vec_d_j.real, vec_d_j.imag  # [B,M,N]
    norm2_x = (xr * xr + xi * xi).sum(dim=-1)
    inner_re = (dr * xr + di * xi).sum(dim=-1)
    inner_im = (dr * xi - di * xr).sum(dim=-1)
    inner_abs2 = inner_re * inner_re + inner_im * inner_im
    q_pos = a * norm2_x + b * inner_abs2
    A_pos = torch.exp(-math.pi * q_pos)  # [B,M]

    # A_dir: [B,M]
    dv = delta_vec_d(vec_d, vec_d_j)  # [B,M,N] complex
    dvr, dvi = dv.real, dv.imag
    norm2_dv = (dvr * dvr + dvi * dvi).sum(dim=-1)
    A_dir = torch.exp(-math.pi * torch.reciprocal(sigma_perp) * norm2_dv)  # [B,M]

    # Gain
    gain_zero = alpha_j * A_pos * A_dir  # [B,M]
    T_zero = (gain_zero.unsqueeze(-1) * T_hat_j).sum(dim=1)  # [B,S]

    if return_components == T_Omega_Components.ZERO:
        return T_zero

    # ============================================================
    #                      TAIL  (HYBRID)
    # ============================================================
    D = 2 * N
    c_val = float(N)
    nu_val = float(N - 1)

    tiny_f32 = torch.finfo(a.dtype).tiny

    norm2_dj = (dr * dr + di * di).sum(dim=-1)                      # [B,M]
    inv_norm_dj = torch.rsqrt(torch.clamp(norm2_dj, min=tiny_f32))  # [B,M]
    ur = dr * inv_norm_dj.unsqueeze(-1)                             # [B,M,N]
    ui = di * inv_norm_dj.unsqueeze(-1)                             # [B,M,N]

    c_re = (ur * xr + ui * xi).sum(dim=-1)                          # [B,M]
    c_im = (ur * xi - ui * xr).sum(dim=-1)                          # [B,M]
    inner_abs2_u = c_re * c_re + c_im * c_im                        # [B,M]

    kappa = b / torch.clamp(a, min=tiny_f32)                        # [B,M]
    onepk = 1.0 + kappa                                             # [B,M]

    vr = a.unsqueeze(-1) * xr + b.unsqueeze(-1) * (c_re.unsqueeze(-1) * ur - c_im.unsqueeze(-1) * ui)
    vi = a.unsqueeze(-1) * xi + b.unsqueeze(-1) * (c_re.unsqueeze(-1) * ui + c_im.unsqueeze(-1) * ur)
    norm2_v = (vr * vr + vi * vi).sum(dim=-1)                       # [B,M]
    gamma2 = torch.clamp(norm2_v / torch.clamp(a, min=tiny_f32), min=0.0)  # [B,M]

    K_D = (2.0 ** nu_val) * torch.pow(torch.clamp(a, min=tiny_f32), -c_val)  # [B,M]
    beta = (2.0 * math.sqrt(math.pi)) * torch.sqrt(gamma2)          # [B,M]

    def _stat(name, t):
        tc = t.detach().to("cpu")
        fin = tc[torch.isfinite(tc)]
        if fin.numel() == 0:
            print(f"[DERIV][{name}] no finite values")
        else:
            print(f"[DERIV][{name}] min={fin.min().item():.3e} max={fin.max().item():.3e} mean={fin.mean().item():.3e}")

    print(f"[DERIV][const] D={D}, c={c_val:.1f}, nu={nu_val:.1f}")
    _stat("||d_j||^2", norm2_dj)
    _stat("kappa=b/a", kappa); _stat("1+kappa", onepk)
    _stat("|<x,u>|^2", inner_abs2_u)
    _stat("gamma2 (via v)", gamma2)
    _stat("beta=2√πγ", beta)
    _stat("K_D", K_D)

    # ============================================================
    #                JACOBI
    # ============================================================
    Q_THETA = 24
    alpha_old = -0.5
    beta_old  = float(nu_val - 0.5)

    x_jac, w_jac = _t_omega_roots_jacobi(
        N=Q_THETA,
        alpha=alpha_old,
        beta=beta_old,
        normalize=True,
        return_weights=True,
        dtype=dtype_c,
        device=device,
    )

    # Константа Куммера
    log_Ck = float(gammaln(nu_val + 1.0) - gammaln(nu_val + 0.5) - gammaln(0.5))
    Ck = float(np.exp(log_Ck))

    t_theta_bm = x_jac.view(1, 1, -1)                                              # [1,1,Q]
    w_theta_bm = w_jac.view(1, 1, -1)                                              # [1,1,Q]

    one64 = torch.tensor(1.0, dtype=torch.float64, device=device)
    lam_theta = one64 + kappa.to(torch.float64)[..., None] * (one64 - t_theta_bm)    # [B,M,Q]
    lam_theta = torch.clamp(lam_theta, min=torch.finfo(torch.float64).tiny)
    beta_theta = beta.to(torch.float64)[..., None] / torch.sqrt(lam_theta)           # [B,M,Q]

    print(f"[JACOBI][params] Q={Q_THETA}, α_old={alpha_old:.3f}, β_old={beta_old:.3f}")


    # ============================================================
    #         Masks
    # ============================================================
    dev = z.device
    tiny64 = np.finfo(np.float64).tiny

    lam_np    = lam_theta.detach().cpu().numpy()                         # [B,M,Q]
    beta_t_np = beta_theta.detach().cpu().numpy()                        # [B,M,Q]
    KD_np     = K_D.detach().to(torch.float64).cpu().numpy()             # [B,M]
    wth_np    = w_theta_bm.detach().cpu().numpy()                        # [1,1,Q]
    Adir_np   = A_dir.detach().to(torch.float64).cpu().numpy()           # [B,M]
    alpha_np  = alpha_j.detach().to(torch.float64).cpu().numpy()         # [B,M]
    beta_np   = beta.detach().to(torch.float64).cpu().numpy()            # [B,M]
    qpos_np   = q_pos.detach().to(torch.float64).cpu().numpy()           # [B,M]
    gamma2_np = gamma2.detach().to(torch.float64).cpu().numpy()          # [B,M]

    Delta_np  = np.pi * (gamma2_np[..., None] / np.clip(lam_np, tiny64, None) - qpos_np[..., None])  # [B,M,Q]
    mask_J    = (Delta_np > 0.0)
    mask_I    = ~mask_J

    def _stat_np(name, arr):
        t = torch.from_numpy(arr).to(dev, dtype=torch.float64)
        fin = t[torch.isfinite(t)]
        if fin.numel() == 0:
            print(f"[HYBRID][{name}] no finite values")
        else:
            print(f"[HYBRID][{name}] min={fin.min().item():.3e} max={fin.max().item():.3e} mean={fin.mean().item():.3e}")

    _stat_np("Δ(t)", Delta_np)
    print(f"[HYBRID][counts] I-branch nodes: {int(mask_I.sum())},  J-branch nodes: {int(mask_J.sum())}")

    # ============================================================
    #                 Branch I_ν : Gauss–Laguerre (log-domein)
    # ============================================================
    Q_RAD = 128
    print(f"[HYBRID][I-branch] Q_RAD={Q_RAD}, nu={nu_val:.1f}")

    alpha_L = 0.5 * nu_val
    u_nodes, w_nodes = roots_genlaguerre(Q_RAD, alpha_L)                 # [Qr]
    U_L   = u_nodes.reshape(1, 1, 1, Q_RAD)                               # [1,1,1,Qr]
    W_L   = w_nodes.reshape(1, 1, 1, Q_RAD)                               # [1,1,1,Qr]
    logW_L = np.log(np.clip(W_L, tiny64, None))                           # [1,1,1,Qr]

    Z_I = np.maximum(beta_t_np[..., None] * np.sqrt(U_L), tiny64)         # [B,M,Q,Qr]
    z_small = (Z_I <= 1e-12)
    log_I = np.empty_like(Z_I)
    ive_small = np.clip(ive(nu_val, np.maximum(Z_I[z_small], tiny64)), tiny64, None) if np.any(z_small) else None
    ive_big   = np.clip(ive(nu_val, np.maximum(Z_I[~z_small], tiny64)), tiny64, None) if np.any(~z_small) else None
    if np.any(z_small):
        log_I[z_small] = np.log(ive_small) + Z_I[z_small]
    if np.any(~z_small):
        log_I[~z_small] = np.log(ive_big) + Z_I[~z_small]

    beta2_over_4 = 0.25 * (beta_t_np ** 2)                                # [B,M,Q]
    log_phi_I   = log_I - beta2_over_4[..., None]                         # [B,M,Q,Qr]
    log_sum_u_I = logsumexp(logW_L + log_phi_I, axis=-1)                  # [B,M,Q]

    lam_cl   = np.clip(lam_np, tiny64, None)
    log_lam  = np.log(lam_cl)                                             # [B,M,Q]
    log_pref_I = math.log(0.5) - (nu_val / 2.0 + 1.0) * log_lam           # [B,M,Q]

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

    _stat_np("I: log Σ_u (α=ν/2)", log_sum_u_I)
    _stat_np("I: log G(t)", log_G_I)
    _stat_np("I: log gain_I", log_gain_I)

    # ============================================================
    #                 Branch J_ν
    # ============================================================
    print(f"[HYBRID][J-branch] closed-form, nu={nu_val:.1f}")

    lamJ = lam_cl                                                          # [B,M,Q]
    beta2_over_4_abs = (beta_np[..., None] ** 2) / (4.0 * lamJ)            # [B,M,Q]
    coeffJ = (beta_np[..., None] ** nu_val) * np.power(2.0 * lamJ, -(nu_val + 1.0))  # [B,M,Q]
    R_J = coeffJ * np.exp(-np.clip(beta2_over_4_abs, 0.0, 7.0e2))          # [B,M,Q]

    G_J = (Ck * KD_np[..., None] * wth_np *
           Adir_np[..., None] * alpha_np[..., None] * R_J)                 # [B,M,Q]
    G_J_masked = np.where(mask_J, G_J, 0.0)
    gain_J_np  = np.sum(G_J_masked, axis=-1)                               # [B,M]

    _stat_np("J: coeffJ", coeffJ)
    _stat_np("J: exp(-β^2/(4λ))", np.exp(-np.clip(beta2_over_4_abs, 0.0, 7.0e2)))
    _stat_np("J: R_J(t)", R_J)
    _stat_np("J: G_J(t)", G_J)
    _stat_np("J: gain_J", gain_J_np)

    # ============================================================
    #                Assembly gain_tail and T_tail
    # ============================================================
    gain_tail_np = gain_I_np + gain_J_np                                   # [B,M]
    gain_tail = torch.from_numpy(gain_tail_np).to(device=device, dtype=torch.float64).to(a.dtype)
    T_tail = (gain_tail.unsqueeze(-1) * T_hat_j).sum(dim=1)                # [B,S]

    def _stat_t(name, t):
        fin = t[torch.isfinite(t)]
        if fin.numel() == 0:
            print(f"[HYBRID][{name}] no finite values")
        else:
            print(f"[HYBRID][{name}] min={fin.min().item():.3e} max={fin.max().item():.3e} mean={fin.mean().item():.3e}")
    _stat_np("gain_tail (numpy)", gain_tail_np)
    _stat_t("gain_tail", gain_tail)
    _stat_t("T_tail.re", T_tail.real)
    _stat_t("T_tail.im", T_tail.imag)


    if return_components == T_Omega_Components.TAIL:
        return T_tail
    elif return_components == T_Omega_Components.BOTH:
        return T_zero, T_tail
    elif return_components == T_Omega_Components.UNION:
        return T_zero + T_tail
    else:
        raise ValueError(f"Unknown mode: '{return_components=}'")
