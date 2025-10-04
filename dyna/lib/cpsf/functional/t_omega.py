import torch

from typing import Tuple, Optional

from enum import Enum, auto as enum_auto

from dyna.lib.cpsf.functional.core_math import delta_vec_d


class T_Omega_Components(Enum):
    ZERO = enum_auto()
    TAIL = enum_auto()
    BOTH = enum_auto()
    UNION = enum_auto()

import torch


def theta3_log(
    *,
    z: torch.Tensor,  # [...], real or complex
    q: torch.Tensor,  # [...], real in (0,1), broadcastable to z
    m: torch.Tensor,  # [M], real
    device: torch.device,  # target device
    dtype_r: torch.dtype,  # real dtype
    dtype_c: torch.dtype,  # complex dtype
    shift: float = 1.18e-38,  # small shift to avoid log(0)
) -> torch.Tensor:
    # move to device/dtype
    zc = z.to(device=device, dtype=dtype_c)
    qr = q.to(device=device, dtype=dtype_r)
    m = m.to(device=device, dtype=dtype_r)

    # 2*pi and phases
    two_pi = torch.tensor(2.0 * torch.pi, device=device, dtype=dtype_r)
    theta = two_pi.to(dtype=dtype_c) * zc  # [...], complex

    # cos for all m: shape [..., M]
    cos_m_theta = torch.cos(theta.unsqueeze(-1) * m.to(dtype=dtype_c))  # [..., M], complex

    # q^{m^2}
    q_b, _ = torch.broadcast_tensors(qr, zc.real)  # [...], real
    m2 = m * m  # [M], real
    q_pow = torch.pow(q_b.unsqueeze(-1), m2)  # [..., M], real
    q_pow_c = q_pow.to(dtype=dtype_c)  # [..., M], complex

    # series sum and log
    one_c = torch.ones((), device=device, dtype=dtype_c)
    S = (one_c * (1.0 + float(shift))) + 2.0 * torch.sum(q_pow_c * cos_m_theta, dim=-1)  # [...], complex

    return torch.log(S)


def hermite_gauss(
    *,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = torch.arange(1, n, device=device, dtype=dtype)
    beta = torch.sqrt(k * 0.5)

    J = torch.zeros((n, n), device=device, dtype=dtype)
    J.diagonal(1).copy_(beta)
    J.diagonal(-1).copy_(beta)

    evals, evecs = torch.linalg.eigh(J)
    x = evals
    mu_0 = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))
    w = mu_0 * (evecs[0, :] ** 2)

    return x, w


def T_Omega(
    *,
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
    gh_order: int = 16, # Hermite-Gauss order
    theta_series_tol: Optional[float] = None,  # theta-3 series truncation
    theta_series_m_cap: int = 512,  # theta-3 series elements cap
) -> torch.Tensor:
    # ============================================================
    # SIMPLE GUARDS
    # ============================================================
    assert int(gh_order) >= 1, "CPSF/T_Omega requires gh_order >= 1."
    assert int(theta_series_m_cap) >= 1, "CPSF/T_Omega requires theta_series_m_cap >= 1."

    # ============================================================
    # HEAVY GUARDS
    # ============================================================
    if bool(guards):
        assert (alpha_j > 0).all(), "CPSF/T_Omega requires alpha_j > 0."
        assert (sigma_par > 0).all(), "CPSF/T_Omega requires sigma_par > 0."
        assert (sigma_perp > 0).all(), "CPSF/T_Omega requires sigma_perp > 0."

    # ============================================================
    # VARIABLES
    # ============================================================
    GH_Q = int(gh_order)
    M_CAP = int(theta_series_m_cap)

    # ============================================================
    # BASE
    # ============================================================
    device = z.device
    dtype_r = z.real.dtype
    dtype_c = z.dtype
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
    PI = torch.tensor(torch.pi, dtype=dtype_r, device=device)
    INV_SQRT_PI_C = (1.0 / torch.sqrt(PI)).to(dtype=dtype_c)

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
    # PREPARE AND EXECUTE THETA-3  (1D-HS integral, memory-lean)
    # ============================================================
    # 1) Gauss–Hermite nodes/weights (measure e^{-y^2})
    gh_n, gh_w = hermite_gauss(n=GH_Q, device=device, dtype=dtype_r)  # n:[K], w:[K]
    K = gh_n.shape[0]

    # 2) HS shifts: s = s_scale * y; xi = (s / 2pi) for osc, i*(s / 2pi) for hyp
    s = s_scale.unsqueeze(-1) * gh_n.view(1, 1, K)  # [B,M,K], real
    xi_real = xi_scale * s  # [B,M,K], real
    osc_mask = is_osc.unsqueeze(-1).to(dtype_r)  # [B,M,1], real {0,1}
    xi_c = torch.complex(xi_real * osc_mask, xi_real * (1.0 - osc_mask)) # [B,M,K], complex

    # 3) Theta arguments: z_arg = x + xi(s) * u
    x_arg_r = x_wrapped_R2N.unsqueeze(-1)  # [B,M,2N,1], real
    u_arg_r = u_R2N.unsqueeze(-1)  # [B,M,2N,1], real
    xi_arg_c = xi_c.unsqueeze(-2)  # [B,M,1,K],  complex
    z_arg = x_arg_r + u_arg_r * xi_arg_c  # [B,M,2N,K], complex (via promotion)

    # 4) Theta-3 execution
    # 4.1) Per-batch optimal m_vec
    tolerance = float(theta_series_tol) if (theta_series_tol is not None) else float(eps)
    q_max = torch.amax(q)
    q_max = torch.clamp(q_max, max=1.0 - eps)
    ln_q   = torch.log(q_max)
    target = torch.log(torch.tensor(tolerance, dtype=dtype_r, device=device)) - torch.log(torch.tensor(4.0, dtype=dtype_r, device=device))
    m_max_f = torch.sqrt(target / ln_q)
    m_max_i = int(torch.clamp(torch.ceil(m_max_f), min=torch.tensor(0.0, dtype=dtype_r, device=device)).item())
    m_max_i = max(0, min(M_CAP, m_max_i))
    m_vec = torch.arange(1, m_max_i + 1, device=device, dtype=dtype_r)    

    # 4.2) Batched theta3 (single shot over axes and GH nodes)
    log_theta = theta3_log(
        z=z_arg,
        q=q,
        m=m_vec,
        device=device,
        dtype_r=dtype_r,
        dtype_c=dtype_c,
        shift=float(tiny),
    )  # [B,M,2N,K], complex

    # 5) Reduce: sum over 2N axes, GH-weighted sum over K
    L = torch.sum(log_theta, dim=-2)  # [B,M,K], complex
    prod_theta_minus_1 = torch.exp(L) - 1.0  # [B,M,K], complex
    prod_theta_minus_1.mul_(gh_w.view(1, 1, K))  # in-place weight multiply
    I_tail_c = INV_SQRT_PI_C * torch.sum(prod_theta_minus_1, dim=-1)  # [B,M], complex
    I_tail = I_tail_c.real  # [B,M], real

    # ============================================================
    # ASSEMBLY TAIL
    # ============================================================
    gauss_dim_pref = sigma_par_clamped * torch.pow(sigma_perp_clamped, float(N - 1))  # [B,M]
    gain_tail = alpha_j * A_dir * gauss_dim_pref * I_tail  # [B,M]

    T_tail = (gain_tail.unsqueeze(-1) * T_hat_j).sum(dim=1)  # [B,S]

    if return_components == T_Omega_Components.TAIL:
        return T_tail
    elif return_components == T_Omega_Components.BOTH:
        return T_zero, T_tail
    elif return_components == T_Omega_Components.UNION:
        return T_zero + T_tail
    else:
        raise ValueError(f"Unknown mode: '{return_components=}'")
