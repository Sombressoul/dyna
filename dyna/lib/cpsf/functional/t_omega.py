import math
import torch
from dyna.lib.cpsf.functional.core_math import delta_vec_d

# z: [B,N] (complex)
# z_j: [B,M,N] (complex)
# vec_d: [B,N] (complex)
# vec_d_j: [B,M,N] (complex)
# T_hat_j: [B,M,S] (complex)
# alpha_j: [B,M] (real)
# sigma_par: [B,M] (real)
# sigma_perp: [B,M] (real)
# return_components: "zero" | "tail" | "both" | "union"; default = "union"
# k_cap: axial dual-mode cap (int >= 0)


def T_Omega(
    z: torch.Tensor,
    z_j: torch.Tensor,
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    T_hat_j: torch.Tensor,
    alpha_j: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
    return_components: str = "union",
    k_cap: int = 4,
) -> torch.Tensor:
    # Broadcast
    B, M, N = vec_d_j.shape
    z = z.unsqueeze(1).expand(B, M, N)
    vec_d = vec_d.unsqueeze(1).expand(B, M, N)

    # === Common (Zero-frame) ===
    x = z - z_j  # [B,M,N] complex
    xr, xi = x.real, x.imag  # [B,M,N]

    a = torch.reciprocal(sigma_perp)  # [B,M]
    b = torch.reciprocal(sigma_par) - a  # [B,M]
    den = (a + b).clamp_min(1e-30)  # [B,M]

    dr, di = vec_d_j.real, vec_d_j.imag  # [B,M,N]

    # q_pos: [B,M]
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

    gain_zero = alpha_j * A_pos * A_dir  # [B,M]
    T_zero = (gain_zero.unsqueeze(-1) * T_hat_j).sum(dim=1)  # [B,S]

    if return_components == "zero":
        return T_zero

    # === Tail: Poisson-dual axial residual (vectorized, O(B*M*N*k_cap)) ===
    # C0 = 1/sqrt(det(A_pos)) in 2N real dims: a^(N-1) * (a+b)
    C0 = (sigma_perp ** (x.shape[-1] - 1)) * sigma_par  # [B,M]

    # wrap POS to [-0.5,0.5): [B,M,N]
    x_re = torch.remainder(xr + 0.5, 1.0) - 0.5
    x_im = torch.remainder(xi + 0.5, 1.0) - 0.5

    # u_i^2 per axis in R^{2N} (normalized): [B,M,N] for Re and Im
    u_norm2 = (dr * dr + di * di).sum(dim=-1).clamp_min(1e-30)  # [B, M]
    inv_u_norm2 = torch.reciprocal(u_norm2).unsqueeze(-1)  # [B,M,1]
    u_re_sq = (dr * dr) * inv_u_norm2  # [B,M,N]
    u_im_sq = (di * di) * inv_u_norm2  # [B,M,N]

    # mu_axis = (A^{-1})_axis = 1/a - (b/(a*(a+b))) * u_axis^2
    gamma = (b / (a * den)).unsqueeze(-1)  # [B,M,1]
    inv_a = torch.reciprocal(a).unsqueeze(-1)  # [B,M,1]
    mu_re = (inv_a - gamma * u_re_sq).clamp_min(1e-12)  # [B,M,N]
    mu_im = (inv_a - gamma * u_im_sq).clamp_min(1e-12)  # [B,M,N]

    # axial k-modes m=1..k_cap, sum symmetrically (±m) -> 2*cos(...)
    m = torch.arange(1, k_cap + 1, device=z.device, dtype=mu_re.dtype)  # [K]
    m2 = (m * m).view(1, 1, 1, -1)  # [1,1,1,K]

    # [B,M,N,K]
    expo_re = torch.exp(-math.pi * mu_re.unsqueeze(-1) * m2)
    expo_im = torch.exp(-math.pi * mu_im.unsqueeze(-1) * m2)

    # [B,M,N,K]
    phase_re = torch.cos(2.0 * math.pi * x_re.unsqueeze(-1) * m.view(1, 1, 1, -1))
    phase_im = torch.cos(2.0 * math.pi * x_im.unsqueeze(-1) * m.view(1, 1, 1, -1))

    sum_re = (expo_re * phase_re).sum(dim=-1)  # [B,M,N]
    sum_im = (expo_im * phase_im).sum(dim=-1)  # [B,M,N]

    # Clamp to avoid negative from truncation
    tail_re = 2 * sum_re.clamp_min(0.0)  # [B,M,N]
    tail_im = 2 * sum_im.clamp_min(0.0)  # [B,M,N]

    tail_axis = torch.cat((tail_re, tail_im), dim=-1)  # [B,M,2N]

    one_plus_tail = 1.0 + tail_axis  # [B,M,2N]

    # prod stable: use log for large N
    log_one_plus = torch.log(one_plus_tail.clamp_min(1e-30))  # [B,M,2N]
    sum_log = log_one_plus.sum(dim=-1)  # [B,M]
    prod = torch.exp(sum_log)  # [B,M]

    Theta_approx = C0 * prod  # [B,M]

    F = (Theta_approx - A_pos).clamp_min(0.0)  # [B,M]

    # Tail_j = alpha_j * A_dir * F
    gain_tail = alpha_j * A_dir * F  # [B,M]
    T_tail = (gain_tail.unsqueeze(-1) * T_hat_j).sum(dim=1)  # [B,S]

    if return_components == "tail":
        return T_tail
    elif return_components == "both":
        return T_zero, T_tail
    elif return_components == "union":
        return T_zero + T_tail
    else:
        return T_zero
