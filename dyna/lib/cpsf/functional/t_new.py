import torch
import math

from typing import List

from dyna.lib.cpsf.functional.core_math import delta_vec_d, R, R_ext

def get_L(a: float, error: float, d: int, max_L: int = 16, m_cap: int = 512) -> int:
    if a <= 0 or error <= 0:
        raise ValueError("a and error must be positive")
    log_error = math.log(error)
    L = 0
    while True:
        log_tail_terms: List[float] = []
        m = L + 1
        while True:
            if m == 0:
                m += 1
                continue
            log_A = math.log(d) + (d - 1) * math.log(2 * m)
            log_term = log_A - a * m * m
            log_tail_terms.append(log_term)
            if log_term < log_error - math.log(10**10):  # term < error / 10^10
                break
            m += 1
            if m > m_cap:
                break
        if log_tail_terms:
            max_log = max(log_tail_terms)
            sum_exp = sum(math.exp(lt - max_log) for lt in log_tail_terms)
            log_tail = max_log + math.log(sum_exp)
            if log_tail < log_error:
                return L
        L += 1
        if L > max_L:
            return max_L

def T_New(
    *,
    z: torch.Tensor,
    z_j: torch.Tensor,
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    T_hat_j: torch.Tensor,
    alpha_j: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
    error_budget: float = 1.0e-5,
) -> torch.Tensor:
    device = z.device
    dtype_r = z.real.dtype
    tiny = torch.finfo(dtype_r).tiny

    B, M, N = vec_d_j.shape
    z = z.unsqueeze(1).expand(B, M, N)
    vec_d = vec_d.unsqueeze(1).expand(B, M, N)

    PI = torch.tensor(math.pi, dtype=dtype_r, device=device)

    dz = z - z_j
    frac_re = torch.remainder(dz.real, 1.0)
    frac_im = torch.remainder(dz.imag, 1.0)

    Rmat = R(vec_d_j)
    RH = Rmat.conj().transpose(-2, -1)

    delta_d = delta_vec_d(vec_d, vec_d_j)
    delta_d_norm_sq = (delta_d.real**2 + delta_d.imag**2).sum(dim=-1)
    sigma_q_inv = torch.reciprocal(sigma_perp.clamp(min=tiny))
    C_dir_j = torch.exp(-PI * sigma_q_inv * delta_d_norm_sq)

    t = 1.0

    sigma_min = min(sigma_par.min().item(), sigma_perp.min().item())
    a = math.pi / t * sigma_min
    d = 2 * N
    L = get_L(a, error_budget, d)

    prefac = sigma_par * (sigma_perp ** (N - 1)) * (t ** (-N))

    if L == 0:
        Theta_pos = prefac
    else:
        ranges = [torch.arange(-L, L + 1, dtype=dtype_r, device=device) for _ in range(d)]
        grids = torch.meshgrid(*ranges, indexing='ij')
        offsets = torch.stack(grids, dim=-1).reshape(-1, d)  # [O, d]

        k_r = offsets[:, :N]
        k_i = offsets[:, N:]
        k_c = torch.complex(k_r, k_i)

        y = torch.einsum("bmnk,ok->bmon", RH, k_c)

        y_abs2 = y.real**2 + y.imag**2
        s = y_abs2.sum(dim=-1)
        p_abs2 = y_abs2[..., 0]
        quad_k = sigma_perp.unsqueeze(-1) * s + (sigma_par - sigma_perp).unsqueeze(-1) * p_abs2

        exp_k = torch.exp(-(PI / t) * quad_k)

        dot = (k_r.unsqueeze(0).unsqueeze(0) * frac_re.unsqueeze(2)).sum(dim=-1) + \
              (k_i.unsqueeze(0).unsqueeze(0) * frac_im.unsqueeze(2)).sum(dim=-1)
        ang = 2 * PI * dot
        phase = torch.complex(torch.cos(ang), torch.sin(ang))

        weight = exp_k * phase
        Theta_pos = (prefac.unsqueeze(-1) * weight).sum(dim=-1)

    eta_j = C_dir_j * Theta_pos
    gain = alpha_j * eta_j.real.to(T_hat_j.dtype)
    T = (gain.unsqueeze(-1) * T_hat_j).sum(dim=1)

    return T