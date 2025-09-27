import math
import torch

from enum import Enum, auto as enum_auto

from dyna.lib.cpsf.functional.core_math import delta_vec_d
from dyna.lib.cpsf.functional.t_omega_specials import (
    _get_real_dtype,
    _gauss_jacobi_01_nodes,
    _precompute_gauss_jacobi_for_1f1_half,
    _hyp1f1_half_negx_gauss_jacobi,
)

# ============================================================
#                      NOTES
# ============================================================
#
# Offline:
#   cache = _precompute_gauss_jacobi_for_1f1_half(N=N, Q_t=8, dtype=torch.float32, device=z.device)
#
# Online:
#   y_1f1 = _hyp1f1_half_negx_gauss_jacobi(x=x, N=N, Q_t=8, dtype=z.real.dtype, device=z.device, cache=cache)
#
# ============================================================


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
    #                      TAIL
    # ============================================================
    T_tail = torch.zeros_like(T_zero)  # Placeholder.

    if return_components == T_Omega_Components.TAIL:
        return T_tail
    elif return_components == T_Omega_Components.TAIL:
        return T_zero, T_tail
    elif return_components == T_Omega_Components.UNION:
        return T_zero + T_tail
    else:
        raise ValueError(f"Unknown mode: '{return_components=}'")
