import torch, math

from typing import Optional

from dyna.lib.cpsf.functional.core_math import (
    lift,
    delta_vec_d,
    R,
    R_ext,
    iota,
    q,
)


def T_PD_window(
    *,
    z: torch.Tensor,
    z_j: torch.Tensor,
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    T_hat_j: torch.Tensor,
    alpha_j: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
    offsets: torch.Tensor,
    t: float = 1.0,  # Poisson/Ewald scale, t > 0
    R_j: Optional[torch.Tensor] = None,
    q_max: Optional[float] = None,  # Warning: introduces positive bias
) -> torch.Tensor:
    if t <= 0.0:
        raise ValueError("T_positional_dual_window: t must be > 0.")

    r_dtype = z.real.dtype
    B, M, N = vec_d_j.shape

    z = z.unsqueeze(1).expand(B, M, N)
    vec_d = vec_d.unsqueeze(1).expand(B, M, N)

    dz = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)

    Rmat = R(vec_d_j) if R_j is None else R_j

    k_r = offsets[:, :N].to(r_dtype)
    k_i = offsets[:, N:].to(r_dtype)
    k_c = torch.complex(k_r, k_i)
    RH = Rmat.conj().transpose(-2, -1)
    y = torch.einsum("bmnk,ok->bmon", RH, k_c)
    y_abs2 = y.real**2 + y.imag**2
    s = y_abs2.sum(dim=-1)
    p_abs2 = y_abs2[..., 0]
    quad_k = (sigma_perp.unsqueeze(-1) * s) + (
        (sigma_par - sigma_perp).unsqueeze(-1) * p_abs2
    )
    dot = (k_r.unsqueeze(0).unsqueeze(0) * dz.real.unsqueeze(2)).sum(dim=-1) + (
        k_i.unsqueeze(0).unsqueeze(0) * dz.imag.unsqueeze(2)
    ).sum(dim=-1)
    phase = torch.exp(2j * math.pi * dot)
    prefac = (sigma_par * (sigma_perp ** (N - 1))) / (t**N)
    weight = torch.exp(-(math.pi / t) * quad_k) * phase
    Theta_pos = (prefac.unsqueeze(-1) * weight).sum(dim=-1)

    Rext = R_ext(Rmat)
    zeros_u = torch.zeros_like(dd)
    w_dir = iota(zeros_u, dd)
    q_dir = q(
        w=w_dir,
        R_ext=Rext,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
    )

    if q_max is not None:
        q_cap = torch.as_tensor(q_max, dtype=q_dir.dtype, device=q_dir.device)
        q_dir = torch.clamp(q_dir, max=q_cap)

    C_dir_j = torch.exp(-math.pi * q_dir)
    eta_j = C_dir_j * Theta_pos
    w = (alpha_j.to(T_hat_j.real.dtype) * eta_j.real).unsqueeze(-1)
    T = w * T_hat_j
    T_out = T.sum(dim=1)

    return T_out
