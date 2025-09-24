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


def T_PD_window_dual(
    *,
    z: torch.Tensor,
    z_j: torch.Tensor,
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    T_hat_j: torch.Tensor,
    alpha_j: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
    offsets: torch.Tensor,  # Dual-space k modes on Z^{2N}
    t: float = 1.0,  # Poisson/Ewald scale, for any t>0 - exact
    R_j: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if t <= 0.0:
        raise ValueError("T_PD_window: t must be > 0.")

    device = z.device
    r_dtype = z.real.dtype
    B, M, N = vec_d_j.shape

    # Broadcast inputs
    z = z.unsqueeze(1).expand(B, M, N)
    vec_d = vec_d.unsqueeze(1).expand(B, M, N)

    # Geometric deltas
    dz = lift(z) - lift(z_j)  # [B,M,N] complex
    dd = delta_vec_d(vec_d, vec_d_j)  # [B,M,N] complex

    # Frames
    Rmat = R(vec_d_j) if R_j is None else R_j
    RH = Rmat.conj().transpose(-2, -1)  # [B,M,N,N]
    Rext = R_ext(Rmat)

    # Const
    pi = torch.tensor(math.pi, dtype=r_dtype, device=device)

    # Directional Gaussian
    zeros_u = torch.zeros_like(dd)  # [B,M,N]
    w_dir = iota(zeros_u, dd)  # [B,M,N]
    q_dir = q(
        w=w_dir,
        R_ext=Rext,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
    )
    C_dir_j = torch.exp(-pi * q_dir)  # [B,M]

    # k on Z^{2N}
    k_r = offsets[:, :N].to(device=device, dtype=r_dtype)  # [O, N]
    k_i = offsets[:, N:].to(device=device, dtype=r_dtype)  # [O, N]
    k_c = torch.complex(k_r, k_i)  # [O, N]

    # y = R^H k
    y = torch.einsum("bmnk,ok->bmon", RH, k_c)  # [B,M,O,N]
    y_abs2 = y.real**2 + y.imag**2
    s = y_abs2.sum(dim=-1)  # [B,M,O]
    p_abs2 = y_abs2[..., 0]  # [B,M,O]
    quad_k = (sigma_perp.unsqueeze(-1) * s) + (
        (sigma_par - sigma_perp).unsqueeze(-1) * p_abs2
    )  # [B,M,O]

    # Phase
    dot = (
        k_r.unsqueeze(0).unsqueeze(0) * torch.remainder(dz.real, 1.0).unsqueeze(2)
    ).sum(dim=-1) + (
        k_i.unsqueeze(0).unsqueeze(0) * torch.remainder(dz.imag, 1.0).unsqueeze(2)
    ).sum(
        dim=-1
    )
    ang = (2.0 * pi).to(dot.dtype) * dot
    phase = torch.polar(torch.ones_like(ang), ang)

    # Prefactor: det(Sigma_pos) / t^N  (complex-N convention)
    prefac = (sigma_par * (sigma_perp ** (N - 1))) / (
        t**N
    )  # [B, M] - No sqrt! C^{N}, not R^{2N}
    weight = torch.exp(-(pi / t) * quad_k) * phase  # [B,M,O]
    Theta_pos = (prefac.unsqueeze(-1) * weight).sum(dim=-1)  # [B,M]

    eta_j = (C_dir_j * Theta_pos).real.to(T_hat_j.real.dtype)  # [B,M]
    w = (alpha_j.to(T_hat_j.real.dtype) * eta_j).unsqueeze(-1)  # [B,M,1]
    T = w * T_hat_j  # [B,M,S]
    T_out = T.sum(dim=1)  # [B,S]

    return T_out
