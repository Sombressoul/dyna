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
    z: torch.Tensor,  # [B, N] - complex
    z_j: torch.Tensor,  # [B, M, N] - complex
    vec_d: torch.Tensor,  # [B, N] - (same dtype as z)
    vec_d_j: torch.Tensor,  # [B, M, N] - (same dtype as z)
    T_hat_j: torch.Tensor,  # [B, M, S] - complex (contributes to final T)
    alpha_j: torch.Tensor,  # [B, M] - real
    sigma_par: torch.Tensor,  # [B, M] - real > 0
    sigma_perp: torch.Tensor,  # [B, M] - real > 0
    offsets: torch.Tensor,  # [O, 2N] - long (dual lattice indices for position only)
    t: float = 1.0,  # Poisson/Ewald scale, t > 0
    R_j: Optional[
        torch.Tensor
    ] = None,  # [B, M, N, N] complex (optional precomputed R(vec_d_j))
) -> torch.Tensor:
    if t <= 0.0:
        raise ValueError("T_positional_dual_window: t must be > 0.")

    r_dtype = z.real.dtype

    B, M, N = vec_d_j.shape
    O, twoN = offsets.shape

    assert (
        twoN == 2 * N
    ), "k_offsets must have shape [O, 2N] with 2N matching the complex positional dimension."

    z = z.unsqueeze(1).expand(B, M, N)
    vec_d = vec_d.unsqueeze(1).expand(B, M, N)

    # -- 1) Canonical displacements (lifted), and directional delta:
    dz = lift(z) - lift(z_j)  # [B, M, N] complex
    dd = delta_vec_d(vec_d, vec_d_j)  # [B, M, N] complex

    # -- 2) Geometry R(vec_d_j) and its block extension:
    Rmat = R(vec_d_j) if R_j is None else R_j  # [B, M, N, N] complex
    Rext = R_ext(Rmat)  # [B, M, 2N, 2N] complex  (for q_dir via core q())

    # -- 3) Directional multiplier C_dir_j = exp(-pi * q_dir) using canonical q():
    zeros_u = torch.zeros_like(dd)  # [B, M, N] complex
    w_dir = iota(zeros_u, dd)  # [B, M, 2N] complex
    q_dir = q(
        w=w_dir,
        R_ext=Rext,  # [...,] real >= 0
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
    )
    C_dir_j = torch.exp(-math.pi * q_dir)  # [B, M] real in (0,1]

    # -- 4) Dual (Poisson) positional theta-sum:
    # k in Z^{2N} encodes [k_R, k_I] for the positional complex coordinates
    k_r = offsets[:, :N].to(r_dtype)  # [O, N] real
    k_i = offsets[:, N:].to(r_dtype)  # [O, N] real
    k_c = torch.complex(k_r, k_i)  # [O, N] complex

    # 4a) Quadratic form k^T A_pos^{-1} k  with A_pos^{-1} = R * diag(sp, sq,...) * R^H
    #     Use rank-1 structure:  sq * ||y||^2 + (sp - sq) * |y_1|^2, where y = R^H k_c and y_1 is its first component.
    RH = Rmat.conj().transpose(-2, -1)  # [B, M, N, N]
    # y = torch.einsum("bmnk,ok->bmno", RH, k_c)  # [B, M, O, N]  y = R^H k
    y = torch.einsum("bmnk,ok->bmon", RH, k_c)  # [B, M, O, N]
    y_abs2 = y.real**2 + y.imag**2  # [B, M, O, N]
    s = y_abs2.sum(dim=-1)  # [B, M, O] = ||y||^2
    p_abs2 = y_abs2[..., 0]  # [B, M, O] = |<k, r1>|^2

    sp = sigma_par  # [B, M]
    sq = sigma_perp  # [B, M]
    quad_k = (sq.unsqueeze(-1) * s) + ((sp - sq).unsqueeze(-1) * p_abs2)  # [B, M, O]

    # 4b) Phase term exp(2pi i * k · b_pos), where b_pos = dz (Re/Im split)
    #     k · b_pos = sum_n k_R * Re(dz) + k_I * Im(dz)
    dot = (k_r.unsqueeze(0).unsqueeze(0) * dz.real.unsqueeze(2)).sum(dim=-1) + (
        k_i.unsqueeze(0).unsqueeze(0) * dz.imag.unsqueeze(2)
    ).sum(
        dim=-1
    )  # [B, M, O] real
    phase = torch.exp(2j * math.pi * dot)  # [B, M, O] complex

    # 4c) Determinant prefactor (real 2N-dimensional Poisson): 1 / (t^N * det(A_pos))
    #     det(A_pos) = (1/sp) * (1/sq)^{N-1}  =>  prefac = (sp * sq^{N-1}) / t^N
    prefac = (sp * (sq ** (N - 1))) / (t**N)  # [B, M] real
    weight = torch.exp(-(math.pi / t) * quad_k) * phase  # [B, M, O] complex
    Theta_pos = (prefac.unsqueeze(-1) * weight).sum(dim=-1)  # [B, M] complex

    # -- 5) eta_j and field assembly:
    eta_j = C_dir_j * Theta_pos  # [B, M] complex
    w = (alpha_j * eta_j.real).unsqueeze(-1).to(T_hat_j.real.dtype)
    T = w * T_hat_j  # [B, M, S] complex
    T_out = T.sum(dim=1)  # [B, S] complex

    return T_out
