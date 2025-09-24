import torch, math

from typing import Optional, Iterable, Tuple

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
    offsets: torch.Tensor,  # Dual-space k modes on Z^{2N}
    t: float = 1.0,  # Poisson/Ewald scale, for any t>0 - exact
    R_j: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    CPSF field via Poisson-dual (PD) WINDOW over k-modes (single-shot fixed batch).

    Purpose
    -------
    Computes T(z, vec_d) using the Poisson-dual representation of the positional
    lattice sum and a closed-form directional Gaussian factor. Offsets are k-modes
    in Z^{2N} passed as a single fixed batch. The scale t>0 is the Poisson/Ewald
    parameter (the infinite sum is t-invariant; with a finite window, t trades off
    decay between real/dual sides).

    Shapes and Dtypes
    -----------------
    Let:
    - B: batch dimension
    - N: ambient dimension
    - M: number of contributions (atoms)
    - S: output channels / field dimension
    - O: number of dual k-modes in the window

    Required tensor shapes:
    - z             : [B, N]            (complex)
    - z_j           : [B, M, N]         (complex)
    - vec_d         : [B, N]            (same dtype as z)
    - vec_d_j       : [B, M, N]         (same dtype as z)
    - T_hat_j       : [B, M, S]         (complex, contributes to final T)
    - alpha_j       : [B, M]            (real)
    - sigma_par     : [B, M]            (real; parallel scale)
    - sigma_perp    : [B, M]            (real; perpendicular scale)
    - offsets       : [O, 2N]           (LongTensor; integer k in Z^{2N};
                                         first N columns -> real part, last N -> imag)
    - R_j (opt)     : [B, M, N, N]      (dtype compatible with q())
    - t (opt)       : float > 0         (Poisson/Ewald scale)

    Semantics
    ---------
    - Only the positional block is periodized/summed. The directional block is NOT
      periodized; it contributes a multiplicative Gaussian factor:

        C_dir = exp(-pi * q([0, delta_vec_d]))
        with Sigma built from (sigma_par, sigma_perp) and R_ext(R(vec_d_j)).

    - Positional sum is evaluated in the dual (k) domain:

        y = R(vec_d_j)^H * k  (complex N-vector per k),
        quad_k = sigma_perp * sum(|y_i|^2) + (sigma_par - sigma_perp) * |y_0|^2,
        phase = exp(2*pi*i * k dot frac(delta_z)),
        Theta_pos = (sigma_par * sigma_perp^(N-1) / t^N) * sum_k exp(-(pi/t)*quad_k) * phase.

        Here frac(delta_z) is taken componentwise on the unit torus.

    - The contribution is eta_j = C_dir * Theta_pos; the field is

        T = sum_j (alpha_j * Re(eta_j)) * T_hat_j.

    - Row order of `offsets` is non-contractual; sorting changes only order, not coverage.
      Execution is one-shot vectorized over the provided k-window.

    Returns
    -------
    T : [B, S] (complex, same dtype as T_hat_j)

    Notes
    -----
    - Exact w.r.t. the infinite lattice for any t>0. With a finite k-window, accuracy
      depends on the window size and t (standard Ewald trade-off).
    - Peak memory ~ O(B * M * O) due to per-k accumulation.
    """
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

    # Prefactor: det(A_pos) / t^N  (complex-N convention)
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


def T_PD_full(
    *,
    z: torch.Tensor,
    z_j: torch.Tensor,
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    T_hat_j: torch.Tensor,
    alpha_j: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
    packs: Iterable[Tuple[int, int, torch.Tensor]],
    R_j: Optional[torch.Tensor] = None,
    t: float = 1.0,
    tol_abs: Optional[float] = None,
    tol_rel: Optional[float] = None,
    consecutive_below: int = 1,
) -> torch.Tensor:
    if t <= 0.0:
        raise ValueError("T_PD_full: t must be > 0.")

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
    Rmat = R(vec_d_j) if R_j is None else R_j  # [B,M,N,N]
    RH = Rmat.conj().transpose(-2, -1)  # [B,M,N,N]
    Rext = R_ext(Rmat)

    # Const
    pi = torch.tensor(math.pi, dtype=r_dtype, device=device)

    # Directional Gaussian (no lattice in dir)
    zeros_u = torch.zeros_like(dd)  # [B,M,N]
    w_dir = iota(zeros_u, dd)  # [B,M,N]
    q_dir = q(w=w_dir, R_ext=Rext, sigma_par=sigma_par, sigma_perp=sigma_perp)  # [B,M]
    C_dir_j = torch.exp(-pi * q_dir)  # [B,M]

    # Phase base (torus fractional coords)
    dz_re_frac = torch.remainder(dz.real, 1.0)  # [B,M,N]
    dz_im_frac = torch.remainder(dz.imag, 1.0)  # [B,M,N]

    # Prefactor (complex-N convention): det_C(Sigma_pos) / t^N
    prefac = (sigma_par * (sigma_perp ** (N - 1))) / (t**N)  # [B,M], no sqrt

    # Accumulators
    T_acc = torch.zeros(B, T_hat_j.shape[-1], dtype=T_hat_j.dtype, device=device)
    below = 0

    # Early stopping flags
    use_abs = tol_abs is not None
    use_rel = (tol_rel is not None) and (tol_rel >= 0.0)

    for _, _, offsets in packs:
        # k on Z^{2N}
        k_r = offsets[:, :N].to(device=device, dtype=r_dtype)  # [O,N]
        k_i = offsets[:, N:].to(device=device, dtype=r_dtype)  # [O,N]
        k_c = torch.complex(k_r, k_i)  # [O,N]

        # Dual quadratic form: k^T A_pos^{-1} k = sigma_perp*||y||^2 + (sigma_par-sigma_perp)*|y_0|^2
        # where y = R^H k
        y = torch.einsum("bmnk,ok->bmon", RH, k_c)  # [B,M,O,N]
        y_abs2 = (y.real**2) + (y.imag**2)
        s = y_abs2.sum(dim=-1)  # [B,M,O]
        p_abs2 = y_abs2[..., 0]  # [B,M,O]
        quad_k = (sigma_perp.unsqueeze(-1) * s) + (
            (sigma_par - sigma_perp).unsqueeze(-1) * p_abs2
        )  # [B,M,O]

        # Phase: exp(2π i k · frac(dz))
        dot = (k_r.unsqueeze(0).unsqueeze(0) * dz_re_frac.unsqueeze(2)).sum(dim=-1) + (
            k_i.unsqueeze(0).unsqueeze(0) * dz_im_frac.unsqueeze(2)
        ).sum(
            dim=-1
        )  # [B,M,O]
        ang = (2.0 * pi).to(dot.dtype) * dot
        phase = torch.polar(torch.ones_like(ang), ang)  # [B,M,O] complex (unit modulus)

        # Weight and per-pack positional theta contribution
        weight = torch.exp(-(pi / t) * quad_k) * phase  # [B,M,O]
        Theta_pos_pack = (prefac.unsqueeze(-1) * weight).sum(dim=-1)  # [B,M] complex

        # eta_j contribution from this pack
        eta_pack = (C_dir_j * Theta_pos_pack).real.to(T_hat_j.real.dtype)  # [B,M]
        w = (alpha_j.to(T_hat_j.real.dtype) * eta_pack).unsqueeze(-1)  # [B,M,1]
        T_delta = (w * T_hat_j).sum(dim=1)  # [B,S] complex

        # Accumulate
        T_acc = T_acc + T_delta

        # Early stopping bookkeeping
        if use_abs or use_rel:
            pack_max = T_delta.abs().amax().item()
            acc_max = T_acc.abs().amax().item()
            below_abs = use_abs and (pack_max <= tol_abs)
            below_rel = use_rel and (acc_max > 0.0) and (pack_max <= tol_rel * acc_max)
            if below_abs or below_rel:
                below += 1
                if below >= consecutive_below:
                    break
            else:
                below = 0

    return T_acc
