import torch

from typing import Optional, Union, List, Tuple


def _cpu_safe_arange(
    *,
    N: int,
    start: int = 0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    if dtype in (torch.complex128, torch.complex64):
        rtype = torch.float64 if dtype == torch.complex128 else torch.float32
    elif dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        rtype = dtype
    else:
        rtype = torch.float32

    return torch.arange(start, start + N, dtype=rtype, device=device)


def _t_omega_jv(
    *,
    v: Union[torch.FloatTensor, List[float]],
    z: Union[torch.FloatTensor, List[float]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Bessel J_v(z) (first kind) for nonnegative *integer* orders v and real z>=0.
    Vectorized over v and z with broadcasting. Works on CPU/GPU.

    Assumptions matching T_Omega usage:
      - v >= 0, integer (e.g., v = N-1).
      - z >= 0 (in our pipeline z = beta = sqrt(4*pi*||x'||^2)).

    Returns:
      Tensor broadcasted to the common shape of v and z with dtype (float32/float64).
    """

    if device is None:
        device = torch.device("cpu")

    if dtype is None:
        dtype = torch.float64
    else:
        dtype = (
            torch.float64
            if dtype in [torch.complex128, torch.float64]
            else torch.float32
        )

    v = torch.as_tensor(v, dtype=dtype, device=device)
    z = torch.as_tensor(z, dtype=dtype, device=device)

    # Integer, nonnegative orders
    v_rounded = v.round()
    if not torch.allclose(v, v_rounded, atol=0, rtol=0):
        raise ValueError("Only nonnegative *integer* orders v are supported.")
    v_long = v_rounded.to(torch.long)
    if (v_long < 0).any():
        raise ValueError("Orders v must be >= 0.")
    # Arguments z >= 0 (in our pipeline they are)
    if (z < 0).any():
        raise ValueError("Arguments z must be >= 0 for this implementation.")

    PI = torch.tensor(torch.pi, dtype=dtype, device=device)

    # Broadcasting shape
    v_b, z_b = torch.broadcast_tensors(v_long, z)
    out_shape = v_b.shape
    z_flat = z_b.reshape(-1)
    v_flat = v_b.reshape(-1)

    # Numerics
    finfo = torch.finfo(dtype)
    tiny = finfo.tiny
    eps = 1e-14 if dtype == torch.float64 else 1e-7  # series tolerance
    z_thresh = 10.0  # switch between series and asymptotics

    def j0_j1_series(z_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z2 = z_vec * z_vec
        z2_over_4 = 0.25 * z2

        # J0
        term0 = torch.ones_like(z_vec)  # k=0
        sum0 = term0.clone()
        # J1
        term1 = 0.5 * z_vec  # k=0 term = (z/2)
        sum1 = term1.clone()

        Kmax = 128 if dtype == torch.float64 else 64
        for k in range(1, Kmax + 1):
            # term0 *= -(z^2/4) / (k*k)
            term0 = term0 * (-z2_over_4) / (float(k) * float(k))
            sum0 = sum0 + term0

            # term1 *= -(z^2/4) / (k*(k+1))
            term1 = term1 * (-z2_over_4) / (float(k) * float(k + 1))
            sum1 = sum1 + term1

            # crude convergence check (vectorized)
            if k % 8 == 0:
                done0 = term0.abs() <= eps * (sum0.abs() + tiny)
                done1 = term1.abs() <= eps * (sum1.abs() + tiny)
                if torch.all(done0 & done1):
                    break

        return sum0, sum1

    def j0_j1_asymp(z_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_cl = torch.clamp(z_vec, min=torch.as_tensor(tiny, dtype=dtype, device=device))
        invz = 1.0 / z_cl
        invz2 = invz * invz
        invz3 = invz2 * invz
        sqrt_pref = torch.sqrt(2.0 / (PI * z_cl))

        # ν = 0
        phi0 = z_cl - (PI / 4.0)
        cos0 = torch.cos(phi0)
        sin0 = torch.sin(phi0)
        P0 = 1.0 - (9.0 / 128.0) * invz2
        Q0 = (1.0 / 8.0) * invz - (25.0 / 3072.0) * invz3
        J0 = sqrt_pref * (cos0 * P0 - sin0 * Q0)

        # ν = 1
        phi1 = z_cl - (3.0 * PI / 4.0)
        cos1 = torch.cos(phi1)
        sin1 = torch.sin(phi1)
        P1 = 1.0 - (15.0 / 128.0) * invz2
        Q1 = (3.0 / 8.0) * invz - (105.0 / 3072.0) * invz3
        J1 = sqrt_pref * (cos1 * P1 - sin1 * Q1)

        return J0, J1

    mask_small = z_flat <= z_flat.new_tensor(z_thresh)
    J0 = torch.empty_like(z_flat)
    J1 = torch.empty_like(z_flat)

    if mask_small.any():
        j0s, j1s = j0_j1_series(z_flat[mask_small])
        J0[mask_small] = j0s
        J1[mask_small] = j1s
    if (~mask_small).any():
        j0l, j1l = j0_j1_asymp(z_flat[~mask_small])
        J0[~mask_small] = j0l
        J1[~mask_small] = j1l

    if (z_flat == 0).any():
        z0 = z_flat == 0
        J0[z0] = 1.0
        J1[z0] = 0.0

    v_max = int(v_flat.max().item()) if v_flat.numel() > 0 else 0
    if v_max == 0:
        J_stack = J0.unsqueeze(0)  # [1, L]
    else:
        J_list = [J0, J1]
        for n in range(1, v_max):
            n2_over_z = (2.0 * float(n)) / torch.clamp(
                z_flat, min=torch.as_tensor(tiny, dtype=dtype, device=device)
            )
            Jnp1 = n2_over_z * J_list[-1] - J_list[-2]
            if (z_flat == 0).any():
                Jnp1 = torch.where(z_flat == 0, torch.zeros_like(Jnp1), Jnp1)
            J_list.append(Jnp1)
        J_stack = torch.stack(J_list, dim=0)  # [(v_max+1), L]

    idx = v_flat.unsqueeze(0)  # [1, L]
    out_flat = torch.gather(J_stack, 0, idx).squeeze(0)  # [L]

    return out_flat.reshape(out_shape)


def _t_omega_roots_jacobi(
    *,
    N: int,
    alpha: Union[torch.FloatTensor, float],
    beta: Union[torch.FloatTensor, float],
    return_weights: bool = False,
    normalize: bool = False,  # (0,1) mapping + optional weight normalization
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")

    if dtype is None:
        dtype = torch.float64
    else:
        dtype = (
            torch.float64
            if dtype in [torch.complex128, torch.float64]
            else torch.float32
        )

    if N < 1 or N != int(N):
        raise ValueError("N must be a positive integer.")

    a = torch.as_tensor(alpha, dtype=dtype, device=device)
    b = torch.as_tensor(beta, dtype=dtype, device=device)
    n = _cpu_safe_arange(N=N, dtype=dtype, device=device)
    m = _cpu_safe_arange(N=N - 1, dtype=dtype, device=device)

    diag = torch.empty(N, dtype=dtype, device=device)

    diag0 = (b - a) / (2.0 + a + b)
    if N > 1:
        n1 = n[1:]
        d0 = 2.0 * n1 + a + b
        d1 = d0 + 2.0
        diag[1:] = (b * b - a * a) / (d0 * d1)
    diag[0] = diag0

    mp = m + 1.0
    num = 4.0 * mp * (mp + a) * (mp + b) * (mp + a + b)
    den = (
        (2.0 * m + a + b + 1.0) * (2.0 * m + a + b + 3.0) * (2.0 * m + a + b + 2.0) ** 2
    )
    off = torch.sqrt(torch.clamp(num / den, min=0.0))

    J = torch.zeros((N, N), dtype=dtype, device=device)
    J.diagonal().copy_(diag)
    if N > 1:
        J.diagonal(1).copy_(off)
        J.diagonal(-1).copy_(off)
    J = 0.5 * (J + (J.conj().T if torch.is_complex(J) else J.T))

    if not return_weights:
        nodes = torch.linalg.eigvalsh(J)
        nodes = nodes.to(dtype=dtype)

        return 0.5 * (nodes + 1.0) if normalize else nodes
    else:
        evals, evecs = torch.linalg.eigh(J)
        nodes = evals.to(dtype=dtype)

        log_mu_0 = (
            (a + b + 1.0) * torch.log(torch.as_tensor(2.0, dtype=dtype, device=device))
            + torch.lgamma(a + 1.0)
            + torch.lgamma(b + 1.0)
            - torch.lgamma(a + b + 2.0)
        )
        mu_0 = torch.exp(log_mu_0)

        v_0 = evecs[0, :]
        weights = (mu_0 * (v_0 * v_0)).to(dtype=dtype)

        if normalize:
            nodes = 0.5 * (nodes + 1.0)
            weights = weights / mu_0

        return nodes, weights


def _t_omega_roots_gen_laguerre(
    *,
    N: int,
    alpha: Union[torch.FloatTensor, float],
    return_weights: bool = False,
    normalize: bool = False,  # (0,1) mapping via u->u/(1+u) + optional weight normalization
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")

    if dtype is None:
        dtype = torch.float64
    else:
        dtype = (
            torch.float64
            if dtype in [torch.complex128, torch.float64]
            else torch.float32
        )

    if N < 1 or N != int(N):
        raise ValueError("N must be a positive integer.")

    a = torch.as_tensor(alpha, dtype=dtype, device=device)
    if torch.any(a <= -1):
        raise ValueError("alpha must be greater than -1.")

    k = _cpu_safe_arange(N=N, dtype=dtype, device=device)  # 0..N-1
    m = _cpu_safe_arange(N=N - 1, dtype=dtype, device=device)  # 0..N-2

    diag = 2.0 * k + a + 1.0
    off = torch.sqrt(torch.clamp((m + 1.0) * (m + a + 1.0), min=0.0))

    J = torch.zeros((N, N), dtype=dtype, device=device)
    J.diagonal().copy_(diag)
    if N > 1:
        J.diagonal(1).copy_(off)
        J.diagonal(-1).copy_(off)
    J = 0.5 * (J + (J.conj().T if torch.is_complex(J) else J.T))

    if not return_weights:
        nodes = torch.linalg.eigvalsh(J)
        nodes = nodes.to(dtype=dtype)

        return nodes / (nodes + 1.0) if normalize else nodes
    else:
        evals, evecs = torch.linalg.eigh(J)
        nodes = evals.to(dtype=dtype)

        log_mu_0 = torch.lgamma(a + 1.0)
        mu_0 = torch.exp(log_mu_0)

        v_0 = evecs[0, :]
        weights = (mu_0 * (v_0 * v_0)).to(dtype=dtype)

        if normalize:
            nodes = nodes / (nodes + 1.0)
            weights = weights / mu_0

        return nodes, weights
