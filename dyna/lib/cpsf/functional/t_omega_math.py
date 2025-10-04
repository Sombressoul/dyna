import torch

from typing import Optional, Union


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


def roots_jacobi(
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


def roots_gen_laguerre(
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
