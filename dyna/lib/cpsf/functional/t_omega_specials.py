import torch
import math

from typing import Optional


def _get_real_dtype(
    *,
    dtype: torch.dtype,
) -> torch.dtype:
    if dtype == torch.complex64 or dtype == torch.float32:
        return torch.float32
    if dtype == torch.complex128 or dtype == torch.float64:
        return torch.float64
    raise TypeError(f"Unsupported dtype: {dtype}")


def _gauss_jacobi_01_nodes(
    *,
    Q: int,
    alpha: float,
    beta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    rtype = _get_real_dtype(dtype=dtype)
    alpha = torch.as_tensor(alpha, dtype=rtype, device=device)
    beta = torch.as_tensor(beta, dtype=rtype, device=device)

    n = torch.arange(Q, dtype=rtype, device=device)
    m = torch.arange(Q - 1, dtype=rtype, device=device)
    d = (beta * beta - alpha * alpha) / (
        (2 * n + alpha + beta) * (2 * n + alpha + beta + 2.0)
    )
    num = (
        4.0
        * (m + 1.0)
        * (m + 1.0 + alpha)
        * (m + 1.0 + beta)
        * (m + 1.0 + alpha + beta)
    )
    den = (
        (2 * m + alpha + beta + 1.0)
        * (2 * m + alpha + beta + 3.0)
        * torch.pow(2 * m + alpha + beta + 2.0, 2.0)
    )
    e = torch.sqrt(num / den)

    J = torch.zeros((Q, Q), dtype=rtype, device=device)
    J.diagonal().copy_(d)
    J.diagonal(1).copy_(e)
    J.diagonal(-1).copy_(e)

    x, V = torch.linalg.eigh(J)
    t_q = (x + 1.0) * 0.5

    log_beta = (
        torch.lgamma(alpha + 1.0)
        + torch.lgamma(beta + 1.0)
        - torch.lgamma(alpha + beta + 2.0)
    )
    w_q = torch.exp(log_beta) * (V[0, :].abs() ** 2)

    return t_q.to(rtype), w_q.to(rtype)


def _precompute_gauss_jacobi_for_1f1_half(
    *,
    N: int,
    Q_t: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
    rtype = _get_real_dtype(dtype=dtype)
    alpha_neg = -0.5
    beta_neg = float(N) - 1.5
    t_neg, w_neg = _gauss_jacobi_01_nodes(
        Q=Q_t,
        alpha=alpha_neg,
        beta=beta_neg,
        dtype=dtype,
        device=device,
    )
    logC_neg = (
        torch.lgamma(torch.as_tensor(float(N), dtype=rtype, device=device))
        - 0.5 * math.log(math.pi)
        - torch.lgamma(torch.as_tensor(float(N) - 0.5, dtype=rtype, device=device))
    )

    alpha_pos = float(N) - 1.5
    beta_pos = -0.5
    t_pos, w_pos = _gauss_jacobi_01_nodes(
        Q=Q_t,
        alpha=alpha_pos,
        beta=beta_pos,
        dtype=dtype,
        device=device,
    )
    logC_pos = logC_neg

    return {
        "neg": {"t": t_neg, "w": w_neg, "logC": logC_neg},
        "pos": {"t": t_pos, "w": w_pos, "logC": logC_pos},
        "dtype": rtype,
    }


def _hyp1f1_half_negx_gauss_jacobi(
    *,
    x: torch.Tensor,
    N: int,
    Q_t: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    cache: Optional[dict] = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = x.dtype if x.is_floating_point() else torch.float32
    if device is None:
        device = x.device

    rtype = _get_real_dtype(dtype=dtype)
    x = x.to(rtype).to(device)
    z = -x

    if cache is None or cache.get("dtype", None) != rtype:
        cache = _precompute_gauss_jacobi_for_1f1_half(
            N=N,
            Q_t=Q_t,
            dtype=rtype,
            device=device,
        )

    m_neg = z <= 0
    m_pos = ~m_neg

    if m_neg.any():
        t = cache["neg"]["t"]
        w = cache["neg"]["w"]
        logC = cache["neg"]["logC"]
        z_neg = z[m_neg].unsqueeze(-1)
        y_neg = torch.exp(logC) * torch.sum(w * torch.exp(z_neg * t), dim=-1)

    if m_pos.any():
        t = cache["pos"]["t"]
        w = cache["pos"]["w"]
        logC = cache["pos"]["logC"]
        z_pos = z[m_pos].unsqueeze(-1)
        y_pos = torch.sum(w * torch.exp(z_pos * (1.0 - t)), dim=-1) * torch.exp(logC)

    y = torch.zeros_like(z)
    if m_neg.any():
        y[m_neg] = y_neg
    if m_pos.any():
        y[m_pos] = y_pos

    return y
