# Run as (example):
# > pytest -q .\dyna\lib\cpsf\pytest\test_CPSF_q.py

import torch
import pytest
from typing import Callable, List, Tuple

from dyna.lib.cpsf.functional.core_math import (
    R,
    R_ext,
    Sigma,
    q,
)

# =========================
# Global config
# =========================
TARGET_DEVICE = torch.device("cpu")

Q_IMPLS: List[Tuple[str, Callable[..., torch.Tensor]]] = [
    (
        "q",
        lambda w, Rext, sp, sq: q(
            w=w, R_ext=Rext, sigma_par=sp, sigma_perp=sq
        ),
    ),
]

DTYPES = [torch.complex64, torch.complex128]
NS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

SEED = 1337
_GEN = {}

# =========================
# Tolerances
# =========================
_TOLS = {
    torch.complex64: dict(rtol=5e-5, atol=5e-6),
    torch.complex128: dict(rtol=1e-12, atol=1e-12),
}


def _get_tols(dtype: torch.dtype):
    t = _TOLS[dtype]
    return t["rtol"], t["atol"]


# =========================
# Helpers
# =========================
def _gen_for(device: torch.device) -> torch.Generator:
    dev = torch.device(device)
    key = (dev.type, dev.index if dev.index is not None else -1)
    if key not in _GEN:
        _GEN[key] = torch.Generator(device=dev).manual_seed(SEED)
    return _GEN[key]


def rand_unit_vector(
    n: int,
    dtype: torch.dtype,
    device: torch.device = TARGET_DEVICE,
) -> torch.Tensor:
    """Complex gaussian -> normalize to unit vector (device-aware RNG)."""
    gen = _gen_for(device)
    xr = torch.randn(n, generator=gen, device=device, dtype=torch.float64)
    xi = torch.randn(n, generator=gen, device=device, dtype=torch.float64)
    v = (xr + 1j * xi).to(dtype)
    nrm = torch.linalg.vector_norm(v)
    if float(nrm.real) < torch.finfo(v.real.dtype).eps:
        v = torch.zeros_like(v)
        v[0] = 1
        return v
    return v / nrm


def rand_complex_vector(
    n: int,
    dtype: torch.dtype,
    device: torch.device = TARGET_DEVICE,
) -> torch.Tensor:
    """Unnormalized complex gaussian vector."""
    gen = _gen_for(device)
    xr = torch.randn(n, generator=gen, device=device, dtype=torch.float64)
    xi = torch.randn(n, generator=gen, device=device, dtype=torch.float64)
    return (xr + 1j * xi).to(dtype)


def call_q(
    fn: Callable[..., torch.Tensor],
    w: torch.Tensor,
    Rext: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
) -> torch.Tensor:
    return fn(w, Rext, sigma_par, sigma_perp)


def call_sigma(
    fn: Callable[..., torch.Tensor],
    Rext: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
) -> torch.Tensor:
    return fn(Rext, sigma_par, sigma_perp)


def ref_q_via_Sigma_solve(
    w: torch.Tensor,
    Rext: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
) -> torch.Tensor:
    Sig = Sigma(R_ext=Rext, sigma_par=sigma_par, sigma_perp=sigma_perp)
    L = torch.linalg.cholesky(Sig)
    x = torch.cholesky_solve(w.unsqueeze(-1), L).squeeze(-1)
    q = (w.conj() * x).sum(dim=-1).real
    return q


# =========================
# Tests
# =========================


# ============================> TEST: Q01 — shape/dtype/device & arg validation
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_Q01_shape_dtype_device_and_args(impl_name, fn_q, dtype, N):
    device = TARGET_DEVICE
    twoN = 2 * N
    w_good = rand_complex_vector(twoN, dtype=dtype, device=device)
    Rext_good = torch.eye(twoN, dtype=dtype, device=device)
    sp = torch.tensor(1.0, device=device)
    sq = torch.tensor(2.0, device=device)

    w_real = w_good.real.to(w_good.real.dtype)
    with pytest.raises(ValueError):
        _ = fn_q(w_real, Rext_good, sp, sq)

    Rext_real = Rext_good.real.to(w_good.real.dtype)
    with pytest.raises(ValueError):
        _ = fn_q(w_good, Rext_real, sp, sq)

    Rext_nonsquare = torch.zeros((twoN, twoN + 1), dtype=dtype, device=device)
    with pytest.raises(ValueError):
        _ = fn_q(w_good, Rext_nonsquare, sp, sq)

    Rext_mismatch = torch.eye(twoN + 2, dtype=dtype, device=device)
    with pytest.raises(ValueError):
        _ = fn_q(w_good, Rext_mismatch, sp, sq)

    w_odd = rand_complex_vector(twoN + 1, dtype=dtype, device=device)
    Rext_odd = torch.eye(twoN + 1, dtype=dtype, device=device)
    with pytest.raises(ValueError):
        _ = fn_q(w_odd, Rext_odd, sp, sq)

    other_dtype = torch.complex128 if dtype == torch.complex64 else torch.complex64
    Rext_wrong_dtype = Rext_good.to(other_dtype)
    with pytest.raises(ValueError):
        _ = fn_q(w_good, Rext_wrong_dtype, sp, sq)

    if torch.cuda.is_available():
        if TARGET_DEVICE.type == torch.device("cuda").type:
            Rext_cpu = Rext_good.to("cpu")
            with pytest.raises(ValueError):
                _ = fn_q(w_good, Rext_cpu, sp, sq)
        else:
            Rext_cuda = Rext_good.to("cuda")
            with pytest.raises(ValueError):
                _ = fn_q(w_good, Rext_cuda, sp, sq)

    sp_bad = torch.tensor(0.0, device=device)
    with pytest.raises(ValueError):
        _ = fn_q(w_good, Rext_good, sp_bad, sq)

    sq_bad = torch.tensor(-1.0, device=device)
    with pytest.raises(ValueError):
        _ = fn_q(w_good, Rext_good, sp, sq_bad)


# ============================> TEST: Q02 — equivalence with Σ-solve reference
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_Q02_equivalence_with_sigma_solve(impl_name, fn_q, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    twoN = 2 * N
    d = rand_unit_vector(N, dtype=dtype, device=device)
    Rmat = R(d)
    Rext = R_ext(Rmat)
    w = rand_complex_vector(twoN, dtype=dtype, device=device)
    sp = torch.tensor(0.7, device=device)
    sq = torch.tensor(1.9, device=device)
    q = call_q(fn_q, w, Rext, sp, sq)
    q_ref = ref_q_via_Sigma_solve(w, Rext, sp, sq)

    assert torch.allclose(q, q_ref, rtol=rtol, atol=atol), (
        f"{impl_name}: mismatch vs Σ-solve reference "
        f"(dtype={dtype}, N={N}) -> q={q.item()} vs q_ref={q_ref.item()}"
    )


# ============================> TEST: Q03 — isotropic case (sp == sq)
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_Q03_isotropic_case(impl_name, fn_q, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    twoN = 2 * N
    d = rand_unit_vector(N, dtype=dtype, device=device)
    Rmat = R(d)
    Rext = R_ext(Rmat)
    w = rand_complex_vector(twoN, dtype=dtype, device=device)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    s = torch.tensor(1.5, device=device, dtype=REAL)
    sp = s
    sq = s
    q = call_q(fn_q, w, Rext, sp, sq)
    wnorm2 = (w.conj() * w).sum().real.to(REAL)
    q_ref = wnorm2 / s

    assert torch.allclose(
        q, q_ref, rtol=rtol, atol=atol
    ), f"{impl_name}: isotropic mismatch (dtype={dtype}, N={N}) -> q={q.item()} vs q_ref={q_ref.item()}"


# ============================> TEST: Q04 — scaling (1/α) and monotonicity in sigmas
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("alpha", [0.5, 2.0])
def test_Q04_scaling_and_monotonicity(impl_name, fn_q, dtype, N, alpha):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    twoN = 2 * N
    d = rand_unit_vector(N, dtype=dtype, device=device)
    Rmat = R(d)
    Rext = R_ext(Rmat)
    w = rand_complex_vector(twoN, dtype=dtype, device=device)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    sp = torch.tensor(0.7, device=device, dtype=REAL)
    sq = torch.tensor(1.9, device=device, dtype=REAL)
    q_base = call_q(fn_q, w, Rext, sp, sq)

    a = torch.tensor(alpha, device=device, dtype=REAL)
    q_scaled = call_q(fn_q, w, Rext, sp * a, sq * a)
    q_expect = q_base / a
    assert torch.allclose(q_scaled, q_expect, rtol=rtol, atol=atol), (
        f"{impl_name}: scaling mismatch (dtype={dtype}, N={N}, alpha={alpha}) -> "
        f"q_scaled={q_scaled.item()} vs q_expect={q_expect.item()}"
    )

    gamma = torch.tensor(1.5, device=device, dtype=REAL)
    q_mono = call_q(fn_q, w, Rext, sp * gamma, sq * gamma)
    eps = 10 * atol
    assert q_mono.item() <= q_base.item() + eps, (
        f"{impl_name}: monotonicity failed (dtype={dtype}, N={N}) -> "
        f"q(gamma*Σ)={q_mono.item()} > q(Σ)+eps={q_base.item()+eps}"
    )


# ============================> TEST: Q05 — batch semantics / broadcasting (expanded)
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("b_shape", [(4,), (2, 3)])
@pytest.mark.parametrize("bcast", ["scalar", "per_batch", "mixed_left", "mixed_right"])
def test_Q05_batch_semantics_and_broadcast(impl_name, fn_q, dtype, N, b_shape, bcast):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    twoN = 2 * N
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    gen = _gen_for(device)
    xr = torch.randn(*b_shape, N, generator=gen, device=device, dtype=REAL)
    xi = torch.randn(*b_shape, N, generator=gen, device=device, dtype=REAL)
    d = (xr + 1j * xi).to(dtype)
    d = d / torch.clamp(torch.linalg.vector_norm(d, dim=-1, keepdim=True), min=1e-12)
    Rmat = R(d)
    Rext = R_ext(Rmat)
    xr_w = torch.randn(*b_shape, twoN, generator=gen, device=device, dtype=REAL)
    xi_w = torch.randn(*b_shape, twoN, generator=gen, device=device, dtype=REAL)
    w = (xr_w + 1j * xi_w).to(dtype)
    base_sp = 0.7
    base_sq = 1.9

    if bcast == "scalar":
        sp = torch.tensor(base_sp, device=device, dtype=REAL)
        sq = torch.tensor(base_sq, device=device, dtype=REAL)
    elif bcast == "per_batch":
        sp = torch.full(b_shape, base_sp, device=device, dtype=REAL) * 1.0
        sq = torch.full(b_shape, base_sq, device=device, dtype=REAL) * 1.0
    elif bcast == "mixed_left":
        sp = torch.tensor(base_sp, device=device, dtype=REAL)
        sq = torch.full(b_shape, base_sq, device=device, dtype=REAL)
    elif bcast == "mixed_right":
        sp = torch.full(b_shape, base_sp, device=device, dtype=REAL)
        sq = torch.tensor(base_sq, device=device, dtype=REAL)
    else:
        raise RuntimeError("Unknown bcast")

    q = call_q(fn_q, w, Rext, sp, sq)
    q_ref = ref_q_via_Sigma_solve(w, Rext, sp, sq)

    assert q.shape == torch.empty(*b_shape, dtype=REAL, device=device).shape
    assert q.dtype == REAL
    assert q.device.type == device.type
    assert torch.allclose(q, q_ref, rtol=rtol, atol=atol), (
        f"{impl_name}: broadcast mismatch "
        f"(dtype={dtype}, N={N}, b_shape={b_shape}, case={bcast})"
    )


# ============================> TEST: Q06 — right action U(N-1) invariance via R → R·diag(1,U)
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_Q06_right_action_invariance(impl_name, fn_q, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    twoN = 2 * N
    d = rand_unit_vector(N, dtype=dtype, device=device)
    Rmat = R(d)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64

    if N >= 2:
        gen = _gen_for(device)
        Ar = torch.randn(N - 1, N - 1, generator=gen, device=device, dtype=REAL)
        Ai = torch.randn(N - 1, N - 1, generator=gen, device=device, dtype=REAL)
        A = (Ar + 1j * Ai).to(dtype)
        U, _ = torch.linalg.qr(A)
        D = torch.eye(N, device=device, dtype=dtype)
        D[1:, 1:] = U
    else:
        pytest.skip("CPSF requires N >= 2; N==1 is disallowed by design.")

    R_right = Rmat @ D
    Rext = R_ext(Rmat)
    Rext_right = R_ext(R_right)
    w = rand_complex_vector(twoN, dtype=dtype, device=device)
    sp = torch.tensor(0.7, device=device, dtype=REAL)
    sq = torch.tensor(1.9, device=device, dtype=REAL)
    q_base = call_q(fn_q, w, Rext, sp, sq)
    q_right = call_q(fn_q, w, Rext_right, sp, sq)

    assert torch.allclose(q_base, q_right, rtol=rtol, atol=atol), (
        f"{impl_name}: right-action invariance failed (dtype={dtype}, N={N}) -> "
        f"q(R)={q_base.item()} vs q(R·diag(1,U))={q_right.item()}"
    )


# ============================> TEST: Q07 — first-column phase invariance (R → R·diag(e^{iφ}, I))
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("phi", [0.0, 0.3, 1.0, 2.0])
def test_Q07_first_column_phase_invariance(impl_name, fn_q, dtype, N, phi):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    twoN = 2 * N
    d = rand_unit_vector(N, dtype=dtype, device=device)
    Rmat = R(d)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    phase = torch.tensor(phi, device=device, dtype=REAL)
    c = torch.complex(torch.cos(phase), torch.sin(phase)).to(dtype)
    D = torch.eye(N, device=device, dtype=dtype)
    D[0, 0] = c
    R_phase = Rmat @ D
    Rext = R_ext(Rmat)
    Rext_phase = R_ext(R_phase)
    w = rand_complex_vector(twoN, dtype=dtype, device=device)
    sp = torch.tensor(0.7, device=device, dtype=REAL)
    sq = torch.tensor(1.9, device=device, dtype=REAL)
    q0 = call_q(fn_q, w, Rext, sp, sq)
    q1 = call_q(fn_q, w, Rext_phase, sp, sq)

    assert torch.allclose(
        q0, q1, rtol=rtol, atol=atol
    ), f"{impl_name}: first-column phase invariance failed " "".join(
        [
            f"(dtype={dtype}, N={N}, phi={phi}) -> q(R)={q0.item()} vs q(R·diag(e^",
            "{iφ}",
            f", I))={q1.item()}",
        ]
    )


# ============================> TEST: Q08 — positivity: q >= 0 and q == 0 iff w == 0
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_Q08_positivity_and_zero_vector(impl_name, fn_q, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    twoN = 2 * N
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    d = rand_unit_vector(N, dtype=dtype, device=device)
    Rmat = R(d)
    Rext = R_ext(Rmat)
    sp = torch.tensor(0.7, device=device, dtype=REAL)
    sq = torch.tensor(1.9, device=device, dtype=REAL)
    w_zero = torch.zeros(twoN, dtype=dtype, device=device)
    q_zero = call_q(fn_q, w_zero, Rext, sp, sq)

    assert torch.allclose(
        q_zero, torch.zeros((), dtype=REAL, device=device), rtol=rtol, atol=atol
    ), f"{impl_name}: q(0) != 0 (dtype={dtype}, N={N}) -> q_zero={q_zero.item()}"

    w = rand_complex_vector(twoN, dtype=dtype, device=device)
    q = call_q(fn_q, w, Rext, sp, sq)
    eps = 10 * atol

    assert (
        q.item() >= -eps
    ), f"{impl_name}: q(w) negative beyond tolerance (dtype={dtype}, N={N}) -> q={q.item()}"
    assert (
        q.item() > eps
    ), f"{impl_name}: q(w) not strictly positive for non-zero w (dtype={dtype}, N={N}) -> q={q.item()}"


# ============================> TEST: Q10 — numerical stability (extreme sigma values)
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6])
def test_Q10_numerical_stability_extremal_sigmas(impl_name, fn_q, dtype, N, scale):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    twoN = 2 * N
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    d = rand_unit_vector(N, dtype=dtype, device=device)
    Rmat = R(d)
    Rext = R_ext(Rmat)
    w = rand_complex_vector(twoN, dtype=dtype, device=device)
    s = torch.tensor(scale, device=device, dtype=REAL)
    sp = torch.tensor(0.7, device=device, dtype=REAL) * s
    sq = torch.tensor(1.9, device=device, dtype=REAL) * s
    q = call_q(fn_q, w, Rext, sp, sq)
    q_ref = ref_q_via_Sigma_solve(w, Rext, sp, sq)

    assert torch.allclose(q, q_ref, rtol=rtol, atol=atol), (
        f"{impl_name}: extreme-sigma mismatch "
        f"(dtype={dtype}, N={N}, scale={scale}) -> q={q.item()} vs q_ref={q_ref.item()}"
    )


# ============================> TEST: Q11 — device propagation and CPU/GPU parity (optional)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", [2, 8, 16])
def test_Q11_device_propagation_and_cpu_gpu_parity(impl_name, fn_q, dtype, N):
    rtol, atol = _get_tols(dtype)
    twoN = 2 * N
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    d_cpu = rand_unit_vector(N, dtype=dtype, device=cpu)
    R_cpu = R(d_cpu)
    Rext_cpu = R_ext(R_cpu)
    w_cpu = rand_complex_vector(twoN, dtype=dtype, device=cpu)
    sp_cpu = torch.tensor(0.7, device=cpu, dtype=REAL)
    sq_cpu = torch.tensor(1.9, device=cpu, dtype=REAL)
    q_cpu = call_q(fn_q, w_cpu, Rext_cpu, sp_cpu, sq_cpu)

    assert q_cpu.device.type == "cpu"

    d_gpu = d_cpu.to(cuda)
    R_gpu = R(d_gpu)
    Rext_gpu = R_ext(R_gpu)
    w_gpu = w_cpu.to(cuda)
    sp_gpu = sp_cpu.to(cuda)
    sq_gpu = sq_cpu.to(cuda)
    q_gpu = call_q(fn_q, w_gpu, Rext_gpu, sp_gpu, sq_gpu)

    assert q_gpu.device.type == "cuda"
    assert torch.allclose(q_cpu, q_gpu.to(cpu), rtol=rtol, atol=atol), (
        f"{impl_name}: CPU/GPU parity failed "
        f"(dtype={dtype}, N={N}) -> q_cpu={q_cpu.item()} vs q_gpu={q_gpu.item()}"
    )


# ============================> TEST: Q12 — global phase invariance in w (w → e^{iφ} w)
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("phi", [0.0, 0.7, 2.4])
def test_Q12_global_phase_invariance_in_w(impl_name, fn_q, dtype, N, phi):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    twoN = 2 * N
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    d = rand_unit_vector(N, dtype=dtype, device=device)
    Rmat = R(d)
    Rext = R_ext(Rmat)
    w = rand_complex_vector(twoN, dtype=dtype, device=device)
    sp = torch.tensor(0.7, device=device, dtype=REAL)
    sq = torch.tensor(1.9, device=device, dtype=REAL)
    q0 = call_q(fn_q, w, Rext, sp, sq)
    phase = torch.tensor(phi, device=device, dtype=REAL)
    c = torch.complex(torch.cos(phase), torch.sin(phase)).to(dtype)
    w_phase = w * c
    q1 = call_q(fn_q, w_phase, Rext, sp, sq)

    assert torch.allclose(q0, q1, rtol=rtol, atol=atol), (
        f"{impl_name}: global phase invariance failed "
        f"(dtype={dtype}, N={N}, phi={phi}) -> q(w)={q0.item()} vs q(e^{1j*phi}w)={q1.item()}"
    )


# ============================> TEST: Q13 — block additivity and swap symmetry for w = [u; v]
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_Q13_block_additivity_and_swap_symmetry(impl_name, fn_q, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    d = rand_unit_vector(N, dtype=dtype, device=device)
    Rmat = R(d)
    Rext = R_ext(Rmat)
    u = rand_complex_vector(N, dtype=dtype, device=device)
    v = rand_complex_vector(N, dtype=dtype, device=device)
    z = torch.zeros(N, dtype=dtype, device=device)
    w_u0 = torch.cat([u, z], dim=-1)
    w_0v = torch.cat([z, v], dim=-1)
    w_uv = torch.cat([u, v], dim=-1)
    w_vu = torch.cat([v, u], dim=-1)
    sp = torch.tensor(0.7, device=device, dtype=REAL)
    sq = torch.tensor(1.9, device=device, dtype=REAL)
    q_u0 = call_q(fn_q, w_u0, Rext, sp, sq)
    q_0v = call_q(fn_q, w_0v, Rext, sp, sq)
    q_uv = call_q(fn_q, w_uv, Rext, sp, sq)
    q_vu = call_q(fn_q, w_vu, Rext, sp, sq)

    assert torch.allclose(q_uv, q_u0 + q_0v, rtol=rtol, atol=atol), (
        f"{impl_name}: block additivity failed "
        f"(dtype={dtype}, N={N}) -> q([u;v])={q_uv.item()} vs q([u;0])+q([0;v])={(q_u0+q_0v).item()}"
    )
    assert torch.allclose(q_uv, q_vu, rtol=rtol, atol=atol), (
        f"{impl_name}: swap symmetry failed "
        f"(dtype={dtype}, N={N}) -> q([u;v])={q_uv.item()} vs q([v;u])={q_vu.item()}"
    )


# ============================> TEST: Q14 — special case R == I
@pytest.mark.parametrize("impl_name,fn_q", Q_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_Q14_special_case_R_eq_I(impl_name, fn_q, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    twoN = 2 * N
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    Rmat = torch.eye(N, dtype=dtype, device=device)
    Rext = R_ext(Rmat)
    w = rand_complex_vector(twoN, dtype=dtype, device=device)
    sp = torch.tensor(0.7, device=device, dtype=REAL)
    sq = torch.tensor(1.9, device=device, dtype=REAL)
    q = call_q(fn_q, w, Rext, sp, sq)
    u = w[..., :N]
    v = w[..., N:]
    par = (u[..., 0].abs().pow(2) + v[..., 0].abs().pow(2)).to(REAL) / sp
    perp = (
        u[..., 1:].abs().pow(2).sum(dim=-1) + v[..., 1:].abs().pow(2).sum(dim=-1)
    ).to(REAL) / sq
    q_ref = par + perp

    assert torch.allclose(q, q_ref, rtol=rtol, atol=atol), (
        f"{impl_name}: R==I case mismatch (dtype={dtype}, N={N}) -> "
        f"q={q.item()} vs q_ref={q_ref.item()}"
    )


# =========================
# Direct run help.
# =========================
if __name__ == "__main__":
    print("\nUse pytest to run:")
    print("\tpytest -q ./test_CPSF_q.py\n")
