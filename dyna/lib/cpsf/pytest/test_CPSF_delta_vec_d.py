# Run as (example):
# > pytest -q .\dyna\lib\cpsf\pytest\test_CPSF_delta_vec_d.py

import torch
import pytest
from typing import Callable, List, Tuple

from dyna.lib.cpsf.functional.core_math import (
    delta_vec_d,
)

# =========================
# Global config
# =========================
TARGET_DEVICE = torch.device("cpu")

DELTA_IMPLS: List[Tuple[str, Callable[..., torch.Tensor]]] = [
    ("delta_vec_d", lambda v, vj, eps: delta_vec_d(v, vj, eps)),
]

DTYPES = [torch.complex64, torch.complex128]
# CPSF assumes N >= 2; keep it that way in tests
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

# factors for specific checks
ZERO_ATOL_FACTOR = 10
PHASE_ATOL_FACTOR = 10
ASYMPT_RTL_MULT = 2  # RTOL for asymptotic check will be ASYMPT_RTL_MULT * rtol


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
    device=TARGET_DEVICE,
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


def call_DELTA(
    fn: Callable[..., torch.Tensor],
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return fn(vec_d, vec_d_j, eps)


# =========================
# Tests
# =========================


# ============================> TEST: D01 — Shape/dtype/device & arg validation
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D01_shape_dtype_device_and_args(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    vec_d = rand_unit_vector(N, dtype, device)
    vec_d_j = rand_unit_vector(N, dtype, device)
    eps = 1.0e-6
    out = call_DELTA(fn, vec_d, vec_d_j, eps)

    assert out.shape == (N,), f"{impl_name}: output shape mismatch"
    assert out.dtype == dtype, f"{impl_name}: output dtype mismatch"
    assert out.device.type == device.type, f"{impl_name}: output device mismatch"

    with pytest.raises(ValueError):
        _ = call_DELTA(fn, vec_d, vec_d_j, 0.0)
    with pytest.raises(ValueError):
        _ = call_DELTA(fn, vec_d, vec_d_j, -1e-8)

    vec_bad = rand_unit_vector(N + 1, dtype, device)
    with pytest.raises(ValueError):
        _ = call_DELTA(fn, vec_bad, vec_d_j, eps)
    with pytest.raises(ValueError):
        _ = call_DELTA(fn, vec_d, vec_bad, eps)

    other_dtype = torch.complex128 if dtype == torch.complex64 else torch.complex64
    with pytest.raises(ValueError):
        _ = call_DELTA(fn, vec_d, vec_d_j.to(other_dtype), eps)

    if torch.cuda.is_available():
        if device.type == "cpu":
            with pytest.raises(ValueError):
                _ = call_DELTA(fn, vec_d, vec_d_j.to("cuda"), eps)
        elif device.type == "cuda":
            with pytest.raises(ValueError):
                _ = call_DELTA(fn, vec_d, vec_d_j.to("cpu"), eps)

    # invalid: N < 2 (CPSF forbids N=1)
    v1 = rand_unit_vector(1, dtype, device)
    with pytest.raises(ValueError):
        _ = call_DELTA(fn, v1, v1, eps)


# ============================> TEST: D02 — Tangency (orthogonality to vec_d_j)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D02_tangency_orthogonality(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    vec_d = rand_unit_vector(N, dtype, device)
    vec_d_j = rand_unit_vector(N, dtype, device)
    delta = call_DELTA(fn, vec_d, vec_d_j, eps=1.0e-6)
    ip = torch.sum(vec_d_j.conj() * delta)
    assert (
        ip.abs().real.item() <= ZERO_ATOL_FACTOR * atol
    ), f"{impl_name}: <d_j, delta> not ~0 (N={N}, dtype={dtype}); |ip|={ip.abs().item():.3e}"

    gen = _gen_for(device)
    ar = torch.randn((), generator=gen, device=device, dtype=REAL)
    ai = torch.randn((), generator=gen, device=device, dtype=REAL)
    alpha = (ar + 1j * ai).to(dtype)

    lhs = torch.sum((alpha * vec_d_j).conj() * delta)
    rhs = alpha.conj() * ip
    assert torch.allclose(
        lhs, rhs, rtol=rtol, atol=PHASE_ATOL_FACTOR * atol
    ), f"{impl_name}: inner product conjugate-linearity failed (N={N}, dtype={dtype})"


# ============================> TEST: D03 — Phase equivariance in vec_d
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D03_phase_equivariance_in_vec_d(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64

    vec_d = rand_unit_vector(N, dtype, device)
    vec_d_j = rand_unit_vector(N, dtype, device)
    eps = 1.0e-6

    delta_base = call_DELTA(fn, vec_d, vec_d_j, eps)

    for phi_val in (0.0, 0.25, -0.7, 1.3):
        phi = torch.tensor(phi_val, dtype=REAL, device=device)
        phase = torch.exp(1j * phi.to(dtype=dtype))
        vec_d_phi = phase * vec_d

        delta_phi = call_DELTA(fn, vec_d_phi, vec_d_j, eps)
        expected = phase * delta_base

        assert torch.allclose(
            delta_phi, expected, rtol=rtol, atol=PHASE_ATOL_FACTOR * atol
        ), (
            f"{impl_name}: phase equivariance in vec_d failed for phi={phi_val} "
            f"(N={N}, dtype={dtype})"
        )


# ============================> TEST: D04 — Joint phase equivariance (vec_d and vec_d_j)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D04_joint_phase_equivariance(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64

    vec_d = rand_unit_vector(N, dtype, device)
    vec_d_j = rand_unit_vector(N, dtype, device)
    eps = 1.0e-6

    delta_base = call_DELTA(fn, vec_d, vec_d_j, eps)

    for psi_val in (0.0, 0.4, -1.1, 2.0):
        psi = torch.tensor(psi_val, dtype=REAL, device=device)
        phase = torch.exp(1j * psi.to(dtype=dtype))

        vec_d_psi = phase * vec_d
        vec_d_j_psi = phase * vec_d_j

        delta_psi = call_DELTA(fn, vec_d_psi, vec_d_j_psi, eps)
        expected = phase * delta_base

        assert torch.allclose(
            delta_psi, expected, rtol=rtol, atol=PHASE_ATOL_FACTOR * atol
        ), (
            f"{impl_name}: joint phase equivariance failed for psi={psi_val} "
            f"(N={N}, dtype={dtype})"
        )


# ============================> TEST: D05 — Zero at collinearity (fixed point)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D05_zero_at_collinearity(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    _, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    eps = 1.0e-6
    vec_d_j = rand_unit_vector(N, dtype, device)

    # Case 1: identical directions
    delta_same = call_DELTA(fn, vec_d_j, vec_d_j, eps)
    assert (
        torch.isfinite(delta_same.real).all() and torch.isfinite(delta_same.imag).all()
    ), f"{impl_name}: NaN/Inf in delta for identical vectors (N={N}, dtype={dtype})"
    nrm = torch.linalg.vector_norm(delta_same).real.item()
    assert nrm <= ZERO_ATOL_FACTOR * atol, (
        f"{impl_name}: ||delta|| not ~0 for identical vectors (||delta||={nrm:.3e}) "
        f"(N={N}, dtype={dtype})"
    )

    # Case 2: collinear via global phase
    for phi_val in (0.3, -1.1, 2.5):
        phi = torch.tensor(phi_val, dtype=REAL, device=device)
        phase = torch.exp(1j * phi.to(dtype=dtype))
        vec_d = phase * vec_d_j
        delta_col = call_DELTA(fn, vec_d, vec_d_j, eps)

        assert (
            torch.isfinite(delta_col.real).all()
            and torch.isfinite(delta_col.imag).all()
        ), f"{impl_name}: NaN/Inf in delta for phase-collinear vectors (phi={phi_val}, N={N}, dtype={dtype})"
        nrm_col = torch.linalg.vector_norm(delta_col).real.item()
        assert nrm_col <= ZERO_ATOL_FACTOR * atol, (
            f"{impl_name}: ||delta|| not ~0 for phase-collinear vectors (phi={phi_val}, "
            f"||delta||={nrm_col:.3e}) (N={N}, dtype={dtype})"
        )


# ============================> TEST: D06 — Norm upper bound by theta
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D06_norm_upper_bounded_by_theta(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    _, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    eps = 1.0e-6

    for _ in range(3):
        vec_d = rand_unit_vector(N, dtype, device)
        vec_d_j = rand_unit_vector(N, dtype, device)
        delta = call_DELTA(fn, vec_d, vec_d_j, eps)
        nrm = torch.linalg.vector_norm(delta).to(REAL)
        ip = torch.sum(vec_d_j.conj() * vec_d)
        c = ip.abs().to(REAL).clamp(0.0, 1.0)
        theta = torch.acos(c)

        assert (nrm <= theta + ZERO_ATOL_FACTOR * atol).all().item(), (
            f"{impl_name}: ||delta|| > theta + tol "
            f"(||delta||={nrm.item():.3e}, theta={theta.item():.3e}, "
            f"N={N}, dtype={dtype})"
        )


# ============================> TEST: D07 — Asymptotic regime (away from regularization)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D07_asymptotic_large_angle_match(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    eps = 1.0e-6
    min_theta = 10.0 * (eps**0.5)
    gen = _gen_for(device)

    def sample_pair_with_theta():
        for _ in range(50):
            dj = rand_unit_vector(N, dtype, device)
            d = rand_unit_vector(N, dtype, device)
            ip = torch.sum(dj.conj() * d)
            c = ip.abs().to(REAL).clamp(0.0, 1.0)
            th = torch.acos(c).item()

            if th >= min_theta:
                return d, dj, th

        dj = rand_unit_vector(N, dtype, device)
        zr = torch.randn(N, generator=gen, device=device, dtype=REAL)
        zi = torch.randn(N, generator=gen, device=device, dtype=REAL)
        z = (zr + 1j * zi).to(dtype)
        z = z - torch.sum(dj.conj() * z) * dj
        nrm = torch.linalg.vector_norm(z)

        if nrm.real.item() < torch.finfo(REAL).eps:
            z = torch.zeros(N, dtype=dtype, device=device)
            z[0] = 1.0
            z = z - torch.sum(dj.conj() * z) * dj
            nrm = torch.linalg.vector_norm(z)

        d = (z / nrm).to(dtype)
        th = (torch.pi / 2).item()

        return d, dj, th

    for _ in range(3):
        vec_d, vec_d_j, theta_val = sample_pair_with_theta()

        delta = call_DELTA(fn, vec_d, vec_d_j, eps)
        nrm = torch.linalg.vector_norm(delta).to(REAL).item()
        theta = torch.tensor(theta_val, dtype=REAL, device=device).item()

        err = abs(nrm - theta)
        thr = max(ASYMPT_RTL_MULT * rtol * theta, 10 * atol)
        assert err <= thr, (
            f"{impl_name}: asymptotic mismatch ||delta||~theta failed "
            f"(||delta||={nrm:.6e}, theta={theta:.6e}, err={err:.3e}, thr={thr:.3e}, "
            f"N={N}, dtype={dtype})"
        )


# ============================> TEST: D08 — Smoothness (finite difference contraction)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D08_smoothness_finite_difference(impl_name, fn, dtype, N):
    if N < 2:
        pytest.skip("Directional smoothness is non-informative for N=1.")

    device = TARGET_DEVICE
    _, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    step_eps = 3e-4 if dtype == torch.complex64 else 1e-6
    step_eps_half = step_eps * 0.5
    eps_model = 1.0e-6
    vec_d_j = rand_unit_vector(N, dtype, device)
    vec_d = rand_unit_vector(N, dtype, device)
    gen = _gen_for(device)
    zr = torch.randn(N, generator=gen, device=device, dtype=REAL)
    zi = torch.randn(N, generator=gen, device=device, dtype=REAL)
    z = (zr + 1j * zi).to(dtype)
    h = z - torch.sum(vec_d.conj() * z) * vec_d
    hn = torch.linalg.vector_norm(h)

    if hn.real.item() < torch.finfo(h.real.dtype).eps:
        e = torch.zeros(N, dtype=dtype, device=device)
        e[0] = 1.0
        h = e - torch.sum(vec_d.conj() * e) * vec_d
        hn = torch.linalg.vector_norm(h)

    h = h / hn
    delta0 = call_DELTA(fn, vec_d, vec_d_j, eps_model)

    def fwd(step: float) -> float:
        d_eps = vec_d + step * h
        d_eps = d_eps / torch.linalg.vector_norm(d_eps)
        delta_eps = call_DELTA(fn, d_eps, vec_d_j, eps_model)
        diff = torch.linalg.vector_norm((delta_eps - delta0)).real.item()
        return diff

    d1 = fwd(step_eps)
    d2 = fwd(step_eps_half)

    assert d1 / step_eps < 1.0 / (atol + 1e-30), (
        f"{impl_name}: finite-diff slope too large at step={step_eps}, "
        f"N={N}, dtype={dtype}, val={d1/step_eps:.3e}"
    )
    assert d2 <= 0.8 * d1 + 10 * atol, (
        f"{impl_name}: finite-diff not contracting (d2={d2:.3e}, d1={d1:.3e}), "
        f"N={N}, dtype={dtype}"
    )


# ============================> TEST: D09 — Batch semantics / broadcasting
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)])
def test_D09_batch_semantics_and_broadcast(impl_name, fn, dtype, N, batch_shape):
    device = TARGET_DEVICE
    _, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    eps = 1.0e-6
    gen = _gen_for(device)

    def rand_unit_batch(Bshape, n, dtype, device):
        xr = torch.randn(*Bshape, n, generator=gen, device=device, dtype=REAL)
        xi = torch.randn(*Bshape, n, generator=gen, device=device, dtype=REAL)
        v = (xr + 1j * xi).to(dtype)
        nrm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        nrm = torch.where(nrm.real <= torch.finfo(REAL).eps, torch.ones_like(nrm), nrm)
        return v / nrm

    vec_d_b = rand_unit_batch(batch_shape, N, dtype, device)
    vec_d_j_b = rand_unit_batch(batch_shape, N, dtype, device)
    delta_b = call_DELTA(fn, vec_d_b, vec_d_j_b, eps)

    assert delta_b.shape == (*batch_shape, N)
    assert delta_b.dtype == dtype
    assert delta_b.device.type == device.type

    ip = torch.sum(vec_d_j_b.conj() * delta_b, dim=-1)
    assert torch.all(
        ip.abs().real <= ZERO_ATOL_FACTOR * atol
    ), f"{impl_name}: some batch samples violate tangency (N={N}, dtype={dtype})"

    ip_dd = torch.sum(vec_d_j_b.conj() * vec_d_b, dim=-1)
    c = ip_dd.abs().to(REAL).clamp(0.0, 1.0)
    theta = torch.acos(c)
    nrm = torch.linalg.vector_norm(delta_b, dim=-1).to(REAL)
    assert torch.all(
        nrm <= theta + ZERO_ATOL_FACTOR * atol
    ), f"{impl_name}: some batch samples violate ||delta|| <= theta + tol (N={N}, dtype={dtype})"

    delta_b2 = call_DELTA(fn, vec_d_b.clone(), vec_d_j_b.clone(), eps)
    assert torch.equal(
        delta_b, delta_b2
    ), f"{impl_name}: non-deterministic outputs on repeated call (N={N}, dtype={dtype})"


# ============================> TEST: D10 — Deterministic behavior (no RNG influence)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D10_deterministic_behavior_no_rng(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    eps = 1.0e-6
    vec_d = rand_unit_vector(N, dtype, device)
    vec_d_j = rand_unit_vector(N, dtype, device)
    out1 = call_DELTA(fn, vec_d, vec_d_j, eps)
    out2 = call_DELTA(fn, vec_d.clone(), vec_d_j.clone(), eps)

    if device.type == "cpu":
        assert torch.equal(
            out1, out2
        ), f"{impl_name}: non-deterministic on CPU (N={N}, dtype={dtype})"
    else:
        assert torch.allclose(
            out1, out2, rtol=rtol, atol=atol
        ), f"{impl_name}: non-deterministic on {device.type} (N={N}, dtype={dtype})"

    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(98765)

    _ = torch.randn(8, device=device, dtype=torch.float32)
    out3 = call_DELTA(fn, vec_d, vec_d_j, eps)

    if device.type == "cpu":
        assert torch.equal(
            out1, out3
        ), f"{impl_name}: RNG state influenced output on CPU (N={N}, dtype={dtype})"
    else:
        assert torch.allclose(
            out1, out3, rtol=rtol, atol=atol
        ), f"{impl_name}: RNG state influenced output on {device.type} (N={N}, dtype={dtype})"


# =========================
# Direct run help.
# =========================
if __name__ == "__main__":
    print("\n")
    print("Use pytest to run:")
    print("\tpytest -q ./test.file.name.py")
    print("\n")
