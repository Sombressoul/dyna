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
    """
    Requirements (no body yet):
      - Inputs: vec_d, vec_d_j must have matching shapes [..., N], same dtype/device; N >= 2.
      - eps > 0 is required (reject eps <= 0).
      - Output shape must be [..., N]; dtype/device identical to inputs.
      - Function must not normalize inputs; assumes unit-norm provided by caller.
      Tolerances: exact checks on shape/dtype/device; no numeric compares here.
    """
    pass


# ============================> TEST: D02 — Tangency (orthogonality to vec_d_j)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D02_tangency_orthogonality(impl_name, fn, dtype, N):
    """
    Requirements (no body yet):
      - For random unit vec_d, vec_d_j: <vec_d_j, delta> ~ 0 (Hermitian inner product).
      - Use tolerance: |<vec_d_j, delta>| <= ZERO_ATOL_FACTOR * atol.
      - Also verify linearity of the inner product in complex arithmetic is respected.
    """
    pass


# ============================> TEST: D03 — Phase equivariance in vec_d
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D03_phase_equivariance_in_vec_d(impl_name, fn, dtype, N):
    """
    Requirements (no body yet):
      - For phi in a small set, delta(exp(i*phi)*vec_d, vec_d_j) == exp(i*phi)*delta(vec_d, vec_d_j).
      - Compare with (rtol, PHASE_ATOL_FACTOR*atol) on all components.
      - Norm invariance under phase follows implicitly.
    """
    pass


# ============================> TEST: D04 — Joint phase equivariance (vec_d and vec_d_j)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D04_joint_phase_equivariance(impl_name, fn, dtype, N):
    """
    Requirements (no body yet):
      - For psi in a small set, delta(exp(i*psi)*vec_d, exp(i*psi)*vec_d_j) == exp(i*psi)*delta(vec_d, vec_d_j).
      - Compare with (rtol, PHASE_ATOL_FACTOR*atol).
    """
    pass


# ============================> TEST: D05 — Zero at collinearity (fixed point)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D05_zero_at_collinearity(impl_name, fn, dtype, N):
    """
    Requirements (no body yet):
      - If vec_d == vec_d_j (or vec_d = exp(i*phi)*vec_d_j), we must have delta == 0 vector.
      - Norm bound: ||delta|| <= ZERO_ATOL_FACTOR * atol.
      - No NaN/Inf in output.
    """
    pass


# ============================> TEST: D06 — Norm upper bound by theta
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D06_norm_upper_bounded_by_theta(impl_name, fn, dtype, N):
    """
    Requirements (no body yet):
      - Compute theta = arccos(|<vec_d_j, vec_d>|).
      - Must have ||delta|| <= theta + ZERO_ATOL_FACTOR * atol.
      - Enforce for random pairs of unit inputs.
    """
    pass


# ============================> TEST: D07 — Asymptotic regime (away from regularization)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D07_asymptotic_large_angle_match(impl_name, fn, dtype, N):
    """
    Requirements (no body yet):
      - Pick pairs with theta >= 10*sqrt(eps) (or choose eps small enough).
      - Expect ||delta|| ~ theta.
      - Check | ||delta|| - theta | <= max(ASYMPT_RTL_MULT*rtol*theta, 10*atol).
    """
    pass


# ============================> TEST: D08 — Smoothness (finite difference contraction)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D08_smoothness_finite_difference(impl_name, fn, dtype, N):
    """
    Requirements (no body yet):
      - Fix vec_d_j; choose base vec_d and tangent h (orthogonal to vec_d).
      - Define Delta(eps) = || delta(vec_d + eps*h) - delta(vec_d) ||_2.
      - For eps64 ~ 3e-4, eps128 ~ 1e-6: require Delta(eps/2) <= 0.8*Delta(eps) + 10*atol.
      - Also assert Delta(eps)/eps < 1/(atol + 1e-30).
    """
    pass


# ============================> TEST: D09 — Batch semantics / broadcasting
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)])
def test_D09_batch_semantics_and_broadcast(impl_name, fn, dtype, N, batch_shape):
    """
    Requirements (no body yet):
      - For batched inputs (*B, N): output is shape (*B, N), same dtype/device.
      - For each batch sample, D02 and D06 must hold with same tolerances.
      - Deterministic behavior across the batch (no RNG usage inside).
    """
    pass


# ============================> TEST: D10 — Deterministic behavior (no RNG influence)
@pytest.mark.parametrize("impl_name,fn", DELTA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_D10_deterministic_behavior_no_rng(impl_name, fn, dtype, N):
    """
    Requirements (no body yet):
      - Repeated calls with identical inputs must return identical outputs (bitwise on CPU;
        numerically identical within (rtol, atol) on CUDA if needed).
      - Changing global RNG seeds between calls must not affect outputs.
    """
    pass


# =========================
# Direct run help.
# =========================
if __name__ == "__main__":
    print("\n")
    print("Use pytest to run:")
    print("\tpytest -q ./test.file.name.py")
    print("\n")
