# Run as (example):
# > pytest -q dyna/lib/cpsf/pytest/test_CPSF_psi_over_offsets.py

import torch
import pytest

from dyna.lib.cpsf.functional.core_math import (
    R,
    R_ext,
    q,
    rho,
    iota,
    lift,
    delta_vec_d,
    psi_over_offsets,
)
from dyna.lib.cpsf.periodization import CPSFPeriodization


# =========================
# Global config
# =========================
TARGET_DEVICE = torch.device("cpu")
DTYPES = [torch.complex64, torch.complex128]
NS = [2, 3]
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
    g = _gen_for(device)
    xr = torch.randn(n, generator=g, device=device, dtype=torch.float64)
    xi = torch.randn(n, generator=g, device=device, dtype=torch.float64)
    v = (xr + 1j * xi).to(dtype)
    nrm = torch.linalg.vector_norm(v)
    if float(nrm.real) < torch.finfo(v.real.dtype).eps:
        v = torch.zeros_like(v)
        v[0] = 1
        return v
    return v / nrm


def make_problem(
    N: int,
    M: int,
    dtype: torch.dtype,
    device: torch.device = TARGET_DEVICE,
):
    z = rand_unit_vector(N, dtype=dtype, device=device)
    vec_d_single = rand_unit_vector(N, dtype=dtype, device=device)

    z_j = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)], dim=0
    )
    vec_d_j = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)], dim=0
    )

    vec_d = vec_d_single.unsqueeze(0).expand(M, N).clone()
    rdt = torch.float32 if dtype == torch.complex64 else torch.float64
    g = _gen_for(device)
    sigma_par = torch.rand(M, generator=g, device=device, dtype=rdt) * 0.9 + 0.1
    sigma_perp = torch.rand(M, generator=g, device=device, dtype=rdt) * 0.9 + 0.1

    return z, z_j, vec_d, vec_d_j, sigma_par, sigma_perp


def available_devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


# =========================
# Tests
# =========================
# =========================
# P01 — basic shape/dtype/device and reduction over offsets
# =========================
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P01_shape_dtype_device_and_basic(dtype, N):
    device = TARGET_DEVICE
    M = 5

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=0, device=device, sorted=False)

    out = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    assert out.shape == (M,), "psi_over_offsets must return [..., M]"
    assert out.device.type == device.type, "Output device must match inputs"
    expected_real = torch.float32 if dtype == torch.complex64 else torch.float64
    assert out.dtype == expected_real, "rho(q) is real-valued"
    assert torch.isfinite(out).all(), "Output must be finite"
    assert (out >= 0).all(), "rho(q) = exp(-pi*q), q>=0 => output >= 0"


# =========================
# P02 — explicit loop over offsets equals vectorized psi_over_offsets
# =========================
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P02_matches_explicit_loop_over_offsets(dtype, N):
    device = TARGET_DEVICE
    M = 4
    W = 1  # small but non-trivial window

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    # Vectorized reference
    fast = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    # Explicit Python loop reference
    dz0 = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rm = R(vec_d_j)
    Rex = R_ext(Rm)

    slow = torch.zeros(M, dtype=fast.dtype, device=device)

    for k in range(offsets.shape[0]):
        row = offsets[k]
        n_r = row[:N].to(dtype=dz0.real.dtype)
        n_i = row[N:].to(dtype=dz0.real.dtype)
        n_c = torch.complex(n_r, n_i).to(dtype=dtype)

        dz_k = dz0 + n_c.unsqueeze(0)
        w = iota(dz_k, dd)
        qv = q(w=w, R_ext=Rex, sigma_par=sp, sigma_perp=sq)
        slow = slow + rho(qv)

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        fast, slow, rtol=rtol, atol=atol
    ), "Vectorized psi_over_offsets must equal explicit sum"


# =========================
# P03 — Signature compatibility & return shape
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P03_signature_and_return_shape(device, dtype, N):
    M = 5

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    # Valid offsets: Z^{2N}, from real periodization generator
    per = CPSFPeriodization()
    offsets = per.window(N=N, W=1, device=device, sorted=False)  # [O, 2N]

    out = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    # Must return [..., M]
    assert out.shape == (
        M,
    ), "psi_over_offsets must reduce over offsets and return shape [..., M]"


# =========================
# P04 — Output dtype mapping (real)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P04_output_dtype_is_real_and_matches_input_precision(device, dtype, N):
    M = 3

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=0, device=device, sorted=False)  # single zero offset

    out = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    expected_real = torch.float32 if dtype == torch.complex64 else torch.float64
    assert (
        out.dtype == expected_real
    ), "Output must be real: float32 for c64, float64 for c128"
    assert not torch.is_complex(out), "Output must not be complex"


# =========================
# P05 — Output device equals z.device
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P05_output_device_matches_input(device, dtype, N):
    M = 4

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=1, device=device, sorted=False)

    out = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    assert out.device.type == device.type, "Output must reside on the same device as z"


# =========================
# P06 — offsets must shift BOTH Re and Im parts (imag-only case <> z imaginary shift)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P06_offsets_apply_to_imag_part(device, dtype, N):
    M = 4
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    zero_offsets = per.window(N=N, W=0, device=device, sorted=False)

    imag_offsets = zero_offsets.clone()
    imag_offsets[0, N + 0] = 1

    out_off = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=imag_offsets,
        R_j=None,
        q_max=None,
    )

    e0 = torch.zeros(N, dtype=z.real.dtype, device=device)
    e0[0] = 1
    z_shift = torch.complex(z.real, z.imag + e0)

    out_shift = psi_over_offsets(
        z=z_shift,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=zero_offsets,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        out_off, out_shift, rtol=rtol, atol=atol
    ), "Imaginary offsets must be equivalent to shifting Im(z) by the same integer amount"


# =========================
# P07 — offsets must shift BOTH Re and Im parts (real-only case <> z real shift)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P07_offsets_apply_to_real_part(device, dtype, N):
    M = 4
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    zero_offsets = per.window(N=N, W=0, device=device, sorted=False)

    real_offsets = zero_offsets.clone()
    real_offsets[0, 0] = 1

    out_off = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=real_offsets,
        R_j=None,
        q_max=None,
    )

    e0 = torch.zeros(N, dtype=z.real.dtype, device=device)
    e0[0] = 1
    z_shift = torch.complex(z.real + e0, z.imag)

    out_shift = psi_over_offsets(
        z=z_shift,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=zero_offsets,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        out_off, out_shift, rtol=rtol, atol=atol
    ), "Real offsets must be equivalent to shifting Re(z) by the same integer amount"


# =========================
# P08 — zero-offset window equals a single envelope term (no extra phases)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P08_zero_offset_window_is_single_envelope_term(device, dtype, N):
    M = 5
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    zero_offsets = per.window(N=N, W=0, device=device, sorted=False)

    eta_zero = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=zero_offsets,
        R_j=None,
        q_max=None,
    )

    dz0 = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rm = R(vec_d_j)
    Rex = R_ext(Rm)
    w = iota(dz0, dd)
    qv = q(w=w, R_ext=Rex, sigma_par=sp, sigma_perp=sq)
    eta_manual = rho(qv)

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_zero, eta_manual, rtol=rtol, atol=atol
    ), "Zero-offset window must equal the single unshifted envelope term"


# =========================
# P09 — Leading dims broadcasting: result shape must be [..., M] with leading batch dims preserved
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P09_leading_dims_broadcast(device, dtype, N):
    B = 2
    M = 4

    z = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(B)], dim=0
    ).unsqueeze(1)

    z_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )

    vec_d_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )

    vec_d_query = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(B)], dim=0
    )
    vec_d = vec_d_query.unsqueeze(1).expand(B, M, N).clone()

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    sigma_par = torch.rand(B, M, device=device, dtype=real_dtype) * 0.9 + 0.1
    sigma_perp = torch.rand(B, M, device=device, dtype=real_dtype) * 0.9 + 0.1

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=1, device=device, sorted=False)

    out = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    assert out.shape == (
        B,
        M,
    ), "Result must preserve leading batch dims and end with M contributors"
    assert out.device.type == device.type
    expected_real = torch.float32 if dtype == torch.complex64 else torch.float64
    assert out.dtype == expected_real
    assert torch.isfinite(out).all()


# =========================
# P10 — Contributors axis: trailing dimension must equal M for various M
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("M", [1, 2, 5])
def test_P10_contributors_axis_shape(device, dtype, N, M):
    B = 3

    z = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(B)], dim=0
    ).unsqueeze(1)

    z_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )
    vec_d_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )

    vec_d_query = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(B)], dim=0
    )
    vec_d = vec_d_query.unsqueeze(1).expand(B, M, N).clone()

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    sigma_par = torch.rand(B, M, device=device, dtype=real_dtype) * 0.9 + 0.1
    sigma_perp = torch.rand(B, M, device=device, dtype=real_dtype) * 0.9 + 0.1

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=1, device=device, sorted=False)

    out = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    assert out.shape[:-1] == (B,), "All leading dims must be preserved"
    assert out.shape[-1] == M, "Trailing axis must equal the number of contributors M"


# =========================
# P11 — No implicit broadcasting inside delta_vec_d: mismatched shapes must raise
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P11_no_implicit_broadcast_in_delta_vec_d(device, dtype, N):
    B = 2
    M = 4

    z = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(B)], dim=0
    ).unsqueeze(1)

    z_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )

    vec_d_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )

    vec_d_query = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(B)], dim=0
    )
    vec_d_bad = vec_d_query.unsqueeze(1)

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    sigma_par = torch.rand(B, M, device=device, dtype=real_dtype) * 0.9 + 0.1
    sigma_perp = torch.rand(B, M, device=device, dtype=real_dtype) * 0.9 + 0.1

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=0, device=device, sorted=False)

    with pytest.raises(ValueError):
        _ = psi_over_offsets(
            z=z,
            z_j=z_j,
            vec_d=vec_d_bad,
            vec_d_j=vec_d_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            offsets=offsets,
            R_j=None,
            q_max=None,
        )


# =========================
# P12 — q must be real and non-negative for each offset term
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P12_q_is_real_and_nonnegative(device, dtype, N):
    M = 4
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=1, device=device, sorted=False)

    dz0 = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rm = R(vec_d_j)
    Rex = R_ext(Rm)

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

    for k in range(offsets.shape[0]):
        row = offsets[k]
        n_r = row[:N].to(dtype=dz0.real.dtype)
        n_i = row[N:].to(dtype=dz0.real.dtype)
        n_c = torch.complex(n_r, n_i).to(dtype=dtype)

        dz_k = dz0 + n_c.unsqueeze(0)
        w = iota(dz_k, dd)
        qv = q(w=w, R_ext=Rex, sigma_par=sp, sigma_perp=sq)

        assert qv.dtype == real_dtype, "q must be real-valued"
        assert torch.isfinite(qv).all(), "q must be finite"
        assert (qv >= 0).all(), "q must be non-negative"


# =========================
# P13 — rho(q) is real; strictly positive where no underflow is expected; vectorized sum matches explicit sum
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P13_rho_is_real_positive_and_matches_sum(device, dtype, N):
    M = 3
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=1, device=device, sorted=False)

    eta_vec = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    dz0 = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rm = R(vec_d_j)
    Rex = R_ext(Rm)

    eta_sum = torch.zeros(M, dtype=eta_vec.dtype, device=device)

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    tiny = torch.finfo(real_dtype).tiny
    import math

    q_thresh = (-math.log(float(tiny))) / math.pi
    q_safe = 0.5 * q_thresh  # conservative margin to avoid denorm/underflow

    for k in range(offsets.shape[0]):
        row = offsets[k]
        n_r = row[:N].to(dtype=dz0.real.dtype)
        n_i = row[N:].to(dtype=dz0.real.dtype)
        n_c = torch.complex(n_r, n_i).to(dtype=dtype)

        dz_k = dz0 + n_c.unsqueeze(0)
        w = iota(dz_k, dd)
        qv = q(w=w, R_ext=Rex, sigma_par=sp, sigma_perp=sq)
        rv = rho(qv)

        assert not torch.is_complex(rv)
        assert torch.isfinite(rv).all(), "rho(q) must be finite"
        assert (rv >= 0).all(), "rho(q) must be non-negative"

        # Strict positivity in the no-underflow zone
        mask_safe = qv <= q_safe
        if mask_safe.any():
            assert (
                rv[mask_safe] > 0
            ).all(), "rho(q) must be strictly positive when q is well below the dtype underflow threshold"

        eta_sum = eta_sum + rv

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_vec, eta_sum, rtol=rtol, atol=atol
    ), "psi_over_offsets must equal the sum over rho(q) for all offsets"


# =========================
# P13a — rho(q) underflow edge-case: zero-offset must be strictly positive (after sigma upscaling),
#        and there exists a large K where rho(q) underflows to exact zero.
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P13a_rho_underflow_edge_cases(device, dtype, N):
    M = 3

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    zero_offsets = per.window(N=N, W=0, device=device, sorted=False)

    # If rho already underflows at zero-offset, iteratively increase sigmas.
    # Monotonicity: larger sigmas => smaller q => larger rho.
    attempts = 0
    out0 = None
    while attempts < 8:
        out0 = psi_over_offsets(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            sigma_par=sp,
            sigma_perp=sq,
            offsets=zero_offsets,
            R_j=None,
            q_max=None,
        )
        if torch.isfinite(out0).all() and (out0 > 0).all():
            break
        # Upscale sigmas by x10 and retry
        sp = sp * 10.0
        sq = sq * 10.0
        attempts += 1

    assert out0 is not None
    assert torch.isfinite(out0).all(), "Zero-offset output must be finite"
    assert (out0 > 0).all(), (
        "Zero-offset output must be strictly positive before testing underflow; "
        "sigma upscaling should have prevented premature underflow"
    )

    # Try exponentially growing K; one of them MUST underflow for float32/float64.
    K_candidates = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]

    out_big = None
    hit = False
    for K in K_candidates:
        offsets = zero_offsets.clone()
        offsets[0, 0] = K  # shift along Re component of dim 0
        outK = psi_over_offsets(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            sigma_par=sp,
            sigma_perp=sq,
            offsets=offsets,
            R_j=None,
            q_max=None,
        )
        assert torch.isfinite(
            outK
        ).all(), "Output must remain finite for extreme offsets"
        if torch.all(outK == 0):
            out_big = outK
            hit = True
            break

    assert (
        hit
    ), "Failed to reach underflow region: increase search range for K or adjust scaling strategy."
    assert torch.all(
        out_big == 0
    ), "rho(q) should underflow to exact zero for sufficiently large K"


# =========================
# P14 — Finite outputs for reasonable windows/shells (no NaN/Inf)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P14_finite_outputs_for_reasonable_windows_and_shells(device, dtype, N):
    M = 5
    W = 2

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)
    per = CPSFPeriodization()

    offsets_win = per.window(N=N, W=W, device=device, sorted=False)
    out_win = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_win,
        R_j=None,
        q_max=None,
    )
    assert torch.isfinite(out_win).all(), "Window-based output must be finite"

    for r in range(W + 1):
        offsets_sh = per.shell(N=N, W=r, device=device, sorted=False)
        out_sh = psi_over_offsets(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            sigma_par=sp,
            sigma_perp=sq,
            offsets=offsets_sh,
            R_j=None,
            q_max=None,
        )
        assert torch.isfinite(out_sh).all(), f"Shell W={r} output must be finite"


# =========================
# P15 — Row-order invariance: any permutation of offsets rows leaves result unchanged
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P15_row_order_invariance(device, dtype, N):
    M = 4
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=2, device=device, sorted=False)

    eta = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    idx = torch.arange(offsets.shape[0] - 1, -1, -1, device=device)
    offsets_perm = offsets.index_select(0, idx)

    eta_perm = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_perm,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta, eta_perm, rtol=rtol, atol=atol
    ), "Row order must not affect the sum"


# =========================
# P16 — Duplicates linearity: duplicating a row adds exactly that row's contribution
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P16_duplicates_linearity(device, dtype, N):
    M = 3
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=1, device=device, sorted=False)

    eta = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    # Choose a row to duplicate (first row is fine)
    row = offsets[0:1, :]
    eta_row = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=row,
        R_j=None,
        q_max=None,
    )

    offsets_dup = torch.cat([offsets, row], dim=0)
    eta_dup = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_dup,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_dup, eta + eta_row, rtol=rtol, atol=atol
    ), "Duplicating an offset row must add exactly its contribution"


# =========================
# P17 — Shift + reindex (pure real): η(z; O) == η(z+m_R; O-(m_R, 0))
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P17_shift_reindex_pure_real(device, dtype, N):
    M = 4
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=2, device=device, sorted=False)

    eta0 = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    m_R = torch.zeros(N, dtype=torch.long, device=device)
    m_R[0] = 1

    z_shift = torch.complex(z.real + m_R.to(z.real.dtype), z.imag)

    offsets_shift = offsets.clone()
    offsets_shift[:, :N] -= m_R

    eta1 = psi_over_offsets(
        z=z_shift,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_shift,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta0, eta1, rtol=rtol, atol=atol
    ), "Pure real shift + reindex invariance failed"


# =========================
# P18 — Shift + reindex (pure imaginary): η(z; O) == η(z+i*m_I; O-(0, m_I))
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P18_shift_reindex_pure_imag(device, dtype, N):
    M = 4
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=2, device=device, sorted=False)

    eta0 = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    m_I = torch.zeros(N, dtype=torch.long, device=device)
    m_I[0] = 1

    z_shift = torch.complex(z.real, z.imag + m_I.to(z.real.dtype))

    offsets_shift = offsets.clone()
    offsets_shift[:, N:] -= m_I

    eta1 = psi_over_offsets(
        z=z_shift,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_shift,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta0, eta1, rtol=rtol, atol=atol
    ), "Pure imaginary shift + reindex invariance failed"


# =========================
# P19 — Shift + reindex (mixed): η(z; O) == η(z+m_R+i*m_I; O-(m_R, m_I))
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P19_shift_reindex_mixed(device, dtype, N):
    M = 4
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=2, device=device, sorted=False)

    eta0 = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    m_R = torch.zeros(N, dtype=torch.long, device=device)
    m_R[0] = 1
    idx_im = 1 if N > 1 else 0
    m_I = torch.zeros(N, dtype=torch.long, device=device)
    m_I[idx_im] = 1

    z_shift = torch.complex(
        z.real + m_R.to(z.real.dtype), z.imag + m_I.to(z.real.dtype)
    )

    offsets_shift = offsets.clone()
    offsets_shift[:, :N] -= m_R
    offsets_shift[:, N:] -= m_I

    eta1 = psi_over_offsets(
        z=z_shift,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_shift,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta0, eta1, rtol=rtol, atol=atol
    ), "Mixed (real+imag) shift + reindex invariance failed"


# =========================
# P20 — Split/merge streaming equivalence: sum over union equals sum of partial results
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P20_split_merge_streaming_equivalence(device, dtype, N):
    M = 5
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets_full = per.window(N=N, W=2, device=device, sorted=False)
    O = offsets_full.shape[0]
    assert O >= 3, "W=2 should produce at least 3 rows; otherwise increase W"

    eta_full = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_full,
        R_j=None,
        q_max=None,
    )

    a = O // 3
    b = 2 * O // 3
    chunks = [offsets_full[:a], offsets_full[a:b], offsets_full[b:]]

    eta_parts = torch.zeros_like(eta_full)
    for off in chunks:
        if off.numel() == 0:
            continue
        eta_parts = eta_parts + psi_over_offsets(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            sigma_par=sp,
            sigma_perp=sq,
            offsets=off,
            R_j=None,
            q_max=None,
        )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_full, eta_parts, rtol=rtol, atol=atol
    ), "Sum over a union of offsets must equal the sum of partial results over any partition"


# =========================
# P21 — Window vs shells (set equality): window(N, W) equals concat(shell(N, w), w=0..W)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", [0, 1, 2])
def test_P21_window_vs_shells_set_equality(device, dtype, N, W):
    M = 4
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets_win = per.window(N=N, W=W, device=device, sorted=False)

    shells = [per.shell(N=N, W=w, device=device, sorted=False) for w in range(W + 1)]
    offsets_cat = (
        torch.cat(shells, dim=0)
        if shells
        else torch.empty(0, 2 * N, dtype=torch.long, device=device)
    )

    eta_win = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_win,
        R_j=None,
        q_max=None,
    )
    eta_cat = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_cat,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_win, eta_cat, rtol=rtol, atol=atol
    ), "window(N,W) must equal concatenation of shells w=0..W (order-agnostic)"


# =========================
# P22 — Iterators equivalence: iter_shells / pack_offsets / iter_packed all match window(N, W)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", [1, 2])
def test_P22_iterators_equivalence(device, dtype, N, W):
    M = 5
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets_win = per.window(N=N, W=W, device=device, sorted=False)

    eta_win = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_win,
        R_j=None,
        q_max=None,
    )

    eta_iter_shells = torch.zeros_like(eta_win)
    for w, shell in per.iter_shells(
        N=N, start_radius=0, max_radius=W, device=device, sorted=False
    ):
        assert w >= 0 and w <= W
        if shell.numel() == 0:
            continue
        eta_iter_shells = eta_iter_shells + psi_over_offsets(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            sigma_par=sp,
            sigma_perp=sq,
            offsets=shell,
            R_j=None,
            q_max=None,
        )

    offsets_pack, lengths = per.pack_offsets(
        N=N, max_radius=W, device=device, sorted=False
    )
    assert lengths.numel() == (
        W + 1
    ), "pack_offsets must provide lengths for each radius 0..W"
    eta_pack = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_pack,
        R_j=None,
        q_max=None,
    )

    eta_iter_packed = torch.zeros_like(eta_win)
    for w_start, w_end, pack in per.iter_packed(
        N=N,
        target_points_per_pack=max(1, offsets_win.shape[0] // 3),
        start_radius=0,
        max_radius=W,
        device=device,
        sorted=False,
    ):
        assert 0 <= w_start <= w_end <= W
        if pack.numel() == 0:
            continue
        eta_iter_packed = eta_iter_packed + psi_over_offsets(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            sigma_par=sp,
            sigma_perp=sq,
            offsets=pack,
            R_j=None,
            q_max=None,
        )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_win, eta_iter_shells, rtol=rtol, atol=atol
    ), "iter_shells accumulation must match window"
    assert torch.allclose(
        eta_win, eta_pack, rtol=rtol, atol=atol
    ), "pack_offsets result must match window"
    assert torch.allclose(
        eta_win, eta_iter_packed, rtol=rtol, atol=atol
    ), "iter_packed accumulation must match window"


# =========================
# P23 — Sorted flag neutrality: sorted=True/False must produce identical results for same set
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", [0, 1, 2])
def test_P23_sorted_flag_neutrality(device, dtype, N, W):
    M = 4
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets_unsorted = per.window(N=N, W=W, device=device, sorted=False)
    offsets_sorted = per.window(N=N, W=W, device=device, sorted=True)

    eta_unsorted = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_unsorted,
        R_j=None,
        q_max=None,
    )
    eta_sorted = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_sorted,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_unsorted, eta_sorted, rtol=rtol, atol=atol
    ), "sorted=True/False must be neutral for the same offsets set"


# =========================
# P24 — Vectorized vs scalar loop: psi_over_offsets equals explicit sum over offsets (within tolerances)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P24_vectorized_equals_scalar_loop(device, dtype, N):
    M = 4
    W = 1  # small but non-trivial window size to keep loop affordable

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    # Vectorized result
    eta_vec = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    # Scalar loop reference: sum over rho(q(iota(dz + n, dd))) term-by-term
    dz0 = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rm = R(vec_d_j)
    Rex = R_ext(Rm)

    eta_slow = torch.zeros(M, dtype=eta_vec.dtype, device=device)

    for k in range(offsets.shape[0]):
        row = offsets[k]
        n_r = row[:N].to(dtype=dz0.real.dtype)
        n_i = row[N:].to(dtype=dz0.real.dtype)
        n_c = torch.complex(n_r, n_i).to(dtype=dtype)

        dz_k = dz0 + n_c.unsqueeze(0)
        w = iota(dz_k, dd)
        qv = q(w=w, R_ext=Rex, sigma_par=sp, sigma_perp=sq)
        eta_slow = eta_slow + rho(qv)

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_vec, eta_slow, rtol=rtol, atol=atol
    ), "Vectorized psi_over_offsets must equal explicit scalar-loop accumulation over the same offsets"


# =========================
# P25 — Clamp location: clamping must be applied to q BEFORE rho
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P25_qmax_clamp_applied_before_rho(device, dtype, N):
    M = 4
    W = 2

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)
    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    dz0 = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rm = R(vec_d_j)
    Rex = R_ext(Rm)

    q_values = []
    for k in range(offsets.shape[0]):
        row = offsets[k]
        n_r = row[:N].to(dtype=dz0.real.dtype)
        n_i = row[N:].to(dtype=dz0.real.dtype)
        n_c = torch.complex(n_r, n_i).to(dtype=dtype)
        dz_k = dz0 + n_c.unsqueeze(0)
        w = iota(dz_k, dd)
        qk = q(w=w, R_ext=Rex, sigma_par=sp, sigma_perp=sq)
        q_values.append(qk)
    Q = torch.stack(q_values, dim=0)

    sQ, _ = torch.sort(Q.reshape(-1))
    idx_small = max(0, int(0.25 * (sQ.numel() - 1)))
    q_cap = sQ[idx_small].item()

    eta_clamped = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=q_cap,
    )

    q_cap_t = torch.tensor(q_cap, dtype=Q.dtype, device=device)
    Q_cl = torch.minimum(Q, q_cap_t)
    eta_manual = rho(Q_cl).sum(dim=0)

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_clamped, eta_manual, rtol=rtol, atol=atol
    ), "q_max must clamp q before applying rho"


# =========================
# P26 — Large cap no-op: sufficiently large q_max equals the unclamped result
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P26_qmax_large_cap_noop(device, dtype, N):
    M = 4
    W = 2

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)
    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    eta_unclamped = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    dz0 = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rm = R(vec_d_j)
    Rex = R_ext(Rm)

    q_max_val = torch.tensor(0.0, dtype=eta_unclamped.dtype, device=device)
    for k in range(offsets.shape[0]):
        row = offsets[k]
        n_r = row[:N].to(dtype=dz0.real.dtype)
        n_i = row[N:].to(dtype=dz0.real.dtype)
        n_c = torch.complex(n_r, n_i).to(dtype=dtype)
        dz_k = dz0 + n_c.unsqueeze(0)
        w = iota(dz_k, dd)
        qk = q(w=w, R_ext=Rex, sigma_par=sp, sigma_perp=sq)
        q_max_val = torch.maximum(q_max_val, qk.max().to(q_max_val.dtype))
    q_cap_large = float(q_max_val.item() * 10.0 + 1.0)

    eta_largecap = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=q_cap_large,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_unclamped, eta_largecap, rtol=rtol, atol=atol
    ), "A sufficiently large q_max must be a no-op and match the unclamped result"


# =========================
# P27 — Monotonicity in q_max: smaller q_max yields greater-or-equal eta than larger q_max
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P27_qmax_monotonicity(device, dtype, N):
    M = 4
    W = 2

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)
    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    dz0 = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rm = R(vec_d_j)
    Rex = R_ext(Rm)

    q_values = []
    for k in range(offsets.shape[0]):
        row = offsets[k]
        n_r = row[:N].to(dtype=dz0.real.dtype)
        n_i = row[N:].to(dtype=dz0.real.dtype)
        n_c = torch.complex(n_r, n_i).to(dtype=dtype)
        dz_k = dz0 + n_c.unsqueeze(0)
        w = iota(dz_k, dd)
        qk = q(w=w, R_ext=Rex, sigma_par=sp, sigma_perp=sq)
        q_values.append(qk)
    Q = torch.stack(q_values, dim=0).reshape(-1)

    sQ, _ = torch.sort(Q)
    if sQ.numel() < 4:
        pytest.skip("Not enough q samples to form meaningful quantiles")
    idx_small = int(0.25 * (sQ.numel() - 1))
    idx_big = int(0.75 * (sQ.numel() - 1))
    q_cap_small = float(sQ[idx_small].item())
    q_cap_big = float(sQ[idx_big].item())
    if not (q_cap_small <= q_cap_big):
        q_cap_small, q_cap_big = q_cap_big, q_cap_small

    eta_small = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=q_cap_small,
    )
    eta_big = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=q_cap_big,
    )

    rtol, atol = _get_tols(dtype)
    tol_band = rtol * torch.abs(eta_small) + atol
    assert torch.all(
        eta_small + tol_band >= eta_big
    ), "Monotonicity: decreasing q_max must not decrease eta (rho after min(q, q_max))"


# =========================
# P28 — Provided R_j parity: passing R(vec_d_j) must equal internal construction
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P28_provided_Rj_parity(device, dtype, N):
    B = 2
    M = 4

    z = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(B)], dim=0
    ).unsqueeze(1)
    z_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )
    vec_d_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )
    vec_d_query = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(B)], dim=0
    )
    vec_d = vec_d_query.unsqueeze(1).expand(B, M, N).clone()

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    sigma_par = torch.rand(B, M, device=device, dtype=real_dtype) * 0.9 + 0.1
    sigma_perp = torch.rand(B, M, device=device, dtype=real_dtype) * 0.9 + 0.1

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=2, device=device, sorted=False)

    eta_internal = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    Rj = R(vec_d_j)
    eta_provided = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
        offsets=offsets,
        R_j=Rj,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_internal, eta_provided, rtol=rtol, atol=atol
    ), "Providing R_j must yield identical results to internal R(vec_d_j)"


# =========================
# P29 — Broadcast of R_ext across offsets: psi_over_offsets (with provided R_j) equals explicit loop using R_ext.unsqueeze(-3)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P29_Rext_broadcast_over_offsets(device, dtype, N):
    B = 2
    M = 3
    W = 2

    z = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(B)], dim=0
    ).unsqueeze(1)
    z_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )
    vec_d_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )
    vec_d_query = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(B)], dim=0
    )
    vec_d = vec_d_query.unsqueeze(1).expand(B, M, N).clone()

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    sigma_par = torch.rand(B, M, device=device, dtype=real_dtype) * 0.9 + 0.1
    sigma_perp = torch.rand(B, M, device=device, dtype=real_dtype) * 0.9 + 0.1

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)
    O = offsets.shape[0]
    assert O > 1, "Need multiple offsets to validate broadcast across O"

    Rj = R(vec_d_j)
    eta_vec = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
        offsets=offsets,
        R_j=Rj,
        q_max=None,
    )

    dz0 = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rex = R_ext(Rj)

    eta_loop = torch.zeros_like(eta_vec)
    for k in range(O):
        row = offsets[k]
        n_r = row[:N].to(dtype=dz0.real.dtype)
        n_i = row[N:].to(dtype=dz0.real.dtype)
        n_c = torch.complex(n_r, n_i).to(dtype=dtype)

        dz_k = dz0 + n_c.view(1, 1, N).to(dz0.dtype)
        w = iota(dz_k, dd)
        qv = q(
            w=w,
            R_ext=Rex,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
        )
        eta_loop = eta_loop + rho(qv)

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_vec, eta_loop, rtol=rtol, atol=atol
    ), "Providing R_j must broadcast via R_ext over the offsets axis identically to the explicit loop"


# =========================
# P30 — Dtype matrix: complex128 vs complex64 must both work; results should be close (within c64 tolerances)
# =========================
@pytest.mark.parametrize("N", NS)
def test_P30_dtype_matrix_c128_vs_c64_parity(N):
    device = torch.device("cpu")
    M = 4
    W = 1

    z128, z_j128, vec_d128, vec_d_j128, sp128, sq128 = make_problem(
        N=N, M=M, dtype=torch.complex128, device=device
    )
    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    eta128 = psi_over_offsets(
        z=z128,
        z_j=z_j128,
        vec_d=vec_d128,
        vec_d_j=vec_d_j128,
        sigma_par=sp128,
        sigma_perp=sq128,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    z64 = z128.to(torch.complex64)
    z_j64 = z_j128.to(torch.complex64)
    vec_d64 = vec_d128.to(torch.complex64)
    vec_d_j64 = vec_d_j128.to(torch.complex64)
    sp64 = sp128.to(torch.float32)
    sq64 = sq128.to(torch.float32)

    eta64 = psi_over_offsets(
        z=z64,
        z_j=z_j64,
        vec_d=vec_d64,
        vec_d_j=vec_d_j64,
        sigma_par=sp64,
        sigma_perp=sq64,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(torch.complex64)
    assert torch.allclose(
        eta64.to(torch.float64), eta128, rtol=rtol, atol=atol
    ), "complex64 and complex128 outputs must be numerically close for identical inputs"


# =========================
# P31 — Sigma dtype tolerance: scalars (0-D tensors) vs per-contributor vectors must produce identical results
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P31_sigma_dtype_and_broadcast_tolerance(device, dtype, N):
    M = 5
    W = 2
    z, z_j, vec_d, vec_d_j, sp_vec, sq_vec = make_problem(
        N=N, M=M, dtype=dtype, device=device
    )
    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    s_par_scalar = torch.tensor(0.75, dtype=real_dtype, device=device)
    s_perp_scalar = torch.tensor(0.40, dtype=real_dtype, device=device)

    sp_rep = torch.full(
        (M,), float(s_par_scalar.item()), dtype=real_dtype, device=device
    )
    sq_rep = torch.full(
        (M,), float(s_perp_scalar.item()), dtype=real_dtype, device=device
    )

    eta_scalar = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=s_par_scalar,
        sigma_perp=s_perp_scalar,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    eta_vector = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp_rep,
        sigma_perp=sq_rep,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_scalar, eta_vector, rtol=rtol, atol=atol
    ), "Providing sigma as 0-D tensors or matching per-contributor vectors must yield identical results"


# =========================
# P32 — Device consistency: CPU/GPU parity and offsets device agnosticism
# =========================
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P32_device_consistency_cpu_vs_cuda(dtype, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 3
    W = 1

    cpu = torch.device("cpu")
    gpu = torch.device("cuda")

    z_cpu, z_j_cpu, vec_d_cpu, vec_d_j_cpu, sp_cpu, sq_cpu = make_problem(
        N=N, M=M, dtype=dtype, device=cpu
    )
    per = CPSFPeriodization()
    offsets_cpu = per.window(N=N, W=W, device=cpu, sorted=False)

    eta_cpu = psi_over_offsets(
        z=z_cpu,
        z_j=z_j_cpu,
        vec_d=vec_d_cpu,
        vec_d_j=vec_d_j_cpu,
        sigma_par=sp_cpu,
        sigma_perp=sq_cpu,
        offsets=offsets_cpu,
        R_j=None,
        q_max=None,
    )

    z_gpu = z_cpu.to(gpu)
    z_j_gpu = z_j_cpu.to(gpu)
    vec_d_gpu = vec_d_cpu.to(gpu)
    vec_d_j_gpu = vec_d_j_cpu.to(gpu)
    sp_gpu = sp_cpu.to(gpu)
    sq_gpu = sq_cpu.to(gpu)

    eta_gpu = psi_over_offsets(
        z=z_gpu,
        z_j=z_j_gpu,
        vec_d=vec_d_gpu,
        vec_d_j=vec_d_j_gpu,
        sigma_par=sp_gpu,
        sigma_perp=sq_gpu,
        offsets=offsets_cpu,
        R_j=None,
        q_max=None,
    )

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_cpu.to(gpu), eta_gpu, rtol=rtol, atol=atol
    ), "CPU/GPU parity must hold within tolerances; offsets may reside on either device without changing results"


# =========================
# P33 — Autograd gradients exist (unclamped): finite, nonzero (aggregate) grads for all non-integer inputs
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P33_autograd_gradients_exist_unclamped(device, dtype, N):
    M = 4
    W = 1

    z, z_j, vec_d, vec_d_j, sp_vec, sq_vec = make_problem(
        N=N, M=M, dtype=dtype, device=device
    )
    z = z.detach().clone().requires_grad_(True)
    z_j = z_j.detach().clone().requires_grad_(True)
    vec_d = vec_d.detach().clone().requires_grad_(True)
    vec_d_j = vec_d_j.detach().clone().requires_grad_(True)
    sp = sp_vec.detach().clone().requires_grad_(True)
    sq = sq_vec.detach().clone().requires_grad_(True)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    out = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    loss = out.sum()
    loss.backward()

    for name, t in [
        ("z", z),
        ("z_j", z_j),
        ("vec_d", vec_d),
        ("vec_d_j", vec_d_j),
        ("sigma_par", sp),
        ("sigma_perp", sq),
    ]:
        assert t.grad is not None, f"Gradient for {name} must exist"
        assert torch.isfinite(t.grad).all(), f"Gradient for {name} must be finite"

    total_norm = 0.0
    for t in (z, z_j, vec_d, vec_d_j, sp, sq):
        total_norm += float(torch.sum(torch.abs(t.grad)).item())
    assert (
        total_norm > 0.0
    ), "Aggregate gradient norm must be positive for the unclamped case"


# =========================
# P34 — Autograd with q_max away from boundary: grads finite; if some terms are unclamped, aggregate grad > 0
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P34_autograd_with_qmax_away_from_boundary(device, dtype, N):
    M = 4
    W = 2

    z0, z_j0, vec_d0, vec_d_j0, sp0, sq0 = make_problem(
        N=N, M=M, dtype=dtype, device=device
    )
    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    dz0 = lift(z0) - lift(z_j0)
    dd0 = delta_vec_d(vec_d0, vec_d_j0)
    Rm0 = R(vec_d_j0)
    Rex0 = R_ext(Rm0)

    q_vals = []
    for k in range(offsets.shape[0]):
        row = offsets[k]
        n_r = row[:N].to(dtype=dz0.real.dtype)
        n_i = row[N:].to(dtype=dz0.real.dtype)
        n_c = torch.complex(n_r, n_i).to(dtype=dtype)
        dz_k = dz0 + n_c.unsqueeze(0)
        w = iota(dz_k, dd0)
        qk = q(w=w, R_ext=Rex0, sigma_par=sp0, sigma_perp=sq0)
        q_vals.append(qk)
    Q = torch.stack(q_vals, dim=0).reshape(-1)

    sQ, _ = torch.sort(Q)
    if sQ.numel() < 4:
        pytest.skip("Not enough q samples to choose a safe cap")
    q_low = float(sQ[int(0.25 * (sQ.numel() - 1))].item())
    q_high = float(sQ[int(0.75 * (sQ.numel() - 1))].item())
    q_cap = 0.5 * (q_low + q_high)
    q_cap -= 1e-6

    has_unclamped = bool((Q <= q_cap).any().item())

    z = z0.detach().clone().requires_grad_(True)
    z_j = z_j0.detach().clone().requires_grad_(True)
    vec_d = vec_d0.detach().clone().requires_grad_(True)
    vec_d_j = vec_d_j0.detach().clone().requires_grad_(True)
    sp = sp0.detach().clone().requires_grad_(True)
    sq = sq0.detach().clone().requires_grad_(True)

    out = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=q_cap,
    )
    loss = out.sum()
    loss.backward()

    for name, t in [
        ("z", z),
        ("z_j", z_j),
        ("vec_d", vec_d),
        ("vec_d_j", vec_d_j),
        ("sigma_par", sp),
        ("sigma_perp", sq),
    ]:
        assert t.grad is not None, f"Gradient for {name} must exist with q_max set"
        assert torch.isfinite(
            t.grad
        ).all(), f"Gradient for {name} must be finite with q_max set"

    total_norm = 0.0
    for t in (z, z_j, vec_d, vec_d_j, sp, sq):
        total_norm += float(torch.sum(torch.abs(t.grad)).item())

    if has_unclamped:
        assert total_norm > 0.0, (
            "With q_max chosen away from boundary and at least some unclamped terms, "
            "aggregate gradient norm should be positive"
        )
    else:
        assert total_norm >= 0.0


# =========================
# P35 — Offsets carry no gradients: integer dtype, no .grad; setting requires_grad_ must fail
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
def test_P35_offsets_have_no_grad_and_cannot_require_grad(device, N):
    per = CPSFPeriodization()
    offsets = per.window(N=N, W=1, device=device, sorted=False)
    assert offsets.dtype in (torch.int64, torch.long), "Offsets must be integer dtype"
    assert getattr(offsets, "grad", None) is None, "Offsets must not carry gradients"

    # Explicitly verify that requiring grad on integer offsets is disallowed
    with pytest.raises(RuntimeError):
        offsets.requires_grad_()


# =========================
# P36 — Empty offsets (O=0): returns zeros of shape [..., M]
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P36_empty_offsets_returns_zeros(device, dtype, N):
    M = 4
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    offsets_empty = torch.empty((0, 2 * N), dtype=torch.long, device=device)

    out = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_empty,
        R_j=None,
        q_max=None,
    )

    expected_real = torch.float32 if dtype == torch.complex64 else torch.float64
    assert out.shape == (M,)
    assert out.dtype == expected_real
    assert out.device.type == device.type
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out) == 0, "Sum over empty set must be identically zero"


# =========================
# P37 — Single offset (O=1): equals the single envelope term for that offset (generic nonzero offset)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P37_single_offset_equals_single_term(device, dtype, N):
    M = 3
    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)

    per = CPSFPeriodization()
    offsets_single = per.window(N=N, W=0, device=device, sorted=False)
    offsets_single[0, 0] = 1

    eta_vec = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_single,
        R_j=None,
        q_max=None,
    )

    dz0 = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rm = R(vec_d_j)
    Rex = R_ext(Rm)

    row = offsets_single[0]
    n_r = row[:N].to(dtype=dz0.real.dtype)
    n_i = row[N:].to(dtype=dz0.real.dtype)
    n_c = torch.complex(n_r, n_i).to(dtype=dtype)
    dz_k = dz0 + n_c.unsqueeze(0)

    w = iota(dz_k, dd)
    qv = q(w=w, R_ext=Rex, sigma_par=sp, sigma_perp=sq)
    eta_manual = rho(qv)

    rtol, atol = _get_tols(dtype)
    assert torch.allclose(
        eta_vec, eta_manual, rtol=rtol, atol=atol
    ), "Single-offset result must equal the single envelope term"


# =========================
# P38 — Minimal dimension (N=2): works and returns finite outputs
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
def test_P38_minimal_dimension_N2(device, dtype):
    N = 2
    M = 5
    W = 2

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)
    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    out = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    expected_real = torch.float32 if dtype == torch.complex64 else torch.float64
    assert out.shape == (M,)
    assert out.dtype == expected_real
    assert out.device.type == device.type
    assert torch.isfinite(out).all(), "Outputs must be finite for minimal allowed N"


# =========================
# P39 — Degenerate directions (axis-aligned / nearly singular components): must remain finite
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
def test_P39_degenerate_directions_axis_aligned(device, dtype):
    N = 2
    M = 6
    W = 1

    z = rand_unit_vector(N, dtype=dtype, device=device)
    z_j = torch.stack(
        [rand_unit_vector(N, dtype=dtype, device=device) for _ in range(M)], dim=0
    )

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    e0r = torch.tensor([1.0, 0.0], dtype=real_dtype, device=device)
    e1r = torch.tensor([0.0, 1.0], dtype=real_dtype, device=device)
    tiny = torch.tensor(torch.finfo(real_dtype).eps, dtype=real_dtype, device=device)

    rows = []
    for k in range(M):
        if k % 3 == 0:
            v = e0r.clone()
        elif k % 3 == 1:
            v = e1r.clone()
        else:
            v = torch.stack([tiny, torch.tensor(1.0, dtype=real_dtype, device=device)])
        vc = torch.complex(v, torch.zeros_like(v))
        vc = vc / torch.clamp(torch.linalg.vector_norm(vc), min=1e-20)
        rows.append(vc)
    vec_d_j = torch.stack(rows, dim=0).to(dtype=dtype)

    vec_d_query = torch.complex(
        torch.tensor([1.0, 0.0], dtype=real_dtype, device=device),
        torch.tensor([0.0, 0.0], dtype=real_dtype, device=device),
    ).to(dtype=dtype)
    vec_d = vec_d_query.unsqueeze(0).expand(M, N).clone()

    sp = torch.full((M,), 0.8, dtype=real_dtype, device=device)
    sq = torch.full((M,), 0.5, dtype=real_dtype, device=device)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    out = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    expected_real = torch.float32 if dtype == torch.complex64 else torch.float64
    assert out.shape == (M,)
    assert out.dtype == expected_real
    assert out.device.type == device.type
    assert torch.isfinite(
        out
    ).all(), "Axis-aligned / nearly-degenerate contributor directions must not produce NaN/Inf"


# =========================
# P40 — Determinism: identical inputs (including offsets) produce bit-wise identical outputs
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_P40_determinism_bitwise_stability(device, dtype, N):
    M = 4
    W = 2

    z, z_j, vec_d, vec_d_j, sp, sq = make_problem(N=N, M=M, dtype=dtype, device=device)
    per = CPSFPeriodization()
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    out1 = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )
    out2 = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    assert torch.equal(
        out1, out2
    ), "Determinism: repeated evaluation must be bit-wise identical"


# =========================
# P41 — Tolerance alignment: suite tolerances match the declared values
# =========================
def test_P41_tolerance_alignment_config():
    rtol64, atol64 = _get_tols(torch.complex64)
    rtol128, atol128 = _get_tols(torch.complex128)

    assert rtol64 == pytest.approx(5e-5) and atol64 == pytest.approx(
        5e-6
    ), "complex64 tolerances must be rtol≈5e-5, atol≈5e-6"
    assert rtol128 == pytest.approx(1e-12) and atol128 == pytest.approx(
        1e-12
    ), "complex128 tolerances must be rtol≈1e-12, atol≈1e-12"


# =========================
# Direct run help.
# =========================
if __name__ == "__main__":
    print("\nUse pytest to run:")
    print("\tpytest -q ./test_CPSF_psi_over_offsets.py\n")
