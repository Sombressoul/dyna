# Run as (example):
# > pytest -q .\dyna\lib\cpsf\pytest\test_CPSF_Sigma.py

import torch
import pytest
from typing import Callable, List, Tuple

from dyna.lib.cpsf.functional.core_math import (
    R,
    R_ext,
    Sigma,
)

# =========================
# Global config
# =========================
TARGET_DEVICE = torch.device("cpu")

SIGMA_IMPLS: List[Tuple[str, Callable[..., torch.Tensor]]] = [
    ("Sigma", lambda Rext, sp, sq: Sigma(R_ext=Rext, sigma_par=sp, sigma_perp=sq)),
]

DTYPES = [torch.complex64, torch.complex128]
NS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SEED = 1337
_GEN = {}

# =========================
# Tolerances
# =========================
# baseline per dtype (как в тестах R/R_ext)
_TOLS = {
    torch.complex64: dict(rtol=5e-5, atol=5e-6),
    torch.complex128: dict(rtol=1e-12, atol=1e-12),
}

# Check boosters.
HERMIT_ATOL_FACTOR = 10
ZERO_ATOL_FACTOR = 10
EIG_ABS_ATOL_FACTOR = 20
COND_ATOL_FACTOR = 10


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
    shape_last: int,
    dtype: torch.dtype,
    device=TARGET_DEVICE,
) -> torch.Tensor:
    """Complex gaussian -> normalize to unit vector (device-aware RNG)."""
    gen = _gen_for(device)
    x = torch.randn(shape_last, generator=gen, device=device, dtype=torch.float64)
    y = torch.randn(shape_last, generator=gen, device=device, dtype=torch.float64)
    v = (x + 1j * y).to(dtype)
    n = torch.linalg.vector_norm(v)
    if float(n.real) < torch.finfo(v.real.dtype).eps:
        v = torch.zeros_like(v)
        v[0] = 1
        return v
    return v / n


def call_sigma(
    fn: Callable[..., torch.Tensor],
    Rext: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
) -> torch.Tensor:
    return fn(Rext, sigma_par, sigma_perp)


# =========================
# Tests
# =========================

# ============================> TEST: S01 — shape/dtype/device & arg validation
@pytest.mark.parametrize("impl_name,fn", SIGMA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_S01_shape_dtype_device_and_args(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64

    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    Rext = R_ext(R_base)

    sp = torch.tensor(1.5, dtype=REAL, device=device)
    sq = torch.tensor(0.7, dtype=REAL, device=device)

    got = call_sigma(fn, Rext, sp, sq)
    assert got.shape == (2 * N, 2 * N), "Σ must keep (..., 2N, 2N) shape"
    assert got.dtype == dtype, "Σ must keep complex dtype of R_ext"
    assert got.device == device, "Σ must be on the same device as R_ext"

    with pytest.raises(ValueError):
        _ = call_sigma(fn, Rext, torch.tensor(0.0, dtype=REAL, device=device), sq)
    with pytest.raises(ValueError):
        _ = call_sigma(fn, Rext, sp, torch.tensor(0.0, dtype=REAL, device=device))
    with pytest.raises(ValueError):
        _ = call_sigma(fn, Rext, torch.tensor(-1.0, dtype=REAL, device=device), sq)
    with pytest.raises(ValueError):
        _ = call_sigma(fn, Rext, sp, torch.tensor(-1.0, dtype=REAL, device=device))

    bad_rect = torch.zeros(2 * N, N, dtype=dtype, device=device)
    with pytest.raises(ValueError):
        _ = call_sigma(fn, bad_rect, sp, sq)

    if 2 * N - 1 > 0:
        bad_odd = torch.zeros(2 * N - 1, 2 * N - 1, dtype=dtype, device=device)
        with pytest.raises(ValueError):
            _ = call_sigma(fn, bad_odd, sp, sq)


# ============================> TEST: S02 — Hermitian + SPD (+ Cholesky)
@pytest.mark.parametrize("impl_name,fn", SIGMA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_S02_hermitian_and_spd(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    Rext = R_ext(R_base)

    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    sp = torch.tensor(1.3, dtype=REAL, device=device)
    sq = torch.tensor(0.6, dtype=REAL, device=device)
    Sig = call_sigma(fn, Rext, sp, sq)

    assert torch.allclose(
        Sig, Sig.mH, rtol=rtol, atol=HERMIT_ATOL_FACTOR * atol
    ), f"{impl_name}: Sigma is not Hermitian within tolerance (N={N}, dtype={dtype})"

    L = torch.linalg.cholesky(Sig)
    Sig_rec = L @ L.mH
    err = torch.linalg.vector_norm((Sig_rec - Sig).reshape(-1)).real.item()
    thr = 10 * atol * ((2 * N) ** 0.5)
    assert err <= thr, (
        f"{impl_name}: Cholesky reconstruction error {err:.3e} exceeds {thr:.3e} "
        f"(N={N}, dtype={dtype})"
    )


# ============================> TEST: S03 — Block equivalence: diag(S0, S0)
@pytest.mark.parametrize("impl_name,fn", SIGMA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_S03_block_equivalence(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    Rext = R_ext(R_base)
    sp = torch.tensor(1.2, dtype=REAL, device=device)
    sq = torch.tensor(0.5, dtype=REAL, device=device)
    Sig = call_sigma(fn, Rext, sp, sq)
    diag0 = torch.full((N,), sq, dtype=REAL, device=device)
    diag0[0] = sp
    D0 = torch.diag(diag0).to(dtype)
    S0 = R_base @ (D0 @ R_base.mH)

    tl = Sig[:N, :N]
    br = Sig[N:, N:]
    tr = Sig[:N, N:]
    bl = Sig[N:, :N]

    assert torch.allclose(
        tl, S0, rtol=rtol, atol=atol
    ), f"{impl_name}: TL block != S0 (N={N}, dtype={dtype})"
    assert torch.allclose(
        br, S0, rtol=rtol, atol=atol
    ), f"{impl_name}: BR block != S0 (N={N}, dtype={dtype})"

    z_tr = torch.zeros_like(tr)
    z_bl = torch.zeros_like(bl)
    assert torch.allclose(
        tr, z_tr, rtol=0.0, atol=ZERO_ATOL_FACTOR * atol
    ), f"{impl_name}: TR block not zero (N={N}, dtype={dtype})"
    assert torch.allclose(
        bl, z_bl, rtol=0.0, atol=ZERO_ATOL_FACTOR * atol
    ), f"{impl_name}: BL block not zero (N={N}, dtype={dtype})"


# ============================> TEST: S04 — Spectrum & condition number
@pytest.mark.parametrize("impl_name,fn", SIGMA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_S04_spectrum_and_condition(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    Rext = R_ext(R_base)
    sp = torch.tensor(1.25, dtype=REAL, device=device)
    sq = torch.tensor(0.55, dtype=REAL, device=device)
    Sig = call_sigma(fn, Rext, sp, sq)
    eig = torch.linalg.eigvalsh(Sig).to(REAL)
    tol_eig = EIG_ABS_ATOL_FACTOR * atol
    mult_sp = 2
    mult_sq = 2 * (N - 1)
    cnt_sp = int((torch.abs(eig - sp) <= tol_eig).sum().item())
    cnt_sq = int((torch.abs(eig - sq) <= tol_eig).sum().item())

    assert cnt_sp == mult_sp, (
        f"{impl_name}: expected {mult_sp} eigenvalues near sp={sp.item():.6g}, got {cnt_sp} "
        f"(N={N}, dtype={dtype})"
    )
    assert cnt_sq == mult_sq, (
        f"{impl_name}: expected {mult_sq} eigenvalues near sq={sq.item():.6g}, got {cnt_sq} "
        f"(N={N}, dtype={dtype})"
    )

    kappa_emp = eig.max() / eig.min()
    kappa_ref = torch.max(sp, sq) / torch.min(sp, sq)
    assert torch.allclose(
        kappa_emp, kappa_ref, rtol=rtol, atol=COND_ATOL_FACTOR * atol
    ), (
        f"{impl_name}: condition number mismatch (emp={kappa_emp.item():.6g}, "
        f"ref={kappa_ref.item():.6g}) for N={N}, dtype={dtype}"
    )

    t = torch.tensor(0.9, dtype=REAL, device=device)
    Sig_iso = call_sigma(fn, Rext, t, t)
    I = torch.eye(2 * N, dtype=dtype, device=device)
    t_c = t.to(dtype=dtype)
    diff = torch.linalg.vector_norm((Sig_iso - (t_c * I)).reshape(-1)).real.item()
    thr = 10 * atol * (2 * N)
    assert diff <= thr, (
        f"{impl_name}: isotropic case ||Σ - t*I||_F too large: {diff:.3e} > {thr:.3e} "
        f"(N={N}, dtype={dtype})"
    )


# ============================> TEST: S05 — Right action U(N-1) invariance
@pytest.mark.parametrize("impl_name,fn", SIGMA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_S05_right_action_invariance(impl_name, fn, dtype, N):
    if N < 2:
        pytest.skip("U(N-1) is trivial for N=1; invariance holds vacuously.")

    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    Rext_base = R_ext(R_base)
    sp = torch.tensor(1.4, dtype=REAL, device=device)
    sq = torch.tensor(0.8, dtype=REAL, device=device)
    Sig_base = call_sigma(fn, Rext_base, sp, sq)

    def random_unitary(m: int) -> torch.Tensor:
        gen = _gen_for(device)
        Ar = torch.randn(m, m, generator=gen, device=device, dtype=REAL)
        Ai = torch.randn(m, m, generator=gen, device=device, dtype=REAL)
        A = (Ar + 1j * Ai).to(dtype)
        Q, _ = torch.linalg.qr(A)
        return Q

    for _ in range(3):
        Q = random_unitary(N - 1)
        U = torch.eye(N, dtype=dtype, device=device)
        U[1:, 1:] = Q
        R_tilt = R_base @ U
        Rext_tilt = R_ext(R_tilt)
        Sig_tilt = call_sigma(fn, Rext_tilt, sp, sq)

        assert torch.allclose(
            Sig_tilt, Sig_base, rtol=rtol, atol=10 * atol
        ), f"{impl_name}: Σ not invariant under right action diag(1,Q) for N={N}, dtype={dtype}"


# ============================> TEST: S06 — First-column phase invariance
@pytest.mark.parametrize("impl_name,fn", SIGMA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_S06_phase_invariance(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64

    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    Rext_base = R_ext(R_base)

    sp = torch.tensor(1.1, dtype=REAL, device=device)
    sq = torch.tensor(0.7, dtype=REAL, device=device)

    Sig_base = call_sigma(fn, Rext_base, sp, sq)

    for phi_val in (0.0, 0.3, 1.0, -2.0):
        phi = torch.tensor(phi_val, dtype=REAL, device=device)
        phase = torch.exp(1j * phi.to(dtype=dtype))

        Uphi = torch.eye(N, dtype=dtype, device=device)
        Uphi[0, 0] = phase

        R_phi = R_base @ Uphi
        Rext_phi = R_ext(R_phi)
        Sig_phi = call_sigma(fn, Rext_phi, sp, sq)

        assert torch.allclose(Sig_phi, Sig_base, rtol=rtol, atol=10 * atol), (
            f"{impl_name}: Σ not invariant under first-column phase φ={phi_val} "
            f"(N={N}, dtype={dtype})"
        )


# ============================> TEST: S07 — Linearity and scaling in (sp, sq)
@pytest.mark.parametrize("impl_name,fn", SIGMA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_S07_linearity_and_scaling(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    Rext = R_ext(R_base)
    sp = torch.tensor(1.3, dtype=REAL, device=device)
    sq = torch.tensor(0.6, dtype=REAL, device=device)
    alpha = torch.tensor(2.25, dtype=REAL, device=device)
    Sigma_base = call_sigma(fn, Rext, sp, sq)
    Sigma_scaled = call_sigma(fn, Rext, alpha * sp, alpha * sq)
    alpha_c = alpha.to(dtype=dtype)

    assert torch.allclose(
        Sigma_scaled, alpha_c * Sigma_base, rtol=rtol, atol=10 * atol
    ), f"{impl_name}: scaling failed (N={N}, dtype={dtype})"

    sp1 = torch.tensor(0.9, dtype=REAL, device=device)
    sq1 = torch.tensor(0.4, dtype=REAL, device=device)
    sp2 = torch.tensor(0.7, dtype=REAL, device=device)
    sq2 = torch.tensor(0.5, dtype=REAL, device=device)
    Sigma_1 = call_sigma(fn, Rext, sp1, sq1)
    Sigma_2 = call_sigma(fn, Rext, sp2, sq2)
    Sigma_sum = call_sigma(fn, Rext, sp1 + sp2, sq1 + sq2)

    assert torch.allclose(
        Sigma_1 + Sigma_2, Sigma_sum, rtol=rtol, atol=10 * atol
    ), f"{impl_name}: additivity on D failed (N={N}, dtype={dtype})"


# ============================> TEST: S08 — Action on block vector [u; v]
@pytest.mark.parametrize("impl_name,fn", SIGMA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_S08_block_vector_action(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    Rext = R_ext(R_base)
    sp = torch.tensor(1.2, dtype=REAL, device=device)
    sq = torch.tensor(0.6, dtype=REAL, device=device)
    Sigma_mat = call_sigma(fn, Rext, sp, sq)
    gen = _gen_for(device)

    def rand_vec(n: int) -> torch.Tensor:
        xr = torch.randn(n, generator=gen, device=device, dtype=REAL)
        xi = torch.randn(n, generator=gen, device=device, dtype=REAL)
        return (xr + 1j * xi).to(dtype)

    u = rand_vec(N)
    v = rand_vec(N)
    w = torch.cat([u, v], dim=0)
    diag0 = torch.full((N,), sq, dtype=REAL, device=device)
    diag0[0] = sp
    D0 = torch.diag(diag0).to(dtype)
    S0 = R_base @ (D0 @ R_base.mH)

    expected_top = S0 @ u
    expected_bot = S0 @ v

    got = Sigma_mat @ w
    got_top = got[:N]
    got_bot = got[N:]

    assert torch.allclose(
        got_top, expected_top, rtol=rtol, atol=10 * atol
    ), f"{impl_name}: top block action mismatch (N={N}, dtype={dtype})"
    assert torch.allclose(
        got_bot, expected_bot, rtol=rtol, atol=10 * atol
    ), f"{impl_name}: bottom block action mismatch (N={N}, dtype={dtype})"


# ============================> TEST: S09 — Inverse formula correctness
@pytest.mark.parametrize("impl_name,fn", SIGMA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_S09_inverse_formula(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    Rext = R_ext(R_base)
    sp = torch.tensor(1.7, dtype=REAL, device=device)
    sq = torch.tensor(0.6, dtype=REAL, device=device)
    Sigma_mat = call_sigma(fn, Rext, sp, sq)
    twoN = 2 * N
    diag_full = torch.full((twoN,), (1.0 / sq).to(REAL), dtype=REAL, device=device)
    diag_full[0] = 1.0 / sp
    diag_full[N] = 1.0 / sp
    D_inv = torch.diag(diag_full).to(dtype)
    Rext_h = Rext.mH if torch.is_complex(Rext) else Rext.transpose(-2, -1)
    Sigma_inv = Rext @ (D_inv @ Rext_h)
    I2 = torch.eye(twoN, dtype=dtype, device=device)
    left = Sigma_mat @ Sigma_inv
    right = Sigma_inv @ Sigma_mat

    assert torch.allclose(
        left, I2, rtol=rtol, atol=10 * atol
    ), f"{impl_name}: Σ @ Σ^{-1} != I (N={N}, dtype={dtype})"
    assert torch.allclose(
        right, I2, rtol=rtol, atol=10 * atol
    ), f"{impl_name}: Σ^{-1} @ Σ != I (N={N}, dtype={dtype})"


# ============================> TEST: S10 — Smoothness wrt direction d (finite diff)
@pytest.mark.parametrize("impl_name,fn", SIGMA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_S10_smoothness_finite_difference(impl_name, fn, dtype, N):
    if N < 2:
        pytest.skip("Directional smoothness is non-informative for N=1.")

    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64

    eps0 = 3e-4 if dtype == torch.complex64 else 1e-6
    eps1 = eps0
    eps2 = eps0 * 0.5

    d = rand_unit_vector(N, dtype, device)
    gen = _gen_for(device)
    zr = torch.randn(N, generator=gen, device=device, dtype=REAL)
    zi = torch.randn(N, generator=gen, device=device, dtype=REAL)
    z = (zr + 1j * zi).to(dtype)
    inner = torch.sum(torch.conj(d) * z)
    t = z - inner * d
    nt = torch.linalg.vector_norm(t)

    if nt.real.item() < torch.finfo(t.real.dtype).eps:
        e = torch.zeros(N, dtype=dtype, device=device)
        e[0] = 1.0
        t = e - torch.sum(torch.conj(d) * e) * d
        nt = torch.linalg.vector_norm(t)
        if nt.real.item() < torch.finfo(t.real.dtype).eps and N > 1:
            e = torch.zeros(N, dtype=dtype, device=device)
            e[1] = 1.0
            t = e - torch.sum(torch.conj(d) * e) * d
            nt = torch.linalg.vector_norm(t)

    t = t / nt
    sp = torch.tensor(1.3, dtype=REAL, device=device)
    sq = torch.tensor(0.6, dtype=REAL, device=device)

    def build_sigma(dir_vec: torch.Tensor) -> torch.Tensor:
        Rb = R(dir_vec)
        Rex = R_ext(Rb)
        return call_sigma(fn, Rex, sp, sq)

    Σ0 = build_sigma(d)

    def step(eps: float) -> float:
        d_eps = d + eps * t
        d_eps = d_eps / torch.linalg.vector_norm(d_eps)
        Σ_eps = build_sigma(d_eps)
        diff = torch.linalg.vector_norm((Σ_eps - Σ0).reshape(-1)).real.item()
        return diff

    diff1 = step(eps1)
    diff2 = step(eps2)

    assert diff1 / eps1 < 1.0 / (
        atol + 1e-30
    ), f"{impl_name}: too large finite-diff slope at eps={eps1}, N={N}, dtype={dtype}"
    assert (
        diff2 <= 0.8 * diff1 + 10 * atol
    ), f"{impl_name}: finite-diff not contracting on epsilon halving (N={N}, dtype={dtype})"


# ============================> TEST: S11 — Batch semantics / broadcasting
@pytest.mark.parametrize("impl_name,fn", SIGMA_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)])
def test_S11_batch_semantics_and_broadcast(impl_name, fn, dtype, N, batch_shape):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64

    def rand_unit_batch(Bshape, n, dtype, device):
        gen = _gen_for(device)
        xr = torch.randn(*Bshape, n, generator=gen, device=device, dtype=REAL)
        xi = torch.randn(*Bshape, n, generator=gen, device=device, dtype=REAL)
        v = (xr + 1j * xi).to(dtype)
        nrm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        return v / nrm

    d_b = rand_unit_batch(batch_shape, N, dtype, device)
    flat = d_b.reshape(-1, N)
    R_list = [R(flat[i]) for i in range(flat.shape[0])]
    R_b = torch.stack(R_list, dim=0).reshape(*batch_shape, N, N)
    Rext_b = R_ext(R_b)
    sp = torch.tensor(1.25, dtype=REAL, device=device)
    sq = torch.tensor(0.55, dtype=REAL, device=device)
    Sigma_b = call_sigma(fn, Rext_b, sp, sq)
    diag0 = torch.full((*batch_shape, N), sq, dtype=REAL, device=device)
    diag0[..., 0] = sp
    D0 = torch.diag_embed(diag0).to(dtype)
    S0 = R_b @ (D0 @ R_b.mH)
    expected = torch.zeros(*batch_shape, 2 * N, 2 * N, dtype=dtype, device=device)
    expected[..., :N, :N] = S0
    expected[..., N:, N:] = S0

    assert Sigma_b.shape == (*batch_shape, 2 * N, 2 * N)
    assert Sigma_b.dtype == dtype and Sigma_b.device == device
    assert torch.allclose(
        Sigma_b, expected, rtol=rtol, atol=atol
    ), f"{impl_name}: batched Σ != diag(S0,S0) for scalar sp/sq (N={N}, dtype={dtype})"

    tr = Sigma_b[..., :N, N:]
    bl = Sigma_b[..., N:, :N]
    z_tr = torch.zeros_like(tr)
    z_bl = torch.zeros_like(bl)

    assert torch.allclose(
        tr, z_tr, rtol=0.0, atol=ZERO_ATOL_FACTOR * atol
    ), f"{impl_name}: TR block not zero in batch (N={N}, dtype={dtype})"
    assert torch.allclose(
        bl, z_bl, rtol=0.0, atol=ZERO_ATOL_FACTOR * atol
    ), f"{impl_name}: BL block not zero in batch (N={N}, dtype={dtype})"

    gen = _gen_for(device)
    sp_b = torch.rand(*batch_shape, generator=gen, device=device, dtype=REAL) + 0.8
    sq_b = torch.rand(*batch_shape, generator=gen, device=device, dtype=REAL) + 0.4
    Sigma_b2 = call_sigma(fn, Rext_b, sp_b, sq_b)
    diag0_b = torch.full((*batch_shape, N), 0.0, dtype=REAL, device=device)
    diag0_b[..., 0] = sp_b
    diag0_b[..., 1:] = sq_b.unsqueeze(-1).expand(*batch_shape, N - 1)
    D0_b = torch.diag_embed(diag0_b).to(dtype)
    S0_b = R_b @ (D0_b @ R_b.mH)
    expected_b = torch.zeros(*batch_shape, 2 * N, 2 * N, dtype=dtype, device=device)
    expected_b[..., :N, :N] = S0_b
    expected_b[..., N:, N:] = S0_b

    assert torch.allclose(
        Sigma_b2, expected_b, rtol=rtol, atol=atol
    ), f"{impl_name}: batched Σ != diag(S0,S0) for per-batch sp/sq (N={N}, dtype={dtype})"


# =========================
# Direct run help.
# =========================
if __name__ == "__main__":
    print("\n")
    print("Use pytest to run:")
    print("\tpytest -q ./test.file.name.py")
    print("\n")
