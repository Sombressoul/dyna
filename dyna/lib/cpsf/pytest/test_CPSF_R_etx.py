# Run as (example):
# > pytest -q .\dyna\lib\cpsf\pytest\test_CPSF_R_etx.py

import torch
import math
import pytest
from typing import Callable, List, Tuple

from dyna.lib.cpsf.functional.core_math import R_ext, R

TARGET_DEVICE = torch.device("cpu")
FN_IMPLS: List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
    ("R_ext", lambda d: R_ext(d)),
]
DTYPES = [torch.complex64, torch.complex128]
NS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SEED = 1337
_GEN = {}

# =========================
# Test constants
# =========================
# Tolerances per dtype
_TOLS = {
    torch.complex64: dict(rtol=5e-5, atol=5e-6),
    torch.complex128: dict(rtol=1e-12, atol=1e-12),
}


def _get_tols(dtype: torch.dtype):
    if dtype not in _TOLS:
        raise ValueError(f"No tolerances for dtype={dtype}")
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
    gen = _gen_for(device)

    if dtype.is_complex:
        x = torch.randn(shape_last, generator=gen, device=device, dtype=torch.float64)
        y = torch.randn(shape_last, generator=gen, device=device, dtype=torch.float64)
        v = (x + 1j * y).to(dtype)
    else:
        v = torch.randn(
            shape_last, generator=gen, device=device, dtype=torch.float64
        ).to(dtype)

    n = torch.linalg.vector_norm(v)
    if n.item() < torch.finfo(v.real.dtype).eps:
        v = torch.zeros_like(v)
        v[0] = 1
        return v.to(dtype)

    return (v / n).to(dtype)


def call_FN(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    return fn(x)


# =========================
# Tests
# =========================
# ============================> TEST: 01
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_01_block_diag_equivalence(impl_name, fn, dtype, N):
    d = rand_unit_vector(N, dtype, TARGET_DEVICE)

    R_base = R(d)
    got = call_FN(fn, R_base)
    expected = torch.block_diag(R_base, R_base)

    assert got.shape == (2 * N, 2 * N), f"{impl_name}: shape mismatch for N={N}"
    assert got.dtype == dtype, f"{impl_name}: dtype mismatch"
    assert got.device.type == TARGET_DEVICE.type, f"{impl_name}: device mismatch"

    rtol, atol = _get_tols(dtype)

    assert torch.allclose(
        got, expected, rtol=rtol, atol=atol
    ), f"{impl_name}: R_ext != block_diag(R,R) for N={N}, dtype={dtype}"

    off_tr = got[:N, N:]
    off_bl = got[N:, :N]
    z_tr = torch.zeros_like(off_tr)
    z_bl = torch.zeros_like(off_bl)

    assert torch.allclose(
        off_tr, z_tr, rtol=0.0, atol=10 * atol
    ), f"{impl_name}: top-right block not zero for N={N}, dtype={dtype}"
    assert torch.allclose(
        off_bl, z_bl, rtol=0.0, atol=10 * atol
    ), f"{impl_name}: bottom-left block not zero for N={N}, dtype={dtype}"


# ============================> TEST: 02
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_02_unitarity(impl_name, fn, dtype, N):
    d = rand_unit_vector(N, dtype, TARGET_DEVICE)
    R_base = R(d)
    got = call_FN(fn, R_base)
    rtol, atol = _get_tols(dtype)
    I2 = torch.eye(2 * N, dtype=dtype, device=TARGET_DEVICE)
    gram_l = got.conj().transpose(-2, -1) @ got
    gram_r = got @ got.conj().transpose(-2, -1)

    assert torch.allclose(
        gram_l, I2, rtol=rtol, atol=10 * atol
    ), f"{impl_name}: left Gram not identity for N={N}, dtype={dtype}"
    assert torch.allclose(
        gram_r, I2, rtol=rtol, atol=10 * atol
    ), f"{impl_name}: right Gram not identity for N={N}, dtype={dtype}"


# ============================> TEST: 03
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_03_first_column_alignment(impl_name, fn, dtype, N):
    d = rand_unit_vector(N, dtype, TARGET_DEVICE)
    R_base = R(d)
    got = call_FN(fn, R_base)
    rtol, atol = _get_tols(dtype)
    b = (d / torch.linalg.vector_norm(d)).to(dtype)

    col_R = R_base[:, 0]
    assert torch.allclose(
        col_R, b, rtol=rtol, atol=atol
    ), f"{impl_name}: R(d) first column != normalized d for N={N}, dtype={dtype}"

    col_tl = got[:N, :N][:, 0]
    col_br = got[N:, N:][:, 0]

    assert torch.allclose(
        col_tl, b, rtol=rtol, atol=atol
    ), f"{impl_name}: TL block first column misaligned for N={N}, dtype={dtype}"
    assert torch.allclose(
        col_br, b, rtol=rtol, atol=atol
    ), f"{impl_name}: BR block first column misaligned for N={N}, dtype={dtype}"
    assert torch.allclose(
        col_tl, col_br, rtol=rtol, atol=atol
    ), f"{impl_name}: TL/BR first columns differ for N={N}, dtype={dtype}"

    one = torch.ones((), dtype=col_tl.real.dtype, device=TARGET_DEVICE)
    for v in (b, col_R, col_tl, col_br):
        nv = torch.linalg.vector_norm(v)
        assert torch.allclose(
            nv, one, rtol=rtol, atol=10 * atol
        ), f"{impl_name}: non-unit first column norm in block for N={N}, dtype={dtype}"


# ============================> TEST: 04
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_04_block_column_orthonormality(impl_name, fn, dtype, N):
    d = rand_unit_vector(N, dtype, TARGET_DEVICE)
    R_base = R(d)
    got = call_FN(fn, R_base)
    rtol, atol = _get_tols(dtype)
    I = torch.eye(N, dtype=dtype, device=TARGET_DEVICE)
    TL = got[:N, :N]
    BR = got[N:, N:]
    gram_TL = TL.conj().transpose(-2, -1) @ TL
    gram_BR = BR.conj().transpose(-2, -1) @ BR

    assert torch.allclose(
        gram_TL, I, rtol=rtol, atol=10 * atol
    ), f"{impl_name}: TL block columns not orthonormal for N={N}, dtype={dtype}"
    assert torch.allclose(
        gram_BR, I, rtol=rtol, atol=10 * atol
    ), f"{impl_name}: BR block columns not orthonormal for N={N}, dtype={dtype}"
    assert torch.allclose(
        TL, R_base, rtol=rtol, atol=atol
    ), f"{impl_name}: TL block != R(d) for N={N}, dtype={dtype}"
    assert torch.allclose(
        BR, R_base, rtol=rtol, atol=atol
    ), f"{impl_name}: BR block != R(d) for N={N}, dtype={dtype}"


# ============================> TEST: 05
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_05_smoothness_finite_difference(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    eps0 = 3e-4 if dtype == torch.complex64 else 1e-6
    eps1 = eps0
    eps2 = eps0 * 0.5

    d = rand_unit_vector(N, dtype, device)
    z = rand_unit_vector(N, dtype, device)
    inner = torch.sum(torch.conj(d) * z)
    t = z - inner * d
    nt = torch.linalg.vector_norm(t)
    if nt.real.item() < torch.finfo(t.real.dtype).eps:
        t = torch.zeros_like(d)
        t[(0 if N == 1 else 1)] = 1.0
        nt = torch.linalg.vector_norm(t)
    t = (t / nt).to(dtype)

    def Rext_of_dir(dir_vec: torch.Tensor) -> torch.Tensor:
        Rb = R(dir_vec)
        return call_FN(fn, Rb)

    R0 = Rext_of_dir(d)

    def step(eps: float) -> float:
        d_eps = d + eps * t
        d_eps = d_eps / torch.linalg.vector_norm(d_eps)
        Reps = Rext_of_dir(d_eps)
        diff = Reps - R0
        return torch.linalg.vector_norm(diff.reshape(-1)).real.item()

    diff1 = step(eps1)
    diff2 = step(eps2)

    assert diff1 / eps1 < 1.0 / (
        atol + 1e-30
    ), f"{impl_name}: too large finite-diff slope at eps={eps1}, N={N}, dtype={dtype}"
    assert diff2 / eps2 < 1.0 / (
        atol + 1e-30
    ), f"{impl_name}: too large finite-diff slope at eps={eps2}, N={N}, dtype={dtype}"
    assert (
        diff2 <= 0.8 * diff1 + 10 * atol
    ), f"{impl_name}: finite-diff not contracting with eps halving (N={N}, dtype={dtype})"


# ============================> TEST: 06
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_06_right_action_U_Nm1_invariance_of_Sigma(impl_name, fn, dtype, N):
    raise NotImplementedError("Sigma is not implemented yet.")
    if N < 2:
        pytest.skip("U(N-1) is trivial for N=1; invariance holds vacuously.")

    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    sp = torch.rand((), device=device, dtype=REAL) + 0.5
    sq = torch.rand((), device=device, dtype=REAL) + 0.5
    D = torch.diag(torch.cat([sp.reshape(1), sq.repeat(N - 1)]))
    D = D.to(dtype)
    D_ext = torch.block_diag(D, D)

    def Sigma_from(R_in: torch.Tensor) -> torch.Tensor:
        Rex = call_FN(fn, R_in)
        return Rex.conj().transpose(-2, -1) @ D_ext @ Rex

    Sigma_base = Sigma_from(R_base)

    def random_unitary(m: int) -> torch.Tensor:
        Ar = torch.randn(m, m, device=device, dtype=REAL)
        Ai = torch.randn(m, m, device=device, dtype=REAL)
        A = (Ar + 1j * Ai).to(dtype)
        Q, _ = torch.linalg.qr(A)
        return Q

    for _ in range(3):
        Q = random_unitary(N - 1)
        U = torch.eye(N, device=device, dtype=dtype)
        U[1:, 1:] = Q
        R_tilt = R_base @ U
        Sigma_tilt = Sigma_from(R_tilt)
        assert torch.allclose(
            Sigma_tilt, Sigma_base, rtol=rtol, atol=10 * atol
        ), f"{impl_name}: Σ not invariant under right action diag(1,Q) for N={N}, dtype={dtype}"


# ============================> TEST: 07
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)])
def test_07_batch_mode_block_diagonal_structure(impl_name, fn, dtype, N, batch_shape):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)

    def rand_unit_batch(shape, n, dtype, device):
        xr = torch.randn(*shape, n, device=device, dtype=torch.float64)
        xi = torch.randn(*shape, n, device=device, dtype=torch.float64)
        v = (xr + 1j * xi).to(dtype)
        nrm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        v = v / nrm
        return v

    d_b = rand_unit_batch(batch_shape, N, dtype, device)
    flat = d_b.reshape(-1, N)
    R_list = [R(flat[i]) for i in range(flat.shape[0])]
    R_b = torch.stack(R_list, dim=0).reshape(*batch_shape, N, N)
    got = call_FN(fn, R_b)
    expected = torch.zeros(*batch_shape, 2 * N, 2 * N, device=device, dtype=dtype)
    expected[..., :N, :N] = R_b
    expected[..., N:, N:] = R_b

    assert got.shape == (
        *batch_shape,
        2 * N,
        2 * N,
    ), f"{impl_name}: wrong shape in batch, N={N}, dtype={dtype}, batch={batch_shape}"
    assert got.dtype == dtype and got.device.type == device.type
    assert torch.allclose(
        got, expected, rtol=rtol, atol=atol
    ), f"{impl_name}: batched R_ext != block_diag(R,R) for N={N}, dtype={dtype}"

    off_tr = got[..., :N, N:]
    off_bl = got[..., N:, :N]
    z_tr = torch.zeros_like(off_tr)
    z_bl = torch.zeros_like(off_bl)
    assert torch.allclose(
        off_tr, z_tr, rtol=0.0, atol=10 * atol
    ), f"{impl_name}: batched top-right block not zero for N={N}, dtype={dtype}"
    assert torch.allclose(
        off_bl, z_bl, rtol=0.0, atol=10 * atol
    ), f"{impl_name}: batched bottom-left block not zero for N={N}, dtype={dtype}"


# ============================> TEST: 08
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", [1])  # edge case only
def test_08_edge_case_N1_structure(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)

    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    got = call_FN(fn, R_base)
    expected = torch.block_diag(R_base, R_base)

    assert R_base.shape == (1, 1)
    assert got.shape == (2, 2)
    assert got.dtype == dtype and got.device.type == device.type
    assert torch.allclose(got, expected, rtol=rtol, atol=atol)
    assert torch.allclose(
        got[0, 1], torch.tensor(0, dtype=dtype, device=device), rtol=0.0, atol=10 * atol
    )
    assert torch.allclose(
        got[1, 0], torch.tensor(0, dtype=dtype, device=device), rtol=0.0, atol=10 * atol
    )

    phase = R_base[0, 0]
    assert torch.allclose(got[0, 0], phase, rtol=rtol, atol=atol)
    assert torch.allclose(got[1, 1], phase, rtol=rtol, atol=atol)

    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    one = torch.ones((), dtype=REAL, device=device)
    assert torch.allclose(
        (phase * phase.conj()).real.to(REAL), one, rtol=rtol, atol=10 * atol
    )


# ============================> TEST: 09
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_09_dtype_propagation_and_svd_stability(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64

    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    got = call_FN(fn, R_base)

    assert got.dtype == dtype and got.device.type == device.type

    s = torch.linalg.svdvals(got)
    ones = torch.ones_like(s, dtype=REAL, device=device)
    max_dev = torch.max(torch.abs(s.to(REAL) - ones)).item()
    assert max_dev <= 10 * atol, (
        f"{impl_name}: singular values deviate from 1 by {max_dev} (> {10*atol}) "
        f"for N={N}, dtype={dtype}"
    )

    kappa = (s.max() / s.min()).to(REAL)
    assert torch.allclose(
        kappa, torch.tensor(1.0, dtype=REAL, device=device), rtol=rtol, atol=10 * atol
    ), f"{impl_name}: condition number not ~1 (got {kappa.item():.3e}) for N={N}, dtype={dtype}"


# ============================> TEST: 10
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_10_device_propagation_and_cpu_gpu_parity(impl_name, fn, dtype, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping CPU/GPU parity test.")

    rtol, atol = _get_tols(dtype)

    device_cuda = torch.device("cuda")
    d_gpu = rand_unit_vector(N, dtype, device_cuda)
    R_gpu = R(d_gpu)
    got_gpu = call_FN(fn, R_gpu)
    expected_gpu = torch.block_diag(R_gpu, R_gpu)

    assert got_gpu.device.type == "cuda"
    assert torch.allclose(got_gpu, expected_gpu, rtol=rtol, atol=atol)

    device_cpu = torch.device("cpu")
    d_cpu = d_gpu.to(device_cpu)
    d_cpu = d_cpu / torch.linalg.vector_norm(d_cpu)
    R_cpu = R(d_cpu)
    got_cpu = call_FN(fn, R_cpu)

    assert torch.allclose(
        got_gpu.to(device_cpu), got_cpu, rtol=rtol, atol=10 * atol
    ), f"{impl_name}: CPU/GPU parity failed for N={N}, dtype={dtype}"


# ============================> TEST: 11
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_11_composition_with_block_inputs(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)
    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    Rex = call_FN(fn, R_base)
    gen = _gen_for(device)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64

    def rand_vec(n, dtype, device):
        xr = torch.randn(n, generator=gen, device=device, dtype=REAL)
        xi = torch.randn(n, generator=gen, device=device, dtype=REAL)
        return (xr + 1j * xi).to(dtype)

    for _ in range(3):
        u = rand_vec(N, dtype, device)
        v = rand_vec(N, dtype, device)
        w = torch.cat([u, v], dim=0)

        expected = torch.cat([R_base @ u, R_base @ v], 0)
        got = Rex @ w

        assert torch.allclose(
            got, expected, rtol=rtol, atol=10 * atol
        ), f"{impl_name}: composition failed for N={N}, dtype={dtype}"


# ============================> TEST: 12
@pytest.mark.parametrize("impl_name,fn", FN_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
def test_12_deterministic_behavior_no_rng(impl_name, fn, dtype, N):
    device = TARGET_DEVICE
    rtol, atol = _get_tols(dtype)

    d = rand_unit_vector(N, dtype, device)
    R_base = R(d)
    out0 = call_FN(fn, R_base)

    torch.manual_seed(12345)
    _ = torch.randn(17)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(67890)
        _ = torch.randn(17, device=torch.device("cuda"))

    out1 = call_FN(fn, R_base)

    torch.manual_seed(54321)
    _ = torch.randn(11)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(9876)
        _ = torch.randn(11, device=torch.device("cuda"))

    out2 = call_FN(fn, R_base)

    assert torch.allclose(
        out0, out1, rtol=0.0, atol=atol
    ), f"{impl_name}: non-deterministic output between calls (out0 vs out1), N={N}, dtype={dtype}"
    assert torch.allclose(
        out0, out2, rtol=0.0, atol=atol
    ), f"{impl_name}: non-deterministic output between calls (out0 vs out2), N={N}, dtype={dtype}"


# =========================
# Direct run help.
# =========================
if __name__ == "__main__":
    print("\n")
    print("Use pytest to run:")
    print("\tpytest -q ./test.file.name.py")
    print("\n")
