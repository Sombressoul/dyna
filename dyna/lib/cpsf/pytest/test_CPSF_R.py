import torch
import math
import pytest
from typing import Callable, List, Tuple

from dyna.lib.cpsf.functional.core_math import R

TARGET_DEVICE = torch.device("cpu")
R_IMPLS: List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
    ("R", lambda d: R(d)),
]
DTYPES = [torch.complex64, torch.complex128]
NS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SEED = 1337

# R1+R2, R3, R4, R6, R7
ATOL_128 = 1e-13
RTOL_128 = 1e-10
ATOL_64 = 1e-5
RTOL_64 = 1e-3

# R5
H_VALS_R5 = [2.0 ** (-k) for k in range(3, 9)]
R5_RATIO_MAX = 3.0

# R6
U_TRIALS_R6 = 3

# R7
U_TRIALS_R7 = 5

# R8
TAU_R8 = 0.2

# R9
RHO_MAX = 3.0
G_MIN, G_MAX = 0.6, 1.8
H_VALS = [2.0 ** (-k) for k in range(3, 9)]


# =========================
# Helpers
# =========================
g = torch.Generator(device=TARGET_DEVICE).manual_seed(SEED)


def _get_ATOL(
    x: torch.Tensor,
) -> float:
    if x.dtype in [torch.complex64, torch.float32]:
        return ATOL_64
    elif x.dtype in [torch.complex128, torch.float64]:
        return ATOL_128
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")


def _get_RTOL(
    x: torch.Tensor,
) -> float:
    if x.dtype in [torch.complex64, torch.float32]:
        return RTOL_64
    elif x.dtype in [torch.complex128, torch.float64]:
        return RTOL_128
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")


def rand_unit_vector(
    shape_last: int,
    dtype: torch.dtype,
    device=TARGET_DEVICE,
) -> torch.Tensor:
    if dtype.is_complex:
        x = torch.randn(shape_last, generator=g, device=device, dtype=torch.float64)
        y = torch.randn(shape_last, generator=g, device=device, dtype=torch.float64)
        v = (x + 1j * y).to(dtype)
    else:
        v = torch.randn(shape_last, generator=g, device=device, dtype=torch.float64).to(
            dtype
        )

    n = torch.linalg.vector_norm(v)
    if n.item() < torch.finfo(v.real.dtype).eps:
        v = torch.zeros_like(v)
        v[0] = 1
        return v.to(dtype)

    return (v / n).to(dtype)


def call_R(
    R_fn: Callable[[torch.Tensor], torch.Tensor],
    d: torch.Tensor,
) -> torch.Tensor:
    return R_fn(d)


def _haar_unitary(
    n: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if n == 0:
        return torch.empty(0, 0, dtype=dtype, device=device)

    if dtype.is_complex:
        Ar = torch.randn(n, n, device=device, dtype=torch.float64)
        Ai = torch.randn(n, n, device=device, dtype=torch.float64)
        A = (Ar + 1j * Ai).to(dtype)
    else:
        A = torch.randn(n, n, device=device, dtype=torch.float64).to(dtype)

    Q, R = torch.linalg.qr(A, mode="reduced")
    d = torch.diagonal(R)
    denom = d.abs() + torch.finfo(d.real.dtype).eps
    phase = d / denom
    D = torch.diag(phase.conj()).to(Q.dtype)
    return Q @ D


def _dot_min_for_dtype(
    dt: torch.dtype,
) -> float:
    if dt.is_complex:
        return 0.99 if dt == torch.complex64 else 0.995
    return 0.995 if dt in (torch.float64, torch.bfloat16) else 0.99


# =========================
# Tests R1–R9
# =========================
# ============================> TEST: R1+R2
@pytest.mark.parametrize(
    "name,R_fn", R_IMPLS, ids=[n for n, _ in R_IMPLS] or ["<FILL_R_IMPLS>"]
)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("N", NS)
def test_R1_R2_unitarity(name, R_fn, dtype, N):
    d = rand_unit_vector(N, dtype)
    R = call_R(R_fn, d)

    assert (
        R.ndim == 2 and R.shape[0] == N and R.shape[1] == N
    ), f"{name}: R has wrong shape {tuple(R.shape)}, expected ({N},{N})."

    Rt = R.mH if dtype.is_complex else R.transpose(-2, -1)
    I = torch.eye(N, dtype=R.dtype, device=R.device)

    left = Rt @ R
    right = R @ Rt

    assert torch.allclose(
        left, I, atol=_get_ATOL(R), rtol=_get_RTOL(R)
    ) and torch.allclose(
        right, I, atol=_get_ATOL(R), rtol=_get_RTOL(R)
    ), f"{name}: R is not unitary (R1–R2)."

    col_norms = torch.linalg.vector_norm(R, dim=0)
    row_norms = torch.linalg.vector_norm(R, dim=1)
    one_c = torch.ones_like(col_norms)
    one_r = torch.ones_like(row_norms)
    assert torch.allclose(
        col_norms, one_c, atol=_get_ATOL(R), rtol=_get_RTOL(R)
    ), f"{name}: some column norms deviate from 1 (R1)."
    assert torch.allclose(
        row_norms, one_r, atol=_get_ATOL(R), rtol=_get_RTOL(R)
    ), f"{name}: some row norms deviate from 1 (R2)."

    off_cols = left - I
    off_rows = right - I
    mask = ~torch.eye(N, dtype=torch.bool, device=R.device)
    off_max_cols = off_cols.abs()[mask].max().item() if N > 1 else 0.0
    off_max_rows = off_rows.abs()[mask].max().item() if N > 1 else 0.0
    assert off_max_cols <= (
        _get_ATOL(R) + _get_RTOL(R)
    ), f"{name}: column Gram off-diagonals too large (R1): {off_max_cols:.3e}"
    assert off_max_rows <= (
        _get_ATOL(R) + _get_RTOL(R)
    ), f"{name}: row Gram off-diagonals too large (R2): {off_max_rows:.3e}"


# ============================> TEST: R3
@pytest.mark.parametrize(
    "name,R_fn", R_IMPLS, ids=[n for n, _ in R_IMPLS] or ["<FILL_R_IMPLS>"]
)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("N", NS)
def test_R3_alignment(name, R_fn, dtype, N):
    d = rand_unit_vector(N, dtype)
    R = call_R(R_fn, d)

    dn = torch.linalg.vector_norm(d)
    finfo = torch.finfo(d.real.dtype if dtype.is_complex else d.dtype)
    tiny = torch.sqrt(
        torch.tensor(
            finfo.eps, dtype=(d.real if dtype.is_complex else d).dtype, device=d.device
        )
    )
    b = d / torch.clamp(dn, min=tiny)

    r1 = R[:, 0]
    assert torch.allclose(
        r1, b, atol=_get_ATOL(R), rtol=_get_RTOL(R)
    ), f"{name}: R e1 != d (R3)."

    n1 = torch.linalg.vector_norm(r1)
    one = torch.tensor(1.0, dtype=n1.dtype, device=n1.device)
    assert torch.allclose(
        n1, one, atol=_get_ATOL(R), rtol=_get_RTOL(R)
    ), f"{name}: ||R e1|| != 1 (R3)."

    if dtype.is_complex:
        inner = torch.sum(torch.conj(b) * r1)
        one_c = torch.ones((), dtype=R.dtype, device=R.device)
        assert torch.allclose(
            inner, one_c, atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ), f"{name}: <b, R e1> != 1 (R3)."
    else:
        inner = torch.sum(b * r1)
        one_r = torch.tensor(1.0, dtype=inner.dtype, device=inner.device)
        assert torch.allclose(
            inner, one_r, atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ), f"{name}: <b, R e1> != 1 (R3)."


# ============================> TEST: R4
@pytest.mark.parametrize(
    "name,R_fn", R_IMPLS, ids=[n for n, _ in R_IMPLS] or ["<FILL_R_IMPLS>"]
)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("N", NS)
def test_R4_orthogonal_complement(name, R_fn, dtype, N):
    d = rand_unit_vector(N, dtype)
    R = call_R(R_fn, d)

    b = R[:, 0]
    Q = R[:, 1:] if N > 1 else torch.empty(N, 0, dtype=R.dtype, device=R.device)

    if N == 1:
        assert Q.shape == (1, 0)
        return

    is_c = dtype.is_complex
    I = torch.eye(N, dtype=R.dtype, device=R.device)
    P_perp = I - b.unsqueeze(-1) @ (b.conj().unsqueeze(-2) if is_c else b.unsqueeze(-2))

    inner = (b.conj().unsqueeze(0) @ Q) if is_c else (b.unsqueeze(0) @ Q)
    Z = torch.zeros_like(inner)
    assert torch.allclose(
        inner, Z, atol=_get_ATOL(R), rtol=_get_RTOL(R)
    ), f"{name}: complement not ⟂ b (R4)."

    G = Q.mH @ Q if is_c else Q.transpose(-2, -1) @ Q
    I_nm1 = torch.eye(N - 1, dtype=G.dtype, device=G.device)
    assert torch.allclose(
        G, I_nm1, atol=_get_ATOL(R), rtol=_get_RTOL(R)
    ), f"{name}: columns 2..N not orthonormal (R4)."

    QQh = Q @ (Q.mH if is_c else Q.transpose(-2, -1))
    assert torch.allclose(
        QQh, P_perp, atol=_get_ATOL(R), rtol=_get_RTOL(R)
    ), f"{name}: QQ^H != P_perp (R4)."


# ============================> TEST: R5
@pytest.mark.parametrize(
    "name,R_fn", R_IMPLS, ids=[n for n, _ in R_IMPLS] or ["<FILL_R_IMPLS>"]
)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("N", NS)
def test_R5_smoothness_local(name, R_fn, dtype, N):
    d = rand_unit_vector(N, dtype)
    dn = torch.linalg.vector_norm(d)
    finfo = torch.finfo(d.real.dtype if dtype.is_complex else d.dtype)
    tiny = torch.sqrt(
        torch.tensor(
            finfo.eps, dtype=(d.real if dtype.is_complex else d).dtype, device=d.device
        )
    )
    b = d / torch.clamp(dn, min=tiny)
    I = torch.eye(N, dtype=dtype, device=d.device)
    P_perp = I - b.unsqueeze(-1) @ (
        b.conj().unsqueeze(-2) if dtype.is_complex else b.unsqueeze(-2)
    )
    z = rand_unit_vector(N, dtype, device=d.device)
    xi = (P_perp @ z.unsqueeze(-1)).squeeze(-1)
    nxi = torch.linalg.vector_norm(xi)

    if float(nxi) < 1e-12:
        e2 = torch.zeros(N, dtype=dtype, device=d.device)
        e2[(0 if N == 1 else 1)] = 1.0
        xi = (P_perp @ e2.unsqueeze(-1)).squeeze(-1)
        nxi = torch.linalg.vector_norm(xi)

    xi = xi / torch.clamp(nxi, min=tiny)
    R0 = call_R(R_fn, d)
    deltas = []
    colcos_at_min = None

    for idx, h in enumerate(H_VALS_R5):
        h_val = float(h)
        d_h = b + h_val * xi
        d_h = d_h / torch.linalg.vector_norm(d_h)
        Rh = call_R(R_fn, d_h)
        diff = torch.linalg.matrix_norm(Rh - R0).item()
        deltas.append(diff)

        if idx == len(H_VALS_R5) - 1:
            dots = []
            for k in range(N):
                a = R0[:, k]
                bcol = Rh[:, k]
                if dtype.is_complex:
                    num = torch.sum(torch.conj(a) * bcol).abs()
                else:
                    num = torch.sum(a * bcol).abs()
                den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(bcol)

                if float(den) < 1e-16:
                    dots.append(torch.tensor(0.0, dtype=torch.float64))
                else:
                    dots.append((num / den).to(torch.float64))

            colcos_at_min = torch.min(torch.stack(dots)).item()

    for i in range(1, len(deltas)):
        assert (
            deltas[i - 1] >= deltas[i] - 1e-12
        ), f"{name}: Δ(h) not non-increasing (R5): Δ[{i-1}]={deltas[i-1]:.3e}, Δ[{i}]={deltas[i]:.3e}"

    for i in range(1, len(deltas)):
        ratio = deltas[i - 1] / max(deltas[i], 1e-16)
        assert (
            ratio < R5_RATIO_MAX
        ), f"{name}: non-smooth ΔR scaling (R5): ratio={ratio:.2f} at steps {H_VALS_R5[i-1]}→{H_VALS_R5[i]}"

    DOT_MIN = _dot_min_for_dtype(dtype)
    assert colcos_at_min is not None
    assert (
        colcos_at_min > DOT_MIN
    ), f"{name}: column mismatch under tiny perturbation (R5): min cos={colcos_at_min:.5f}, need > {DOT_MIN}"


# ============================> TEST: R6
@pytest.mark.parametrize(
    "name,R_fn", R_IMPLS, ids=[n for n, _ in R_IMPLS] or ["<FILL_R_IMPLS>"]
)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("N", NS)
def test_R6_right_U_invariance_local(name, R_fn, dtype, N):
    d = rand_unit_vector(N, dtype)
    R = call_R(R_fn, d)

    dn = torch.linalg.vector_norm(d)
    finfo = torch.finfo(d.real.dtype if dtype.is_complex else d.dtype)
    tiny = torch.sqrt(
        torch.tensor(
            finfo.eps, dtype=(d.real if dtype.is_complex else d).dtype, device=d.device
        )
    )
    b = d / torch.clamp(dn, min=tiny)

    def _is_unitary(X: torch.Tensor) -> bool:
        Xt = X.mH if dtype.is_complex else X.transpose(-2, -1)
        I = torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)
        return torch.allclose(
            Xt @ X, I, atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ) and torch.allclose(X @ Xt, I, atol=_get_ATOL(R), rtol=_get_RTOL(R))

    def _projector_from_b(bb: torch.Tensor) -> torch.Tensor:
        I = torch.eye(bb.shape[0], dtype=bb.dtype, device=bb.device)
        return I - bb.unsqueeze(-1) @ (
            bb.conj().unsqueeze(-2) if dtype.is_complex else bb.unsqueeze(-2)
        )

    Q = R[:, 1:] if N > 1 else torch.empty(N, 0, dtype=R.dtype, device=R.device)
    P_perp = _projector_from_b(b)
    assert _is_unitary(R), f"{name}: base frame not unitary (R6 precondition)."
    if N > 1:
        ortho = (
            (b.conj().unsqueeze(0) @ Q) if dtype.is_complex else (b.unsqueeze(0) @ Q)
        )
        assert torch.allclose(
            ortho, torch.zeros_like(ortho), atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ), f"{name}: base complement not ⟂ b (R6 precondition)."
        G = Q.mH @ Q if dtype.is_complex else Q.transpose(-2, -1) @ Q
        I_nm1 = torch.eye(N - 1, dtype=G.dtype, device=G.device)
        assert torch.allclose(
            G, I_nm1, atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ), f"{name}: base complement not orthonormal (R6 precondition)."
        QQh = Q @ (Q.mH if dtype.is_complex else Q.transpose(-2, -1))
        assert torch.allclose(
            QQh, P_perp, atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ), f"{name}: base projector QQ^H != P_perp (R6 precondition)."

    U_list = []
    if N > 1:
        U_I = torch.eye(N - 1, dtype=R.dtype, device=R.device)
        U_list.append(U_I)

        tau = 1.0e-3
        if dtype.is_complex:
            X = (
                torch.randn(N - 1, N - 1, device=R.device, dtype=torch.float64)
                + 1j * torch.randn(N - 1, N - 1, device=R.device, dtype=torch.float64)
            ).to(R.dtype)
            K = X - X.mH
        else:
            X = torch.randn(N - 1, N - 1, device=R.device, dtype=torch.float64).to(
                R.dtype
            )
            K = X - X.transpose(-2, -1)
        U_local = torch.matrix_exp(tau * K)
        U_list.append(U_local)

        for _ in range(U_TRIALS_R6):
            U_list.append(_haar_unitary(N - 1, R.dtype, R.device))

    for U in U_list:
        if N == 1:
            R_ext = R
        else:
            D = torch.eye(N, dtype=R.dtype, device=R.device)
            D[1:, 1:] = U
            R_ext = R @ D

        assert _is_unitary(R_ext), f"{name}: R·diag(1,U) lost unitarity (R6)."

        b_ext = R_ext[:, 0]
        assert torch.allclose(
            b_ext, b, atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ), f"{name}: first column changed under right action (R6)."

        if N > 1:
            Q_ext = R_ext[:, 1:]
            inner = (
                (b.conj().unsqueeze(0) @ Q_ext)
                if dtype.is_complex
                else (b.unsqueeze(0) @ Q_ext)
            )
            assert torch.allclose(
                inner, torch.zeros_like(inner), atol=_get_ATOL(R), rtol=_get_RTOL(R)
            ), f"{name}: complement not ⟂ b after right action (R6)."
            G_ext = (
                Q_ext.mH @ Q_ext
                if dtype.is_complex
                else Q_ext.transpose(-2, -1) @ Q_ext
            )
            I_nm1 = torch.eye(N - 1, dtype=G_ext.dtype, device=G_ext.device)
            assert torch.allclose(
                G_ext, I_nm1, atol=_get_ATOL(R), rtol=_get_RTOL(R)
            ), f"{name}: complement columns not orthonormal after right action (R6)."
            QQh_ext = Q_ext @ (
                Q_ext.mH if dtype.is_complex else Q_ext.transpose(-2, -1)
            )
            assert torch.allclose(
                QQh_ext, P_perp, atol=_get_ATOL(R), rtol=_get_RTOL(R)
            ), f"{name}: projector changed under right action (R6)."


# ============================> TEST: R7
@pytest.mark.parametrize(
    "name,R_fn", R_IMPLS, ids=[n for n, _ in R_IMPLS] or ["<FILL_R_IMPLS>"]
)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("N", NS)
def test_R7_extended_frame_unitary(name, R_fn, dtype, N):
    d = rand_unit_vector(N, dtype)
    R = call_R(R_fn, d)

    if dtype.is_complex:
        dn = torch.linalg.vector_norm(d)
        b = d / torch.clamp(
            dn,
            min=torch.sqrt(
                torch.tensor(
                    torch.finfo(d.real.dtype).eps, dtype=d.real.dtype, device=d.device
                )
            ),
        )
    else:
        dn = torch.linalg.vector_norm(d)
        b = d / torch.clamp(
            dn,
            min=torch.sqrt(
                torch.tensor(torch.finfo(d.dtype).eps, dtype=d.dtype, device=d.device)
            ),
        )

    def _is_unitary(X: torch.Tensor) -> bool:
        Xt = X.mH if dtype.is_complex else X.transpose(-2, -1)
        I = torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)
        return torch.allclose(
            Xt @ X, I, atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ) and torch.allclose(X @ Xt, I, atol=_get_ATOL(R), rtol=_get_RTOL(R))

    if N == 1:
        assert _is_unitary(R), f"{name}: base frame not unitary at N=1 (R7)."
        assert torch.allclose(
            R[:, 0], b, atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ), f"{name}: first column changed at N=1 (R7)."
        return

    for _ in range(U_TRIALS_R7):
        U = _haar_unitary(N - 1, dtype, d.device)
        D = torch.eye(N, dtype=R.dtype, device=R.device)
        D[1:, 1:] = U
        R_ext = R @ D

        assert _is_unitary(R_ext), f"{name}: extended frame not unitary (R7)."
        assert torch.allclose(
            R_ext[:, 0], b, atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ), f"{name}: first column changed under extension (R7)."

        Qext = R_ext[:, 1:]
        inner = (
            torch.conj(b).unsqueeze(0) @ Qext
            if dtype.is_complex
            else b.unsqueeze(0) @ Qext
        )
        assert torch.allclose(
            inner, torch.zeros_like(inner), atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ), f"{name}: complement not orthogonal to b after extension (R7)."

        G = Qext.mH @ Qext if dtype.is_complex else Qext.transpose(-2, -1) @ Qext
        I_nm1 = torch.eye(N - 1, dtype=G.dtype, device=G.device)
        assert torch.allclose(
            G, I_nm1, atol=_get_ATOL(R), rtol=_get_RTOL(R)
        ), f"{name}: complement columns not orthonormal after extension (R7)."


# ============================> TEST: R8
@pytest.mark.parametrize(
    "name,R_fn", R_IMPLS, ids=[n for n, _ in R_IMPLS] or ["<FILL_R_IMPLS>"]
)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("N", NS)
def test_R8_local_trivialization_path_continuity(name, R_fn, dtype, N):
    d0 = rand_unit_vector(N, dtype)
    d1 = rand_unit_vector(N, dtype)

    if dtype.is_complex:
        inner = torch.sum(torch.conj(d0) * d1)
        phi = torch.angle(inner)
        d1 = d1 * torch.exp(-1j * phi)

    c = (
        torch.sum(torch.conj(d0) * d1).abs().item()
        if dtype.is_complex
        else torch.sum(d0 * d1).abs().item()
    )
    c = max(min(c, 1.0), -1.0)
    theta = math.acos(c)
    tau_target = TAU_R8 * 0.5

    if theta < 1e-12:
        K = 3
    else:
        denom = 2.0 * math.asin(max(min(tau_target / 2.0, 1.0), 0.0))
        K = 1 + math.ceil(theta / denom)
        K = max(K, 3)

    t_vals = torch.linspace(0.0, 1.0, K, device=d0.device)
    prev = call_R(R_fn, d0)

    def align_right(prev_R: torch.Tensor, cur_R: torch.Tensor) -> torch.Tensor:
        Qprev = prev_R[..., :, 1:]
        Qcur = cur_R[..., :, 1:]
        W = Qprev.mH @ Qcur if dtype.is_complex else Qprev.transpose(-2, -1) @ Qcur
        S = W.mH @ W if dtype.is_complex else W.transpose(-2, -1) @ W
        evals, U = torch.linalg.eigh(S)
        finfo = torch.finfo(prev_R.real.dtype)
        inv_sqrt = (evals.clamp_min(finfo.eps) ** -0.5).diag_embed().to(S.dtype)
        Uh = U.mH if dtype.is_complex else U.transpose(-2, -1)
        S_inv_half = U @ inv_sqrt @ Uh
        U_polar = W @ S_inv_half

        Nn = cur_R.shape[-1]
        D = torch.eye(Nn, dtype=cur_R.dtype, device=cur_R.device)
        D[1:, 1:] = U_polar

        return prev_R @ D

    for t in t_vals[1:]:
        if theta < 1e-12:
            dt = d0.clone()
        else:
            a = math.sin((1.0 - float(t)) * theta) / math.sin(theta)
            b = math.sin(float(t) * theta) / math.sin(theta)
            dt = a * d0 + b * d1
            dt = dt / torch.linalg.vector_norm(dt)

        Rt = call_R(R_fn, dt)

        Rprev_aligned = align_right(prev, Rt)
        diff = torch.linalg.matrix_norm(Rt - Rprev_aligned).item()

        assert (
            diff < TAU_R8
        ), f"{name}: discontinuity along path (R8), step diff={diff:.3e}"

        prev = Rt


# ============================> TEST: R9
@pytest.mark.parametrize(
    "name,R_fn", R_IMPLS, ids=[n for n, _ in R_IMPLS] or ["<FILL_R_IMPLS>"]
)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("N", NS)
def test_R9_bounded_derivatives_proxy(name, R_fn, dtype, N):
    def rand_unit_vector(
        shape_last: int, dt: torch.dtype, device="cpu"
    ) -> torch.Tensor:
        if dt.is_complex:
            x = torch.randn(shape_last, device=device, dtype=torch.float64)
            y = torch.randn(shape_last, device=device, dtype=torch.float64)
            v = (x + 1j * y).to(dt)
        else:
            v = torch.randn(shape_last, device=device, dtype=torch.float64).to(dt)
        n = torch.linalg.vector_norm(v)
        if float(n) < 1e-16:
            v = torch.zeros_like(v)
            v[0] = 1
            return v.to(dt)
        return (v / n).to(dt)

    def align_right(
        R_ref: torch.Tensor, R_cur: torch.Tensor, is_complex: bool
    ) -> torch.Tensor:
        Qr = R_ref[..., :, 1:]
        Qc = R_cur[..., :, 1:]
        W = Qr.mH @ Qc if is_complex else Qr.transpose(-2, -1) @ Qc
        S = W.mH @ W if is_complex else W.transpose(-2, -1) @ W
        evals, U = torch.linalg.eigh(S)
        finfo = torch.finfo(R_ref.real.dtype)
        inv_sqrt = (evals.clamp_min(finfo.eps) ** -0.5).diag_embed().to(S.dtype)
        Uh = U.mH if is_complex else U.transpose(-2, -1)
        S_inv_half = U @ inv_sqrt @ Uh
        U_polar = W @ S_inv_half

        n = R_cur.shape[-1]
        D = torch.eye(n, dtype=R_cur.dtype, device=R_cur.device)
        D[1:, 1:] = U_polar
        return R_ref @ D

    axes = []
    for k in range(N):
        ek = torch.zeros(N, dtype=dtype)
        ek[k] = 1.0
        axes.append(ek)
    for _ in range(3):
        axes.append(rand_unit_vector(N, dtype))

    is_c = dtype.is_complex

    for d in axes:
        dn = torch.linalg.vector_norm(d)
        finfo = torch.finfo(d.real.dtype if is_c else d.dtype)
        tiny = torch.sqrt(
            torch.tensor(
                finfo.eps, dtype=d.real.dtype if is_c else d.dtype, device=d.device
            )
        )
        b = d / torch.clamp(dn, min=tiny)

        I = torch.eye(N, dtype=dtype, device=d.device)
        P_perp = I - b.unsqueeze(-1) @ (
            b.conj().unsqueeze(-2) if is_c else b.unsqueeze(-2)
        )
        z = rand_unit_vector(N, dtype, device=d.device)
        xi = (P_perp @ z.unsqueeze(-1)).squeeze(-1)
        nx = torch.linalg.vector_norm(xi)
        if float(nx) < 1e-12:
            e2 = torch.zeros(N, dtype=dtype, device=d.device)
            e2[1 % N] = 1.0
            xi = (P_perp @ e2.unsqueeze(-1)).squeeze(-1)
            nx = torch.linalg.vector_norm(xi)
        xi = xi / torch.clamp(nx, min=tiny)

        R0 = call_R(R_fn, d)

        grads = []
        for h in H_VALS:
            h_val = float(h)
            d_h = b + h_val * xi
            d_h = d_h / torch.linalg.vector_norm(d_h)

            Rh = call_R(R_fn, d_h)

            R0_aligned = align_right(R0, Rh, is_c)
            diff = torch.linalg.matrix_norm(Rh - R0_aligned).item()
            grad = diff / h_val
            grads.append(grad)

        assert all(
            math.isfinite(x) for x in grads
        ), f"{name}: non-finite derivative proxy at direction (R9). grads={grads}"

        for i in range(1, len(grads)):
            ratio = grads[i - 1] / max(grads[i], 1e-16)
            assert (
                ratio < RHO_MAX
            ), f"{name}: derivative grows when h↓ (R9), ratio={ratio:.2f}, grads={grads}"

        g_minstep = grads[-1]
        assert (
            G_MIN <= g_minstep <= G_MAX
        ), f"{name}: gradient magnitude off expected range at smallest h (R9). g={g_minstep:.3f}, grads={grads}"


# =========================
# Direct run help.
# =========================
if __name__ == "__main__":
    print("\n")
    print("Use pytest to run:")
    print("\tpytest -q ./test.file.name.py")
    print("\n")
