# Run as (example):
# > pytest -q dyna/lib/cpsf/pytest/test_CPSF_T_classic.py

import torch
import pytest
from typing import Tuple, List

from dyna.lib.cpsf.functional.core_math import (
    T_classic_window,
    T_classic_full,
    psi_over_offsets,
)
from dyna.lib.cpsf.periodization import CPSFPeriodization


# =========================
# Global config
# =========================
TARGET_DEVICE = torch.device("cpu")

DTYPES_C = [torch.complex64, torch.complex128]
DTYPES_REAL = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}
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
T4_EQUIVALENCE_TOLS = (1.0e-7, 1.0e-7)


def _get_tols(dtype_c: torch.dtype):
    t = _TOLS[dtype_c]
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


def available_devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


def make_problem_T(
    *,
    B: int,
    N: int,
    M: int,
    S: int,
    dtype_z: torch.dtype,
    dtype_T: torch.dtype,
    device: torch.device = TARGET_DEVICE,
):
    g = _gen_for(device)
    z = torch.stack(
        [rand_unit_vector(N, dtype=dtype_z, device=device) for _ in range(B)], dim=0
    )
    z_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype_z, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )
    vec_d = torch.stack(
        [rand_unit_vector(N, dtype=dtype_z, device=device) for _ in range(B)], dim=0
    )
    vec_d_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype_z, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )

    REAL_T = DTYPES_REAL[dtype_T]
    Tr = torch.randn(B, M, S, generator=g, device=device, dtype=REAL_T)
    Ti = torch.randn(B, M, S, generator=g, device=device, dtype=REAL_T)
    T_hat_j = torch.complex(Tr, Ti).to(dtype_T)

    REAL_Z = DTYPES_REAL[dtype_z]
    alpha_j = torch.rand(B, M, generator=g, device=device, dtype=REAL_T) * 1.5 + 0.1
    sigma_par = torch.rand(B, M, generator=g, device=device, dtype=REAL_Z) * 0.9 + 0.1
    sigma_perp = torch.rand(B, M, generator=g, device=device, dtype=REAL_Z) * 0.9 + 0.1

    return z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp


def _chunk_offsets_even(offsets: torch.Tensor, num_chunks: int):
    O = int(offsets.shape[0])
    if num_chunks <= 1 or O <= 1:
        return [(0, 0, offsets)]
    num_chunks = min(num_chunks, O)  # no empty chunks
    base = O // num_chunks
    rem = O % num_chunks
    sizes = [base + (1 if i < rem else 0) for i in range(num_chunks)]
    cuts = [0]
    for s in sizes:
        cuts.append(cuts[-1] + s)
    packs = []
    for i in range(num_chunks):
        a, b = cuts[i], cuts[i + 1]
        if b > a:
            packs.append((a, b, offsets[a:b]))
    return packs or [(0, 0, offsets)]


# =========================
# Implementations under test (names for reporting)
# =========================
T_IMPLS: List[Tuple[str]] = [
    ("T_classic_window",),
    ("T_classic_full",),
]


# =========================
# TESTS
# =========================


# =========================
# T1 — Toroidal invariance under joint integer shifts
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T1_joint_integer_shift_invariance(device, N, dtype_z, dtype_T):
    B, M, S = 1, 4, 3

    rtol, atol = _get_tols(dtype_z)
    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )
    per = CPSFPeriodization()
    W = 2
    offsets = per.window(N=N, W=W, device=device, sorted=False)
    packs = list(
        per.iter_packed(
            N=N,
            target_points_per_pack=max(1, offsets.shape[0] // 3),
            start_radius=0,
            max_radius=W,
            device=device,
            sorted=False,
        )
    )
    assert len(packs) >= 1

    g = _gen_for(device)
    REAL = DTYPES_REAL[dtype_z]
    nR = torch.randint(-2, 3, (N,), generator=g, device=device)
    nI = torch.randint(-2, 3, (N,), generator=g, device=device)
    lam = torch.complex(nR.to(REAL), nI.to(REAL))

    z_s = z + lam
    zjs = z_j + lam.view(1, 1, N)

    T0_w = T_classic_window(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, offsets)
    T1_w = T_classic_window(z_s, zjs, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, offsets)
    assert T0_w.dtype == dtype_T and T1_w.dtype == dtype_T
    assert torch.allclose(T0_w, T1_w, rtol=rtol, atol=atol)

    T0_f = T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs)
    T1_f = T_classic_full(z_s, zjs, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs)
    assert T0_f.dtype == dtype_T and T1_f.dtype == dtype_T
    assert torch.allclose(T0_f, T1_f, rtol=rtol, atol=atol)


# =========================
# T2 — Offsets must encode Z^{2N} (imag offsets matter)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T2_offsets_require_imag_part(device, N, dtype_z, dtype_T):
    B, M, S = 1, 4, 3
    rtol, atol = _get_tols(dtype_T)

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )
    per = CPSFPeriodization()
    W = 2

    offsets = per.window(N=N, W=W, device=device, sorted=False)
    zeros = torch.zeros_like(offsets[:, :N])
    offsets_re_only = torch.cat([offsets[:, :N], zeros], dim=-1)

    # WINDOW: canonical vs re-only must not be allclose
    T_fullZ = T_classic_window(
        z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, offsets
    )
    T_reZ = T_classic_window(
        z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, offsets_re_only
    )
    assert T_fullZ.dtype == dtype_T and T_reZ.dtype == dtype_T
    assert not torch.allclose(
        T_fullZ, T_reZ, rtol=rtol, atol=atol
    ), "Using only Re-shifts should deviate from canonical Z^{2N} result"

    # FULL: transform each pack to re-only variant
    packs = list(
        per.iter_packed(
            N=N,
            target_points_per_pack=max(1, offsets.shape[0] // 3),
            start_radius=0,
            max_radius=W,
            device=device,
            sorted=False,
        )
    )
    packs_re = []
    for a, b, off in packs:
        packs_re.append(
            (a, b, torch.cat([off[:, :N], torch.zeros_like(off[:, :N])], dim=-1))
        )

    T_fullZ_F = T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs)
    T_reZ_F = T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs_re)
    assert T_fullZ_F.dtype == dtype_T and T_reZ_F.dtype == dtype_T
    assert not torch.allclose(
        T_fullZ_F, T_reZ_F, rtol=rtol, atol=atol
    ), "Using only Re-shifts in FULL must deviate from canonical Z^{2N}"


# =========================
# T3 — Realness of eta (psi_over_offsets output)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
def test_T3_eta_is_real_up_to_fp_noise(device, N, dtype_z):
    def _eps_for(dtype_c: torch.dtype) -> float:
        return 5e-5 if dtype_c == torch.complex64 else 1e-12

    B, M = 2, 5
    g = _gen_for(device)
    REAL = DTYPES_REAL[dtype_z]

    z = torch.stack(
        [rand_unit_vector(N, dtype=dtype_z, device=device) for _ in range(B)], dim=0
    )
    z_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype_z, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )
    vec_d = torch.stack(
        [rand_unit_vector(N, dtype=dtype_z, device=device) for _ in range(B)], dim=0
    )
    vec_d_j = torch.stack(
        [
            torch.stack(
                [rand_unit_vector(N, dtype=dtype_z, device=device) for _ in range(M)],
                dim=0,
            )
            for _ in range(B)
        ],
        dim=0,
    )
    sigma_par = torch.rand(B, M, generator=g, device=device, dtype=REAL) * 0.9 + 0.1
    sigma_perp = torch.rand(B, M, generator=g, device=device, dtype=REAL) * 0.9 + 0.1

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=2, device=device, sorted=False)

    # Match shapes expected by the kernel (no reliance on T_* expansions here)
    z_exp = z.unsqueeze(1).expand(B, M, N)  # [B,M,N]
    vecd_exp = vec_d.unsqueeze(1).expand(B, M, N)  # [B,M,N]

    eta = psi_over_offsets(
        z=z_exp,
        z_j=z_j,
        vec_d=vecd_exp,
        vec_d_j=vec_d_j,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
        offsets=offsets,
        R_j=None,
        q_max=None,
    )

    # Allow real dtype or complex with tiny imaginary residue
    if torch.is_complex(eta):
        max_im = eta.imag.abs().max().item()
        max_re = eta.real.abs().max().item()
    else:
        max_im = 0.0
        max_re = eta.abs().max().item()

    eps = _eps_for(dtype_z)
    assert max_im <= eps * (
        1.0 + max_re
    ), f"eta imaginary part too large: {max_im} vs {max_re}"


# =========================
# T4 — Window vs Full identity on the same offsets
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T4_window_full_identity_same_offsets(device, N, dtype_z, dtype_T):
    B, M, S = 1, 5, 4
    rtol, atol = T4_EQUIVALENCE_TOLS

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )

    per = CPSFPeriodization()
    W = 3
    offsets = per.window(N=N, W=W, device=device, sorted=False)

    packs = _chunk_offsets_even(offsets, num_chunks=3)

    T_win = T_classic_window(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, offsets)
    T_full = T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs)

    assert T_win.shape == (B, S) and T_full.shape == (B, S)
    assert T_win.dtype == dtype_T and T_full.dtype == dtype_T
    assert torch.allclose(
        T_win, T_full, rtol=rtol, atol=atol
    ), "FULL with packs(offsets) must equal WINDOW on the same offsets"


# =========================
# T5 — Monotone tail dec. and cumulative inc. (envelope weights)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T5_monotone_tail_and_cumulative(device, N, dtype_z, dtype_T):
    """
    The per-shell envelope weights w_shell(W) = sum_j alpha_j * eta_j(W).real are >= 0,
    hence the cumulative sum over shells is non-decreasing in W, and the tail
    sum is non-increasing in W (up to floating-point slack).
    """
    slack = 5e-5 if dtype_z == torch.complex64 else 1e-12

    B, M, S = 1, 6, 3
    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )

    B_, M_, N_ = z_j.shape
    z_exp = z.unsqueeze(1).expand(B_, M_, N_)
    vecd_exp = vec_d.unsqueeze(1).expand(B_, M_, N_)

    per = CPSFPeriodization()
    W_max = 4
    shells = list(
        per.iter_shells(
            N=N, start_radius=0, max_radius=W_max, device=device, sorted=False
        )
    )
    assert len(shells) >= W_max + 1

    w_shell_list = []
    for W, offW in shells:
        eta = psi_over_offsets(
            z=z_exp,
            z_j=z_j,
            vec_d=vecd_exp,
            vec_d_j=vec_d_j,
            sigma_par=sp,
            sigma_perp=sq,
            offsets=offW,
            R_j=None,
            q_max=None,
        )
        w_shell = (alpha_j.to(eta.real.dtype) * eta.real).sum(dim=-1)
        w_shell_list.append(w_shell)

    W_count = len(w_shell_list)
    W_stack = torch.stack(w_shell_list, dim=0)
    cum = torch.cumsum(W_stack, dim=0)
    total = cum[-1]
    tail = total.unsqueeze(0) - cum

    tiny = 1e-30
    for b in range(B):
        for k in range(W_count - 1):
            c0, c1 = cum[k, b].item(), cum[k + 1, b].item()
            assert c1 + tiny >= c0 * (
                1.0 - slack
            ), f"Cumulative decreased at W={k+1}: {c1} < {c0}"

        for k in range(W_count - 1):
            t0, t1 = tail[k, b].item(), tail[k + 1, b].item()
            assert (
                t1 <= t0 * (1.0 + slack) + tiny
            ), f"Tail increased at W={k+1}: {t1} > {t0}"


# =========================
# Direct run help.
# =========================
if __name__ == "__main__":
    print("\nUse pytest to run:")
    print("\tpytest -q ./test_CPSF_T_classic.py\n")
