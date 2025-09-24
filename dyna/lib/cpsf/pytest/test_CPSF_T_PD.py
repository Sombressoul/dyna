# Run as (example):
# > pytest -q dyna/lib/cpsf/pytest/test_CPSF_T_PD.py

import torch
import pytest
from typing import Tuple, List

from dyna.lib.cpsf.functional.core_math import (
    T_classic_full,
    psi_over_offsets,
)
from dyna.lib.cpsf.functional.t_pd import T_PD_window_dual
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
T13_CPU_GPU_TOLS = {
    torch.complex64: dict(rtol=1e-5, atol=1e-6),
    torch.complex128: dict(rtol=1e-6, atol=1e-12),
}


def _get_tols(dtype_c: torch.dtype, tols=_TOLS):
    t = tols[dtype_c]
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

    T0_w = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
    T1_w = T_PD_window_dual(
        z=z_s,
        z_j=zjs,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
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
    T_fullZ = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
    T_reZ = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_re_only,
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

    T_win = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
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
# T6 — Early stop semantics (tol_abs / tol_rel, with consecutive_below)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
@pytest.mark.parametrize("mode", ["abs", "rel"])
def test_T6_full_early_stop_semantics(device, N, dtype_z, dtype_T, mode):
    """
    Build packs, pre-compute per-pack contributions T_pack and their norms,
    then choose tol_* so that early stop triggers. Verify that T_full equals
    the partial sum up to (and including) the first index where the condition
    held for `consecutive_below` consecutive packs.
    """
    B, M, S = 1, 6, 3
    consecutive_below = 2
    # Use equivalence-level tolerances (same as in T4)
    rtol, atol = T4_EQUIVALENCE_TOLS

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )

    per = CPSFPeriodization()
    W = 4
    win_offsets = per.window(N=N, W=W, device=device, sorted=False)
    packs = list(
        per.iter_packed(
            N=N,
            target_points_per_pack=max(1, win_offsets.shape[0] // 5),
            start_radius=0,
            max_radius=W,
            device=device,
            sorted=False,
        )
    )
    assert len(packs) >= 3, "Need multiple packs to test early stopping"

    T_packs = []
    norms = []
    for _, _, off in packs:
        T_pack = T_PD_window_dual(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sp,
            sigma_perp=sq,
            offsets=off,
            R_j=None,
        )  # [B,S] in dtype_T
        T_packs.append(T_pack)
        norms.append(T_pack.abs().max().item())

    stop_idx = None
    if mode == "abs":
        tol_abs = max(norms[-consecutive_below:]) * 1.001
        below = 0
        T_partial = torch.zeros_like(T_packs[0])
        for i, T_pack in enumerate(T_packs):
            shell_norm = norms[i]
            cond_abs = shell_norm <= tol_abs
            below = below + 1 if cond_abs else 0
            T_partial = T_partial + T_pack
            if below >= consecutive_below:
                stop_idx = i
                break
        assert stop_idx is not None, "Early stop must trigger with chosen tol_abs"

        T_full = T_classic_full(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sp,
            sigma_perp=sq,
            packs=packs,
            R_j=None,
            q_max=None,
            tol_abs=tol_abs,
            tol_rel=None,
            consecutive_below=consecutive_below,
        )

        assert torch.allclose(
            T_full, T_partial, rtol=rtol, atol=atol
        ), f"FULL early-stop (tol_abs) must equal prefix sum up to pack {stop_idx}"

    else:  # mode == "rel"
        ratios = []
        T_partial_scan = torch.zeros_like(T_packs[0])
        accum_scan = torch.tensor(
            0.0, device=T_partial_scan.device, dtype=T_partial_scan.real.dtype
        )
        for i, T_pack in enumerate(T_packs):
            T_partial_scan = T_partial_scan + T_pack
            accum_scan = torch.maximum(accum_scan, T_partial_scan.abs().max())
            ratio_i = norms[i] / (accum_scan.item() + 1e-30)
            ratios.append(ratio_i)

        win = consecutive_below
        window_maxes = [max(ratios[i : i + win]) for i in range(len(ratios) - win + 1)]
        tol_rel = min(window_maxes) * 1.01

        # Simulate the exact early-stop logic (accum AFTER adding the pack) to get expected prefix
        below = 0
        T_partial = torch.zeros_like(T_packs[0])
        accum_norm = torch.tensor(
            0.0, device=T_partial.device, dtype=T_partial.real.dtype
        )
        stop_idx = None
        for i, T_pack in enumerate(T_packs):
            T_partial = T_partial + T_pack
            accum_norm = torch.maximum(accum_norm, T_partial.abs().max())
            shell_norm = norms[i]
            cond_rel = shell_norm <= (tol_rel * (accum_norm.item() + 1e-30))
            below = below + 1 if cond_rel else 0
            if below >= consecutive_below:
                stop_idx = i
                break
        assert stop_idx is not None, "Early stop must trigger with chosen tol_rel"

        T_full = T_classic_full(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sp,
            sigma_perp=sq,
            packs=packs,
            R_j=None,
            q_max=None,
            tol_abs=None,
            tol_rel=tol_rel,
            consecutive_below=consecutive_below,
        )

        assert torch.allclose(
            T_full, T_partial, rtol=rtol, atol=atol
        ), f"FULL early-stop (tol_rel) must equal prefix sum up to pack {stop_idx}"


# =========================
# T7 — Pack order invariance: sorted flag and in-pack permutations
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T7_full_pack_row_order_invariance(device, N, dtype_z, dtype_T):
    """
    Changing row order inside packs (and toggling `sorted=True/False`) must not
    change T when no early stop is used (pure sum over the same set of offsets).
    """
    B, M, S = 1, 5, 4
    rtol, atol = T4_EQUIVALENCE_TOLS

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )

    per = CPSFPeriodization()
    W = 3
    packs_unsorted = list(
        per.iter_packed(
            N=N,
            target_points_per_pack=128,
            start_radius=0,
            max_radius=W,
            device=device,
            sorted=False,
        )
    )
    packs_sorted = list(
        per.iter_packed(
            N=N,
            target_points_per_pack=128,
            start_radius=0,
            max_radius=W,
            device=device,
            sorted=True,  # only changes in-pack row order
        )
    )

    T_uns = T_classic_full(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        packs=packs_unsorted,
        R_j=None,
        q_max=None,
        tol_abs=None,
        tol_rel=None,
        consecutive_below=1,
    )
    T_srt = T_classic_full(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        packs=packs_sorted,
        R_j=None,
        q_max=None,
        tol_abs=None,
        tol_rel=None,
        consecutive_below=1,
    )

    assert torch.allclose(
        T_uns, T_srt, rtol=rtol, atol=atol
    ), "sorted=True/False must not change T without early stop"

    g = _gen_for(device)
    packs_perm = []
    for a, b, off in packs_unsorted:
        O = off.shape[0]
        if O > 1:
            perm = torch.randperm(O, generator=g, device=off.device)
            off_p = off[perm]
        else:
            off_p = off
        packs_perm.append((a, b, off_p))

    T_perm = T_classic_full(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        packs=packs_perm,
        R_j=None,
        q_max=None,
        tol_abs=None,
        tol_rel=None,
        consecutive_below=1,
    )

    assert torch.allclose(
        T_uns, T_perm, rtol=rtol, atol=atol
    ), "Permuting rows within packs must not change T without early stop"


# =========================
# T8 — Positivity of rho and faster tail for smaller sigmas
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
def test_T8_rho_range_and_tail_vs_sigma(device, N, dtype_z):
    slack = 5e-5 if dtype_z == torch.complex64 else 1e-12

    B, M = 1, 6
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

    sigma_base = torch.rand(B, M, generator=g, device=device, dtype=REAL) * 0.9 + 0.1
    sigma_small = sigma_base * 0.5
    sigma_large = sigma_base * 2.0

    per = CPSFPeriodization()

    off0 = per.window(N=N, W=0, device=device, sorted=False)
    z_exp = z.unsqueeze(1).expand(B, M, N)
    vecd_exp = vec_d.unsqueeze(1).expand(B, M, N)

    eta0 = psi_over_offsets(
        z=z_exp,
        z_j=z_j,
        vec_d=vecd_exp,
        vec_d_j=vec_d_j,
        sigma_par=sigma_base,
        sigma_perp=sigma_base,
        offsets=off0,
        R_j=None,
        q_max=None,
    )
    assert eta0.real.min().item() >= -slack, f"rho negative: {eta0.real.min().item()}"
    assert (
        eta0.real.max().item() <= 1.0 + slack
    ), f"rho exceeded 1: {eta0.real.max().item()}"

    W_max, W0 = 4, 1
    shells = list(
        per.iter_shells(
            N=N, start_radius=0, max_radius=W_max, device=device, sorted=False
        )
    )

    def shell_weights(sig_par, sig_perp):
        ws = []
        for _, offW in shells:
            eta = psi_over_offsets(
                z=z_exp,
                z_j=z_j,
                vec_d=vecd_exp,
                vec_d_j=vec_d_j,
                sigma_par=sig_par,
                sigma_perp=sig_perp,
                offsets=offW,
                R_j=None,
                q_max=None,
            )
            ws.append(eta.real.sum(dim=-1))
        return torch.stack(ws, dim=0)

    w_small = shell_weights(sigma_small, sigma_small)
    w_large = shell_weights(sigma_large, sigma_large)

    tail_small = w_small[W0 + 1 :].sum(dim=0)
    tail_large = w_large[W0 + 1 :].sum(dim=0)

    assert (
        tail_small <= tail_large * (1.0 + slack) + 1e-30
    ).all(), f"Tail with smaller sigma should not exceed tail with larger sigma; got {tail_small} vs {tail_large}"


# =========================
# T9 — q_max clipping: monotone convergence to unclipped
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
def test_T9_qmax_monotone_convergence(device, N, dtype_z):
    rtol, atol = _get_tols(dtype_z)
    slack = 5e-5 if dtype_z == torch.complex64 else 1e-12

    B, M = 1, 6
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
    offsets = per.window(N=N, W=3, device=device, sorted=False)
    z_exp = z.unsqueeze(1).expand(B, M, N)
    vecd_exp = vec_d.unsqueeze(1).expand(B, M, N)

    q_list = [0.1, 0.5, 1.0, 5.0, 1e9]
    w_vals = []
    for qmax in q_list:
        eta = psi_over_offsets(
            z=z_exp,
            z_j=z_j,
            vec_d=vecd_exp,
            vec_d_j=vec_d_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            offsets=offsets,
            R_j=None,
            q_max=qmax,
        )
        w_vals.append(eta.real.sum(dim=-1))
    w_stack = torch.stack(w_vals, dim=0)

    for k in range(1, len(q_list)):
        assert (
            w_stack[k] <= w_stack[k - 1] * (1.0 + slack) + 1e-30
        ).all(), f"w(q_max) must be non-increasing: {w_stack[k-1].item()} -> {w_stack[k].item()}"

    eta_true = psi_over_offsets(
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
    w_true = eta_true.real.sum(dim=-1)
    assert torch.allclose(
        w_stack[-1], w_true, rtol=rtol, atol=atol
    ), "Large q_max must approximate the unclipped result"


# =========================
# T10 — Output dtype equals T_hat_j.dtype and is independent of dtype_z
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize(
    "dtype_z_pair,dtype_T",
    [
        ((torch.complex64, torch.complex128), torch.complex64),
        ((torch.complex64, torch.complex128), torch.complex128),
        ((torch.complex128, torch.complex64), torch.complex64),
        ((torch.complex128, torch.complex64), torch.complex128),
    ],
)
def test_T10_output_dtype_and_z_dtype_independence(N, dtype_z_pair, dtype_T):
    device = torch.device("cpu")
    B, M, S = 1, 5, 4

    g = _gen_for(device)

    def _rand_c128(shape):
        xr = torch.randn(*shape, generator=g, device=device, dtype=torch.float64)
        xi = torch.randn(*shape, generator=g, device=device, dtype=torch.float64)
        return torch.complex(xr, xi)

    def _unitize(x):
        nrm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        nrm = torch.where(nrm == 0, torch.ones_like(nrm), nrm)
        return x / nrm

    z_base = _unitize(_rand_c128((B, N)))
    z_j_base = _unitize(_rand_c128((B, M, N)))
    vec_d_base = _unitize(_rand_c128((B, N)))
    vec_dj_base = _unitize(_rand_c128((B, M, N)))

    alpha_base = (
        torch.rand(B, M, generator=g, device=device, dtype=torch.float64) * 1.3 + 0.2
    )
    sigma_base = (
        torch.rand(B, M, generator=g, device=device, dtype=torch.float64) * 0.9 + 0.1
    )

    REAL_T = DTYPES_REAL[dtype_T]
    Tr = torch.randn(B, M, S, generator=g, device=device, dtype=REAL_T)
    Ti = torch.randn(B, M, S, generator=g, device=device, dtype=REAL_T)
    T_hat = torch.complex(Tr, Ti).to(dtype_T)

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=3, device=device, sorted=False)
    packs = _chunk_offsets_even(offsets, num_chunks=3)

    def _run(dtype_z):
        z = z_base.to(dtype_z)
        z_j = z_j_base.to(dtype_z)
        vec_d = vec_d_base.to(dtype_z)
        vec_dj = vec_dj_base.to(dtype_z)
        sp = sigma_base.to(DTYPES_REAL[dtype_z])
        sq = sigma_base.to(DTYPES_REAL[dtype_z])
        T_w = T_PD_window_dual(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_dj,
            T_hat_j=T_hat,
            alpha_j=alpha_base.to(REAL_T),
            sigma_par=sp,
            sigma_perp=sq,
            offsets=offsets,
        )
        T_f = T_classic_full(
            z, z_j, vec_d, vec_dj, T_hat, alpha_base.to(REAL_T), sp, sq, packs
        )
        return T_w, T_f

    dz1, dz2 = dtype_z_pair

    eff_dtype = (
        torch.complex64
        if (dz1 == torch.complex64 or dz2 == torch.complex64)
        else torch.complex128
    )
    rtol_eff, atol_eff = _get_tols(eff_dtype)

    T_w1, T_f1 = _run(dz1)
    T_w2, T_f2 = _run(dz2)
    assert T_w1.dtype == dtype_T and T_f1.dtype == dtype_T
    assert T_w2.dtype == dtype_T and T_f2.dtype == dtype_T
    assert torch.allclose(T_w1, T_w2, rtol=rtol_eff, atol=atol_eff)
    assert torch.allclose(T_f1, T_f2, rtol=rtol_eff, atol=atol_eff)


# =========================
# T11 — Permutation invariance over j
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T11_permutation_invariance_over_j(device, N, dtype_z, dtype_T):
    B, M, S = 1, 7, 3
    rtol, atol = T4_EQUIVALENCE_TOLS

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )
    per = CPSFPeriodization()
    offsets = per.window(N=N, W=3, device=device, sorted=False)
    packs = _chunk_offsets_even(offsets, num_chunks=4)

    g = _gen_for(device)
    perm = torch.randperm(M, generator=g, device=device)
    z_j_p = z_j.index_select(dim=1, index=perm)
    vec_d_j_p = vec_d_j.index_select(dim=1, index=perm)
    T_hat_j_p = T_hat_j.index_select(dim=1, index=perm)
    alpha_j_p = alpha_j.index_select(dim=1, index=perm)
    sp_p = sp.index_select(dim=1, index=perm)
    sq_p = sq.index_select(dim=1, index=perm)

    T0 = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
    T1 = T_PD_window_dual(
        z=z,
        z_j=z_j_p,
        vec_d=vec_d,
        vec_d_j=vec_d_j_p,
        T_hat_j=T_hat_j_p,
        alpha_j=alpha_j_p,
        sigma_par=sp_p,
        sigma_perp=sq_p,
        offsets=offsets,
    )
    assert torch.allclose(T0, T1, rtol=rtol, atol=atol)

    U0 = T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs)
    U1 = T_classic_full(
        z, z_j_p, vec_d, vec_d_j_p, T_hat_j_p, alpha_j_p, sp_p, sq_p, packs
    )
    assert torch.allclose(U0, U1, rtol=rtol, atol=atol)


# =========================
# T12 — Linearity in T_hat_j and in alpha_j
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T12_linearity(device, N, dtype_z, dtype_T):
    B, M, S = 1, 6, 4
    rtol, atol = T4_EQUIVALENCE_TOLS

    z, z_j, vec_d, vec_d_j, T_hat_j_A, alpha_j_A, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )
    g = _gen_for(device)
    REAL_T = DTYPES_REAL[dtype_T]
    Tr = torch.randn(B, M, S, generator=g, device=device, dtype=REAL_T)
    Ti = torch.randn(B, M, S, generator=g, device=device, dtype=REAL_T)
    T_hat_j_B = torch.complex(Tr, Ti).to(dtype_T)

    alpha_j_B = torch.rand(B, M, generator=g, device=device, dtype=REAL_T) * 1.2 + 0.1

    c1 = 0.7
    c2 = -1.3

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=3, device=device, sorted=False)
    packs = _chunk_offsets_even(offsets, num_chunks=3)

    T_A_w = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j_A,
        alpha_j=alpha_j_A,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
    T_B_w = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j_B,
        alpha_j=alpha_j_A,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
    T_C_w = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=c1 * T_hat_j_A + c2 * T_hat_j_B,
        alpha_j=alpha_j_A,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
    assert torch.allclose(T_C_w, c1 * T_A_w + c2 * T_B_w, rtol=rtol, atol=atol)

    T_A_f = T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j_A, alpha_j_A, sp, sq, packs)
    T_B_f = T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j_B, alpha_j_A, sp, sq, packs)
    T_C_f = T_classic_full(
        z,
        z_j,
        vec_d,
        vec_d_j,
        c1 * T_hat_j_A + c2 * T_hat_j_B,
        alpha_j_A,
        sp,
        sq,
        packs,
    )
    assert torch.allclose(T_C_f, c1 * T_A_f + c2 * T_B_f, rtol=rtol, atol=atol)

    T_a_w = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j_A,
        alpha_j=alpha_j_A,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
    T_b_w = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j_A,
        alpha_j=alpha_j_B,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
    T_c_w = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j_A,
        alpha_j=c1 * alpha_j_A + c2 * alpha_j_B,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
    assert torch.allclose(T_c_w, c1 * T_a_w + c2 * T_b_w, rtol=rtol, atol=atol)

    T_a_f = T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j_A, alpha_j_A, sp, sq, packs)
    T_b_f = T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j_A, alpha_j_B, sp, sq, packs)
    T_c_f = T_classic_full(
        z,
        z_j,
        vec_d,
        vec_d_j,
        T_hat_j_A,
        c1 * alpha_j_A + c2 * alpha_j_B,
        sp,
        sq,
        packs,
    )
    assert torch.allclose(T_c_f, c1 * T_a_f + c2 * T_b_f, rtol=rtol, atol=atol)


# =========================
# T13 — CPU/GPU isomorphism (if CUDA available)
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T13_cpu_gpu_isomorphism(N, dtype_z, dtype_T):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    rtol, atol = _get_tols(dtype_z, T13_CPU_GPU_TOLS)

    B, M, S = 1, 6, 4
    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=cpu
    )

    per = CPSFPeriodization()
    offsets_cpu = per.window(N=N, W=3, device=cpu, sorted=False)
    packs_cpu = _chunk_offsets_even(offsets_cpu, num_chunks=3)

    def _move_packs(packs, device):
        out = []
        for a, b, off in packs:
            out.append((a, b, off.to(device)))
        return out

    T_cpu_w = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_cpu,
    )
    T_gpu_w = T_PD_window_dual(
        z=z.to(cuda),
        z_j=z_j.to(cuda),
        vec_d=vec_d.to(cuda),
        vec_d_j=vec_d_j.to(cuda),
        T_hat_j=T_hat_j.to(cuda),
        alpha_j=alpha_j.to(cuda),
        sigma_par=sp.to(cuda),
        sigma_perp=sq.to(cuda),
        offsets=offsets_cpu.to(cuda),
    )
    assert torch.allclose(T_cpu_w.to(cpu), T_gpu_w.to(cpu), rtol=rtol, atol=atol)

    T_cpu_f = T_classic_full(
        z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs_cpu
    )
    T_gpu_f = T_classic_full(
        z.to(cuda),
        z_j.to(cuda),
        vec_d.to(cuda),
        vec_d_j.to(cuda),
        T_hat_j.to(cuda),
        alpha_j.to(cuda),
        sp.to(cuda),
        sq.to(cuda),
        _move_packs(packs_cpu, cuda),
    )
    assert torch.allclose(T_cpu_f.to(cpu), T_gpu_f.to(cpu), rtol=rtol, atol=atol)


# =========================
# T14 — Periodicity at the boundary via proper reindexing of offsets
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T14_boundary_periodicity_with_reindexed_offsets(device, N, dtype_z, dtype_T):
    B, M, S = 1, 5, 4
    rtol, atol = _get_tols(dtype_z)
    REAL = DTYPES_REAL[dtype_z]

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )

    def _cplx(r, i=0.0):
        return torch.complex(
            torch.as_tensor(r, device=device, dtype=REAL),
            torch.as_tensor(i, device=device, dtype=REAL),
        )

    z_left = z.clone()
    z_right = z.clone()
    z_left[..., 0] = _cplx(0.5)
    z_right[..., 0] = _cplx(-0.5)

    per = CPSFPeriodization()
    W = 3
    offsets = per.window(N=N, W=W, device=device, sorted=False)
    offsets_shifted = offsets.clone()
    offsets_shifted[:, 0] += 1

    T_L_w = T_PD_window_dual(
        z=z_left,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
    T_R_w = T_PD_window_dual(
        z=z_right,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets_shifted,
    )
    assert torch.allclose(T_L_w, T_R_w, rtol=rtol, atol=atol)

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
    packs_shifted = []
    for a, b, off in packs:
        off_s = off.clone()
        off_s[:, 0] += 1
        packs_shifted.append((a, b, off_s))

    T_L_f = T_classic_full(z_left, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs)
    T_R_f = T_classic_full(
        z_right, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs_shifted
    )
    assert torch.allclose(T_L_f, T_R_f, rtol=rtol, atol=atol)


# =========================
# T15 — Zero-radius/center window equals non-periodized central image
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T15_zero_radius_center_equals_central_image(device, N, dtype_z, dtype_T):
    B, M, S = 1, 4, 3
    rtol, atol = _get_tols(dtype_T)

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )

    per = CPSFPeriodization()
    off0 = per.window(N=N, W=0, device=device, sorted=False)

    B_, M_, N_ = z_j.shape
    z_exp = z.unsqueeze(1).expand(B_, M_, N_)
    vecd_exp = vec_d.unsqueeze(1).expand(B_, M_, N_)
    eta0 = psi_over_offsets(
        z=z_exp,
        z_j=z_j,
        vec_d=vecd_exp,
        vec_d_j=vec_d_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=off0,
        R_j=None,
        q_max=None,
    )
    w = alpha_j.to(eta0.real.dtype) * eta0.real
    w = w.to(T_hat_j.real.dtype)
    T_exp = (w.unsqueeze(-1).to(T_hat_j.dtype) * T_hat_j).sum(dim=-2)

    T_w = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=off0,
    )
    assert T_w.dtype == dtype_T
    assert torch.allclose(T_w, T_exp, rtol=rtol, atol=atol)

    packs0 = list(
        per.iter_packed(
            N=N,
            target_points_per_pack=128,
            start_radius=0,
            max_radius=0,
            device=device,
            sorted=False,
        )
    )
    assert len(packs0) == 1
    T_f = T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs0)
    assert torch.allclose(T_f, T_exp, rtol=rtol, atol=atol)


# =========================
# T16 — Edge forms: M=1, S=1, small N>=2 (no special-case branches)
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T16_edge_forms_small_dimensions(device, N, dtype_z, dtype_T):
    B, M, S = 1, 1, 1
    rtol, atol = T4_EQUIVALENCE_TOLS

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )
    per = CPSFPeriodization()
    W = 2
    offsets = per.window(N=N, W=W, device=device, sorted=False)
    packs = _chunk_offsets_even(offsets, num_chunks=2)

    T_w = T_PD_window_dual(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        offsets=offsets,
    )
    T_f = T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs)

    assert T_w.shape == (B, S) and T_f.shape == (B, S)
    assert T_w.dtype == dtype_T and T_f.dtype == dtype_T
    assert torch.allclose(T_w, T_f, rtol=rtol, atol=atol)


# =========================
# T17 — Continuity across the boundary: decreasing eps shrinks the difference
# =========================
@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T17_boundary_continuity_small_eps(device, N, dtype_z, dtype_T):
    B, M, S = 1, 5, 4
    REAL = DTYPES_REAL[dtype_z]

    if REAL == torch.float32:
        eps_list = [1e-3, 5e-4, 1e-4]
        slack = 1e-3
        decay_min = 0.5
    else:
        eps_list = [1e-6, 1e-7, 1e-8]
        slack = 1e-6
        decay_min = 0.2

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )

    def _cplx(r, i=0.0):
        return torch.complex(
            torch.as_tensor(r, device=device, dtype=REAL),
            torch.as_tensor(i, device=device, dtype=REAL),
        )

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=3, device=device, sorted=False)
    packs = _chunk_offsets_even(offsets, num_chunks=3)

    def _D_eps(eps):
        z_left = z.clone()
        z_left[..., 0] = _cplx(0.5 - eps)
        z_right = z.clone()
        z_right[..., 0] = _cplx(-0.5 + eps)
        e1 = torch.zeros_like(z)
        e1[..., 0] = _cplx(1.0)
        z_right_period = z_right + e1

        T_w_L = T_PD_window_dual(
            z=z_left,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sp,
            sigma_perp=sq,
            offsets=offsets,
        )
        T_w_R = T_PD_window_dual(
            z=z_right_period,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sp,
            sigma_perp=sq,
            offsets=offsets,
        )
        D_w = (T_w_L - T_w_R).abs().max().item()

        T_f_L = T_classic_full(
            z_left, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs
        )
        T_f_R = T_classic_full(
            z_right_period, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq, packs
        )
        D_f = (T_f_L - T_f_R).abs().max().item()
        return max(D_w, D_f)

    D = [_D_eps(eps) for eps in eps_list]

    for k in range(1, len(D)):
        assert (
            D[k] <= D[k - 1] * (1.0 + slack) + 1e-30
        ), f"D(eps) grew for eps {eps_list[k]} vs {eps_list[k-1]}: {D[k]} > {D[k-1]}"

    assert (
        D[-1] <= D[0] * decay_min * (1.0 + slack) + 1e-30
    ), f"D({eps_list[-1]}) not sufficiently smaller than D({eps_list[0]}): {D[-1]} vs {D[0]}"


# =========================
# T18 — gradcheck / gradgradcheck on small problem (complex128)
# =========================
def test_T18_gradcheck_and_gradgradcheck_small_complex128():
    device = torch.device("cpu")
    B, N, M, S = 1, 2, 2, 2
    dtype_z = torch.complex128
    dtype_T = torch.complex128
    REAL = DTYPES_REAL[dtype_z]

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )

    per = CPSFPeriodization()
    offsets = per.window(N=N, W=1, device=device, sorted=False)
    packs = _chunk_offsets_even(offsets, num_chunks=3)

    z_g = z.clone().detach().requires_grad_(True)
    z_j_g = z_j.clone().detach().requires_grad_(True)
    vec_d_g = vec_d.clone().detach().requires_grad_(True)
    vec_d_j_g = vec_d_j.clone().detach().requires_grad_(True)
    T_hat_g = T_hat_j.clone().detach().requires_grad_(True)
    alpha_g = alpha_j.detach().to(REAL).clone().requires_grad_(True)
    sp_g = sp.detach().to(REAL).clone().requires_grad_(True)
    sq_g = sq.detach().to(REAL).clone().requires_grad_(True)

    def f_window(z_, z_j_, vd_, vdj_, That_, a_, sp_, sq_):
        T = T_PD_window_dual(
            z=z_,
            z_j=z_j_,
            vec_d=vd_,
            vec_d_j=vdj_,
            T_hat_j=That_,
            alpha_j=a_,
            sigma_par=sp_,
            sigma_perp=sq_,
            offsets=offsets,
        )
        return T.real.sum()

    def f_full(z_, z_j_, vd_, vdj_, That_, a_, sp_, sq_):
        T = T_classic_full(z_, z_j_, vd_, vdj_, That_, a_, sp_, sq_, packs)
        return T.real.sum()

    inputs = (z_g, z_j_g, vec_d_g, vec_d_j_g, T_hat_g, alpha_g, sp_g, sq_g)

    gc_eps = 5e-6
    gc_atol = 1e-6
    gc_rtol = 1e-4

    assert torch.autograd.gradcheck(
        f_window, inputs, eps=gc_eps, atol=gc_atol, rtol=gc_rtol
    )
    assert torch.autograd.gradgradcheck(
        f_window, inputs, eps=gc_eps, atol=gc_atol, rtol=gc_rtol
    )

    assert torch.autograd.gradcheck(
        f_full, inputs, eps=gc_eps, atol=gc_atol, rtol=gc_rtol
    )
    assert torch.autograd.gradgradcheck(
        f_full, inputs, eps=gc_eps, atol=gc_atol, rtol=gc_rtol
    )


# =========================
# T19 — Tail stability: gradient decays with distance (farther shells)
# =========================
@pytest.mark.parametrize("device", [torch.device("cpu")])
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("dtype_z", DTYPES_C)
@pytest.mark.parametrize("dtype_T", DTYPES_C)
def test_T19_tail_gradient_decay(device, N, dtype_z, dtype_T):
    B, M, S = 1, 3, 2

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sp, sq = make_problem_T(
        B=B, N=N, M=M, S=S, dtype_z=dtype_z, dtype_T=dtype_T, device=device
    )

    if dtype_z == torch.complex64:
        K_seq = [3, 5, 7]
        slack = 5e-3
        decay_min = 0.3
    else:
        K_seq = [3, 5, 7]
        slack = 1e-8
        decay_min = 0.2

    def grad_norm_for_K(K):
        offs = torch.zeros(1, 2 * N, dtype=torch.long, device=device)
        offs[:, 0] = K
        z_var = z.clone().detach().requires_grad_(True)
        T = T_PD_window_dual(
            z=z_var,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sp,
            sigma_perp=sq,
            offsets=offs,
        )
        L = (T.conj() * T).real.sum()
        (gz,) = torch.autograd.grad(
            L, z_var, create_graph=False, retain_graph=False, allow_unused=False
        )
        return torch.linalg.vector_norm(gz).item()

    G = [grad_norm_for_K(K) for K in K_seq]

    tiny = 1e-30
    for i in range(1, len(G)):
        assert (
            G[i] <= G[i - 1] * (1.0 + slack) + tiny
        ), f"Gradient grew from K={K_seq[i-1]} to K={K_seq[i]}: {G[i-1]} -> {G[i]}"

    if G[0] > 1e-20:
        assert (
            G[-1] <= G[0] * decay_min * (1.0 + slack) + tiny
        ), f"Insufficient decay: G(K={K_seq[-1]})={G[-1]} vs G(K={K_seq[0]})={G[0]}"


# =========================
# Direct run help.
# =========================
if __name__ == "__main__":
    print("\nUse pytest to run:")
    print("\tpytest -q ./test_CPSF_T_classic.py\n")
