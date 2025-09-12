# Run as (example):
# > pytest -q ./dyna/lib/cpsf/pytest/test_CPSF_T_field.py

import pytest
import torch
from typing import List, Tuple, Optional

from dyna.lib.cpsf.core import CPSFCore
from dyna.lib.cpsf.periodization import (
    CPSFPeriodization,
    CPSFPeriodizationKind,
)
from dyna.lib.cpsf.contribution_store import (
    CPSFContributionStore,
    CPSFContributionSet,
    CPSFContributionField,
)
from dyna.lib.cpsf.functional.core_math import (
    R,
)

# =========================
# Global config
# =========================
TARGET_DEVICE = torch.device(
    "cpu"
)  # cuda is broken due to random generator issue...fuck it

DTYPES = [torch.complex64, torch.complex128]
NS = [2, 3, 4, 5, 6, 7, 8]
S_VALS = [1, 8, 32]
SEED = 1337
_GEN = {}

FIELD_IMPLS: List[Tuple[str, Optional[object]]] = [
    ("Tau_nearest", None),
    ("Tau_dual", None),
    ("T_classic_window", None),
    ("T_classic_full", None),
]

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


def rand_unit_vector(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
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
    n: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    gen = _gen_for(device)
    xr = torch.randn(n, generator=gen, device=device, dtype=torch.float64)
    xi = torch.randn(n, generator=gen, device=device, dtype=torch.float64)
    return (xr + 1j * xi).to(dtype)


# =========================
# Fixtures / builders
# =========================
def make_core() -> CPSFCore:
    return CPSFCore()


def make_store(N: int, S: int) -> CPSFContributionStore:
    return CPSFContributionStore(N=N, S=S)


def _real_dtype_for_complex(cdtype: torch.dtype) -> torch.dtype:
    if cdtype == torch.complex64:
        return torch.float32
    if cdtype == torch.complex128:
        return torch.float64
    raise TypeError(f"Unsupported complex dtype: {cdtype}")


def _rand_unit_complex_matrix(
    m: int,
    n: int,
    cdtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
) -> torch.Tensor:
    rdtype = _real_dtype_for_complex(cdtype)
    x = torch.randn(m, n, dtype=rdtype, device=device, generator=gen)
    y = torch.randn(m, n, dtype=rdtype, device=device, generator=gen)
    z = torch.complex(x, y).to(dtype=cdtype)
    norm = z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return z / norm


def _rand_complex_matrix(
    m: int,
    n: int,
    cdtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
) -> torch.Tensor:
    rdtype = _real_dtype_for_complex(cdtype)
    x = torch.randn(m, n, dtype=rdtype, device=device, generator=gen)
    y = torch.randn(m, n, dtype=rdtype, device=device, generator=gen)
    return torch.complex(x, y).to(dtype=cdtype)


def add_random_contributions(
    store: CPSFContributionStore,
    M: int,
    N: int,
    S: int,
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
):
    z = _rand_complex_matrix(M, N, dtype, device, gen)
    vec_d = _rand_unit_complex_matrix(M, N, dtype, device, gen)
    t_hat = _rand_complex_matrix(M, S, dtype, device, gen)
    rdtype = store.target_dtype_r
    sigma_par = 0.1 + 0.9 * torch.rand(
        M,
        1,
        dtype=rdtype,
        device=device,
        generator=gen,
    )
    sigma_perp = 0.1 + 0.9 * torch.rand(
        M,
        1,
        dtype=rdtype,
        device=device,
        generator=gen,
    )
    alpha = torch.rand(M, 1, dtype=rdtype, device=device, generator=gen)
    cs = CPSFContributionSet(
        idx=None,
        z=z,
        vec_d=vec_d,
        t_hat=t_hat,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
        alpha=alpha,
    )
    store.create(cs)


def _read_active_as_tensors(
    store: CPSFContributionStore,
    dtype: torch.dtype,
    device: torch.device,
):
    cs = store.read_all_active(
        fields=[
            CPSFContributionField.Z,
            CPSFContributionField.VEC_D,
            CPSFContributionField.T_HAT,
            CPSFContributionField.SIGMA_PAR,
            CPSFContributionField.SIGMA_PERP,
            CPSFContributionField.ALPHA,
        ]
    )

    target_c = dict(dtype=dtype, device=device)
    z_j = cs.z.to(**target_c)
    vec_d_j = cs.vec_d.to(**target_c)
    T_hat_j = cs.t_hat.to(**target_c)
    rdtype = store.target_dtype_r
    target_r = dict(dtype=rdtype, device=device)
    sigma_par = cs.sigma_par.to(**target_r).reshape(-1)
    sigma_perp = cs.sigma_perp.to(**target_r).reshape(-1)
    alpha = cs.alpha.to(**target_r).reshape(-1)

    return z_j, vec_d_j, T_hat_j, sigma_par, sigma_perp, alpha


def _offsets_iterator_zero():
    def _it(*, N: int, device: torch.device):
        return (yield torch.zeros(1, N, dtype=torch.long, device=device))

    return _it


def _k_zero(N: int, device: torch.device):
    return torch.zeros(1, N, dtype=torch.long, device=device)


def _rand_unit_vec(
    N: int,
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
):
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    v = (
        torch.randn(N, dtype=REAL, device=device, generator=gen)
        + 1j * torch.randn(N, dtype=REAL, device=device, generator=gen)
    ).to(dtype)
    return v / (v.norm() + 1e-12)


def _real_dtype_of(cdtype: torch.dtype) -> torch.dtype:
    return torch.float32 if cdtype == torch.complex64 else torch.float64


# =========================
# Tests
# =========================
# ===> T01 — shape/dtype/device & arg validation
@pytest.mark.parametrize(
    "impl_name,impl_fn", FIELD_IMPLS, ids=[n for n, _ in FIELD_IMPLS]
)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("S", S_VALS)
def test_T01_shape_dtype_device_and_args(impl_name, impl_fn, dtype, N, S):
    device = TARGET_DEVICE
    gen = _gen_for(device)

    # -------------------------------------------------------------------------
    # Positive path: minimal valid configuration
    # -------------------------------------------------------------------------
    core = make_core()
    store = make_store(N=N, S=S)
    M = 1
    add_random_contributions(
        store=store,
        M=M,
        N=N,
        S=S,
        dtype=dtype,
        device=device,
        gen=gen,
    )

    z_j, vec_d_j, T_hat_j, sigma_par, sigma_perp, alpha = _read_active_as_tensors(
        store=store, dtype=dtype, device=device
    )
    assert z_j.shape == (M, N) and vec_d_j.shape == (M, N)
    assert T_hat_j.shape == (M, S)

    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    z = (
        torch.randn(N, dtype=REAL, device=device, generator=gen)
        + 1j * torch.randn(N, dtype=REAL, device=device, generator=gen)
    ).to(dtype)
    vec_d_single = _rand_unit_vec(N, dtype, device, gen=gen)  # (N,)
    vec_d = vec_d_single.unsqueeze(0).expand(M, N).contiguous()  # (M, N)

    per_win_ok = CPSFPeriodization(kind=CPSFPeriodizationKind.WINDOW, window=1)
    per_full_ok = CPSFPeriodization(kind=CPSFPeriodizationKind.FULL, max_radius=1)
    k_ok = _k_zero(N, device)  # [1, N]

    def _call_backend(z, vec_d, sigma_par, sigma_perp, *, per_iter=None, k=None):
        if impl_name == "Tau_nearest":
            return core.Tau_nearest(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
                R_j=None,
                q_max=None,
            )
        elif impl_name == "Tau_dual":
            assert k is not None, "Tau_dual requires k"
            return core.Tau_dual(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
                k=k,
                R_j=None,
            )
        elif impl_name == "T_classic_window":
            assert per_iter is not None, "T_classic_window requires offsets_iterator"
            return core.T_classic_window(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
                offsets_iterator=per_iter,
                R_j=None,
                q_max=None,
            )
        elif impl_name == "T_classic_full":
            assert per_iter is not None, "T_classic_full requires offsets_iterator"
            return core.T_classic_full(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
                offsets_iterator=per_iter,
                R_j=None,
                q_max=None,
            )
        else:
            raise AssertionError(f"Unknown impl_name={impl_name}")

    if impl_name == "Tau_nearest":
        T = _call_backend(z, vec_d, sigma_par, sigma_perp)
    elif impl_name == "Tau_dual":
        T = _call_backend(z, vec_d, sigma_par, sigma_perp, k=k_ok)
    elif impl_name == "T_classic_window":
        T = _call_backend(
            z, vec_d, sigma_par, sigma_perp, per_iter=per_win_ok.iter_offsets
        )
    elif impl_name == "T_classic_full":
        T = _call_backend(
            z, vec_d, sigma_par, sigma_perp, per_iter=per_full_ok.iter_offsets
        )
    else:
        raise AssertionError(f"Unknown impl_name={impl_name}")

    assert T.shape == (
        S,
    ), f"{impl_name}: expected output shape (S,), got {tuple(T.shape)}"
    assert T.dtype == dtype, f"{impl_name}: expected dtype={dtype}, got {T.dtype}"
    assert (
        T.device.type == device.type
    ), f"{impl_name}: expected device={device}, got {T.device}"

    # -------------------------------------------------------------------------
    # Negative cases: validation & shape guards (minimal violations)
    # -------------------------------------------------------------------------
    sigma_bad = torch.full_like(sigma_par, fill_value=-0.5)
    if impl_name in ("Tau_nearest", "T_classic_window", "T_classic_full"):
        if impl_name == "Tau_nearest":
            with pytest.raises((ValueError, RuntimeError)):
                _ = _call_backend(
                    z,
                    vec_d,
                    sigma_bad,
                    sigma_perp,
                )
            with pytest.raises((ValueError, RuntimeError)):
                _ = _call_backend(
                    z,
                    vec_d,
                    sigma_par,
                    sigma_bad,
                )
        elif impl_name == "T_classic_window":
            with pytest.raises((ValueError, RuntimeError)):
                _ = _call_backend(
                    z,
                    vec_d,
                    sigma_bad,
                    sigma_perp,
                    per_iter=per_win_ok.iter_offsets,
                )
            with pytest.raises((ValueError, RuntimeError)):
                _ = _call_backend(
                    z,
                    vec_d,
                    sigma_par,
                    sigma_bad,
                    per_iter=per_win_ok.iter_offsets,
                )
        elif impl_name == "T_classic_full":
            with pytest.raises((ValueError, RuntimeError)):
                _ = _call_backend(
                    z,
                    vec_d,
                    sigma_bad,
                    sigma_perp,
                    per_iter=per_full_ok.iter_offsets,
                )
            with pytest.raises((ValueError, RuntimeError)):
                _ = _call_backend(
                    z,
                    vec_d,
                    sigma_par,
                    sigma_bad,
                    per_iter=per_full_ok.iter_offsets,
                )

    vec_d_bad = vec_d_single.unsqueeze(0).expand(M + 1, N).contiguous()
    if impl_name == "Tau_nearest":
        with pytest.raises((ValueError, RuntimeError)):
            _ = _call_backend(
                z,
                vec_d_bad,
                sigma_par,
                sigma_perp,
            )
    elif impl_name == "Tau_dual":
        with pytest.raises((ValueError, RuntimeError)):
            _ = _call_backend(
                z,
                vec_d_bad,
                sigma_par,
                sigma_perp,
                k=k_ok,
            )
    elif impl_name == "T_classic_window":
        with pytest.raises((ValueError, RuntimeError)):
            _ = _call_backend(
                z,
                vec_d_bad,
                sigma_par,
                sigma_perp,
                per_iter=per_win_ok.iter_offsets,
            )
    elif impl_name == "T_classic_full":
        with pytest.raises((ValueError, RuntimeError)):
            _ = _call_backend(
                z,
                vec_d_bad,
                sigma_par,
                sigma_perp,
                per_iter=per_full_ok.iter_offsets,
            )

    z_bad = (
        torch.randn(N + 1, dtype=REAL, device=device, generator=gen)
        + 1j * torch.randn(N + 1, dtype=REAL, device=device, generator=gen)
    ).to(dtype)
    if impl_name == "Tau_nearest":
        with pytest.raises((ValueError, RuntimeError)):
            _ = _call_backend(z_bad, vec_d, sigma_par, sigma_perp)
    elif impl_name == "Tau_dual":
        with pytest.raises((ValueError, RuntimeError)):
            _ = _call_backend(z_bad, vec_d, sigma_par, sigma_perp, k=k_ok)
    elif impl_name == "T_classic_window":
        with pytest.raises((ValueError, RuntimeError)):
            _ = _call_backend(
                z_bad, vec_d, sigma_par, sigma_perp, per_iter=per_win_ok.iter_offsets
            )
    elif impl_name == "T_classic_full":
        with pytest.raises((ValueError, RuntimeError)):
            _ = _call_backend(
                z_bad, vec_d, sigma_par, sigma_perp, per_iter=per_full_ok.iter_offsets
            )

    if impl_name == "T_classic_window":
        with pytest.raises((ValueError, RuntimeError)):
            _ = CPSFPeriodization(kind=CPSFPeriodizationKind.WINDOW, window=0)
    if impl_name == "Tau_dual":
        k_bad = torch.zeros(3, N + 1, dtype=torch.long, device=device)
        with pytest.raises((ValueError, RuntimeError)):
            _ = _call_backend(z, vec_d, sigma_par, sigma_perp, k=k_bad)


# ===> T02 — Validate toroidal invariance: T(z + n, d) == T(z, d) for n ∈ Z^N.
# T02 — toroidality in z
@pytest.mark.parametrize("impl_name,impl_fn", FIELD_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("S", S_VALS)
def test_T02_toroidality_in_z(impl_name, impl_fn, dtype, N, S):
    if impl_name in ("T_classic_window", "T_classic_full"):
        pytest.skip("T02 checks toroidality only for Tau_nearest / Tau_dual.")

    device = TARGET_DEVICE
    gen = _gen_for(device)
    rtol, atol = _get_tols(dtype)

    core = make_core()
    store = make_store(N=N, S=S)
    M = 1
    add_random_contributions(
        store=store,
        M=M,
        N=N,
        S=S,
        dtype=dtype,
        device=device,
        gen=gen,
    )

    z_j, vec_d_j, T_hat_j, sigma_par, sigma_perp, alpha = _read_active_as_tensors(
        store=store, dtype=dtype, device=device
    )
    assert z_j.shape == (M, N) and vec_d_j.shape == (M, N) and T_hat_j.shape[-1] == S

    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    z = (
        torch.randn(N, dtype=REAL, device=device, generator=gen)
        + 1j * torch.randn(N, dtype=REAL, device=device, generator=gen)
    ).to(dtype)
    v = (
        torch.randn(N, dtype=REAL, device=device, generator=gen)
        + 1j * torch.randn(N, dtype=REAL, device=device, generator=gen)
    ).to(dtype)
    v = v / (v.norm() + 1e-12)
    vec_d = v.unsqueeze(0).expand(M, N).contiguous()
    n_int = torch.randint(
        low=-1,
        high=2,
        size=(N,),
        device=device,
        dtype=torch.long,
        generator=gen,
    )
    n = n_int.to(dtype=REAL).to(dtype)
    z_shift = z + n

    if impl_name == "Tau_nearest":
        T1 = core.Tau_nearest(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            R_j=None,
            q_max=None,
        )
        T2 = core.Tau_nearest(
            z=z_shift,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            R_j=None,
            q_max=None,
        )

    elif impl_name == "Tau_dual":
        k0 = torch.zeros(1, N, dtype=torch.long, device=device)
        eye = torch.eye(N, dtype=torch.long, device=device)
        k = torch.cat([k0, eye, -eye], dim=0)
        T1 = core.Tau_dual(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            k=k,
            R_j=None,
        )
        T2 = core.Tau_dual(
            z=z_shift,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            k=k,
            R_j=None,
        )

    else:
        raise AssertionError(f"Unexpected impl_name for T02: {impl_name}")

    assert T1.shape == (S,) and T2.shape == (
        S,
    ), f"{impl_name}: expected (S,), got {T1.shape} and {T2.shape}"
    assert torch.allclose(T1, T2, rtol=rtol, atol=atol), (
        f"{impl_name}: toroidality violated beyond tolerances: "
        f"max_abs={torch.max((T1-T2).abs()).item():.3e}, rtol={rtol}, atol={atol}"
    )


# ===> T03 — linearity of T with respect to (a) spectral vectors T_hat_j and (b) weights alpha_j
@pytest.mark.parametrize("impl_name,impl_fn", FIELD_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("S", S_VALS)
def test_T03_linearity_in_hatT_and_alpha(impl_name, impl_fn, dtype, N, S):
    device = TARGET_DEVICE
    gen = _gen_for(device)
    rtol, atol = _get_tols(dtype)

    core = make_core()
    store = make_store(N=N, S=S)
    M = 3
    add_random_contributions(
        store=store,
        M=M,
        N=N,
        S=S,
        dtype=dtype,
        device=device,
        gen=gen,
    )

    z_j, vec_d_j, T_hat_j_base, sigma_par, sigma_perp, alpha_base = (
        _read_active_as_tensors(store=store, dtype=dtype, device=device)
    )
    assert (
        z_j.shape == (M, N) and vec_d_j.shape == (M, N) and T_hat_j_base.shape == (M, S)
    )

    REAL = _real_dtype_of(dtype)
    z = (
        torch.randn(N, generator=gen, dtype=REAL, device=device)
        + 1j * torch.randn(N, generator=gen, dtype=REAL, device=device)
    ).to(dtype)
    v = (
        torch.randn(N, generator=gen, dtype=REAL, device=device)
        + 1j * torch.randn(N, generator=gen, dtype=REAL, device=device)
    ).to(dtype)
    v = v / (v.norm() + 1e-12)
    vec_d = v.unsqueeze(0).expand(M, N).contiguous()
    k = None
    offsets_iterator = None
    if impl_name == "Tau_dual":
        k0 = torch.zeros(1, N, dtype=torch.long, device=device)
        eye = torch.eye(N, dtype=torch.long, device=device)
        k = torch.cat([k0, eye, -eye], dim=0)
    elif impl_name in ("T_classic_window", "T_classic_full"):
        offsets_iterator = _offsets_iterator_zero()

    def call_backend(T_hat_j, alpha_j):
        if impl_name == "Tau_nearest":
            return core.Tau_nearest(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha_j,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
                R_j=None,
                q_max=None,
            )
        elif impl_name == "Tau_dual":
            return core.Tau_dual(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha_j,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
                k=k,
                R_j=None,
            )
        elif impl_name == "T_classic_window":
            return core.T_classic_window(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha_j,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
                offsets_iterator=offsets_iterator,
                R_j=None,
                q_max=None,
            )
        elif impl_name == "T_classic_full":
            return core.T_classic_full(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha_j,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
                offsets_iterator=offsets_iterator,
                R_j=None,
                q_max=None,
            )
        else:
            raise AssertionError(f"Unknown impl_name={impl_name}")

    a = 0.6
    b = 0.4
    X = (
        torch.randn(
            T_hat_j_base.real.size(),
            dtype=T_hat_j_base.real.dtype,
            generator=gen,
            device=device,
        )
        + 1j
        * torch.randn(
            T_hat_j_base.real.size(),
            dtype=T_hat_j_base.real.dtype,
            generator=gen,
            device=device,
        )
    ).to(dtype)
    Y = (
        torch.randn(
            T_hat_j_base.real.size(),
            dtype=T_hat_j_base.real.dtype,
            generator=gen,
            device=device,
        )
        + 1j
        * torch.randn(
            T_hat_j_base.real.size(),
            dtype=T_hat_j_base.real.dtype,
            generator=gen,
            device=device,
        )
    ).to(dtype)

    T_X = call_backend(T_hat_j=X, alpha_j=alpha_base)
    T_Y = call_backend(T_hat_j=Y, alpha_j=alpha_base)
    T_lin_hat = call_backend(T_hat_j=(a * X + b * Y), alpha_j=alpha_base)
    T_comb_hat = a * T_X + b * T_Y

    assert T_X.shape == (S,) and T_Y.shape == (S,) and T_lin_hat.shape == (S,)
    assert torch.allclose(T_lin_hat, T_comb_hat, rtol=rtol, atol=atol), (
        f"{impl_name} (linearity in T_hat_j) failed: "
        f"max_abs={torch.max((T_lin_hat - T_comb_hat).abs()).item():.3e}"
    )

    r_dtype = _real_dtype_of(dtype)
    a_r = 0.25
    b_r = 0.75
    alpha_X = torch.rand(M, generator=gen, dtype=r_dtype, device=device)
    alpha_Y = torch.rand(M, generator=gen, dtype=r_dtype, device=device)

    T_alpha_X = call_backend(T_hat_j=T_hat_j_base, alpha_j=alpha_X)
    T_alpha_Y = call_backend(T_hat_j=T_hat_j_base, alpha_j=alpha_Y)
    T_lin_alpha = call_backend(
        T_hat_j=T_hat_j_base, alpha_j=(a_r * alpha_X + b_r * alpha_Y)
    )
    T_comb_alpha = a_r * T_alpha_X + b_r * T_alpha_Y

    assert (
        T_alpha_X.shape == (S,)
        and T_alpha_Y.shape == (S,)
        and T_lin_alpha.shape == (S,)
    )
    assert torch.allclose(T_lin_alpha, T_comb_alpha, rtol=rtol, atol=atol), (
        f"{impl_name} (linearity in alpha_j) failed: "
        f"max_abs={torch.max((T_lin_alpha - T_comb_alpha).abs()).item():.3e}"
    )


# ===> T04 — Isotropic limit: when sigma_par == sigma_perp
@pytest.mark.parametrize("impl_name,impl_fn", FIELD_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("S", S_VALS)
def test_T04_isotropic_limit_sigma_par_eq_sigma_perp(impl_name, impl_fn, dtype, N, S):
    device = TARGET_DEVICE
    gen = _gen_for(device)
    rtol, atol = _get_tols(dtype)

    def _random_Q_perp(
        N: int, dtype: torch.dtype, device: torch.device, gen: torch.Generator
    ) -> torch.Tensor:
        if N < 2:
            pytest.skip("N must be >= 2 for an orthogonal complement.")
        REAL = _real_dtype_of(dtype)
        Ar = torch.randn(N - 1, N - 1, generator=gen, dtype=REAL, device=device)
        Ai = torch.randn(N - 1, N - 1, generator=gen, dtype=REAL, device=device)
        A = (Ar + 1j * Ai).to(dtype)
        Q_perp, _ = torch.linalg.qr(A)
        Q = torch.eye(N, dtype=dtype, device=device)
        Q[1:, 1:] = Q_perp
        return Q

    core = make_core()
    store = make_store(N=N, S=S)
    M = 3
    add_random_contributions(
        store=store,
        M=M,
        N=N,
        S=S,
        dtype=dtype,
        device=device,
        gen=gen,
    )
    z_j, vec_d_j, T_hat_j, sigma_par, sigma_perp, alpha = _read_active_as_tensors(
        store=store, dtype=dtype, device=device
    )
    assert z_j.shape == (M, N) and vec_d_j.shape == (M, N)

    REAL = _real_dtype_of(dtype)
    z = (
        torch.randn(N, generator=gen, dtype=REAL, device=device)
        + 1j * torch.randn(N, generator=gen, dtype=REAL, device=device)
    ).to(dtype)
    v = (
        torch.randn(N, generator=gen, dtype=REAL, device=device)
        + 1j * torch.randn(N, generator=gen, dtype=REAL, device=device)
    ).to(dtype)
    v = v / (v.norm() + 1e-12)
    vec_d = v.unsqueeze(0).expand(M, N).contiguous()
    sigma_iso = 0.5 * (sigma_par + sigma_perp)
    sp = sigma_iso
    sq = sigma_iso
    R_base = R(vec_d_j)
    R_alt = []

    for j in range(M):
        Qj = _random_Q_perp(N=N, dtype=dtype, device=device, gen=gen)
        R_alt.append(R_base[j] @ Qj)
    R_alt = torch.stack(R_alt, dim=0)

    assert torch.allclose(R_alt[..., :, 0], R_base[..., :, 0], rtol=rtol, atol=atol)
    if N >= 3:
        assert not torch.allclose(R_alt, R_base, rtol=rtol, atol=atol)

    k = None
    offsets_iterator = None
    if impl_name == "Tau_dual":
        k0 = torch.zeros(1, N, dtype=torch.long, device=device)
        eye = torch.eye(N, dtype=torch.long, device=device)
        k = torch.cat([k0, eye, -eye], dim=0)
    elif impl_name in ("T_classic_window", "T_classic_full"):
        offsets_iterator = _offsets_iterator_zero()

    def call_backend(R_j):
        if impl_name == "Tau_nearest":
            return core.Tau_nearest(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha,
                sigma_par=sp,
                sigma_perp=sq,
                R_j=R_j,
                q_max=None,
            )
        elif impl_name == "Tau_dual":
            return core.Tau_dual(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha,
                sigma_par=sp,
                sigma_perp=sq,
                k=k,
                R_j=R_j,
            )
        elif impl_name == "T_classic_window":
            return core.T_classic_window(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha,
                sigma_par=sp,
                sigma_perp=sq,
                offsets_iterator=offsets_iterator,
                R_j=R_j,
                q_max=None,
            )
        elif impl_name == "T_classic_full":
            return core.T_classic_full(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha,
                sigma_par=sp,
                sigma_perp=sq,
                offsets_iterator=offsets_iterator,
                R_j=R_j,
                q_max=None,
            )
        else:
            raise AssertionError(f"Unknown impl_name={impl_name}")

    T_base = call_backend(R_j=R_base)
    T_rot = call_backend(R_j=R_alt)

    assert T_base.shape == (S,) and T_rot.shape == (
        S,
    ), f"{impl_name}: expected (S,), got {T_base.shape} and {T_rot.shape}"
    assert torch.allclose(T_base, T_rot, rtol=rtol, atol=atol), (
        f"{impl_name}: isotropic invariance failed: "
        f"max_abs={torch.max((T_base - T_rot).abs()).item():.3e}"
    )


# ===> T05 — Tau_dual vs T_classic_full: numerical equivalence under matched truncations
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("S", S_VALS)
def test_T05_dual_equals_classic_full(dtype, N, S):
    device = TARGET_DEVICE
    gen = _gen_for(device)
    rtol, atol = _get_tols(dtype)

    if N > 4:
        pytest.skip("T05 is validated for N <= 4 with commensurate truncations (R<=4).")

    def _k_cube(K: int, N: int, device: torch.device) -> torch.Tensor:
        rng = torch.arange(-K, K + 1, dtype=torch.long, device=device)
        grids = torch.meshgrid(*([rng] * N), indexing="ij")
        return torch.stack([g.reshape(-1) for g in grids], dim=-1)

    core = make_core()
    store = make_store(N=N, S=S)
    M = 8
    add_random_contributions(
        store=store,
        M=M,
        N=N,
        S=S,
        dtype=dtype,
        device=device,
        gen=gen,
    )

    z_j_raw, vec_d_j, T_hat_j, _, _, alpha = _read_active_as_tensors(
        store=store,
        dtype=dtype,
        device=device,
    )
    z_j = z_j_raw.real.to(dtype)
    vec_d = vec_d_j.clone()
    r_dtype = store.target_dtype_r
    sigma_val = 1.0
    sp = torch.full((M,), sigma_val, dtype=r_dtype, device=device)
    sq = torch.full((M,), sigma_val, dtype=r_dtype, device=device)
    REAL = _real_dtype_of(dtype)
    B = 2
    z_batch_real = torch.randn(B, N, generator=gen, dtype=REAL, device=device)
    z_batch = z_batch_real.to(dtype)
    per_full = CPSFPeriodization(kind=CPSFPeriodizationKind.FULL, max_radius=8)
    k = _k_cube(K=8, N=N, device=device)

    for b in range(B):
        z = z_batch[b]

        T_full = core.T_classic_full(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha,
            sigma_par=sp,
            sigma_perp=sq,
            offsets_iterator=per_full.iter_offsets,
            R_j=None,
            q_max=None,
        )

        T_dual = core.Tau_dual(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha,
            sigma_par=sp,
            sigma_perp=sq,
            k=k,
            R_j=None,
        )

        assert T_full.shape == (S,) and T_dual.shape == (S,)
        diff = (T_full - T_dual).abs()
        assert torch.allclose(T_full, T_dual, rtol=rtol, atol=atol), (
            f"Dual vs Full mismatch (N={N}, S={S}, dtype={dtype}): "
            f"max_abs={diff.max().item():.3e}, rtol={rtol}, atol={atol}"
        )


# ===> T06 — window convergence: T_classic_window → T_classic_full as window grows
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("S", S_VALS)
def test_T06_window_converges_to_full(dtype, N, S):
    device = TARGET_DEVICE
    gen = _gen_for(device)
    rtol, atol = _get_tols(dtype)

    if N > 4:
        pytest.skip("T06 uses R_full<=5; skip for N>4 to keep cost reasonable.")

    core = make_core()
    store = make_store(N=N, S=S)
    M = 8
    add_random_contributions(
        store=store,
        M=M,
        N=N,
        S=S,
        dtype=dtype,
        device=device,
        gen=gen,
    )

    z_j, vec_d_j, T_hat_j, sigma_par, sigma_perp, alpha = _read_active_as_tensors(
        store=store, dtype=dtype, device=device
    )
    assert z_j.shape == (M, N) and vec_d_j.shape == (M, N) and T_hat_j.shape == (M, S)

    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    z = (
        torch.randn(N, dtype=REAL, device=device, generator=gen)
        + 1j * torch.randn(N, dtype=REAL, device=device, generator=gen)
    ).to(dtype)
    vec_d_single = _rand_unit_vec(N, dtype, device, gen=gen)
    vec_d = vec_d_single.unsqueeze(0).expand(M, N).contiguous()
    R_full = 5
    W_list = [1, 2, 3, 4, 5]
    per_full = CPSFPeriodization(kind=CPSFPeriodizationKind.FULL, max_radius=R_full)
    T_ref = core.T_classic_full(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
        offsets_iterator=per_full.iter_offsets,
        R_j=None,
        q_max=None,
    )
    assert T_ref.shape == (S,)
    ref_mag = T_ref.abs().max().item()

    jitter = rtol * ref_mag + atol
    eps = torch.finfo(torch.float32 if dtype == torch.complex64 else torch.float64).eps
    floor = max(jitter, eps)

    errs = []
    for W in W_list:
        per_win = CPSFPeriodization(kind=CPSFPeriodizationKind.WINDOW, window=W)
        T_win = core.T_classic_window(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            offsets_iterator=per_win.iter_offsets,
            R_j=None,
            q_max=None,
        )
        assert T_win.shape == (S,)
        errs.append((T_win - T_ref).abs().max().item())

    for i in range(len(errs) - 1):
        assert errs[i + 1] <= errs[i] + 5.0 * floor, (
            f"Non-monotone window convergence: W={W_list[i]} err={errs[i]:.3e} "
            f"→ W={W_list[i+1]} err={errs[i+1]:.3e}, floor={floor:.3e}"
        )

    assert errs[-1] <= 10.0 * floor, (
        f"Final window W={W_list[-1]} not close enough to FULL (R={R_full}): "
        f"err_end={errs[-1]:.3e}, floor={floor:.3e}"
    )

    if errs[0] > 20.0 * floor:
        target = max(0.5 * errs[0], 10.0 * floor)
        assert errs[-1] <= target, (
            "Insufficient global improvement: "
            f"err_start={errs[0]:.3e}, err_end={errs[-1]:.3e}, "
            f"target<=max(0.5*start, 10*floor)={target:.3e}, floor={floor:.3e}"
        )


# ===> T07 — Nearest-image accuracy in the interior (large Δq_min)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("S", S_VALS)
def test_T07_nearest_image_accuracy_interior(dtype, N, S):
    device = TARGET_DEVICE
    gen = _gen_for(device)
    rtol, atol = _get_tols(dtype)

    if N > 5:
        pytest.skip("T07 uses R_full<=7; skip for N>5 to keep cost reasonable.")

    core = make_core()
    store = make_store(N=N, S=S)

    M = 1
    add_random_contributions(
        store=store,
        M=M,
        N=N,
        S=S,
        dtype=dtype,
        device=device,
        gen=gen,
    )

    z_j, vec_d_j, T_hat_j, sigma_par0, sigma_perp0, alpha = _read_active_as_tensors(
        store=store,
        dtype=dtype,
        device=device,
    )
    assert z_j.shape == (M, N) and vec_d_j.shape == (M, N) and T_hat_j.shape == (M, S)

    z = z_j.squeeze(0)
    vec_d = vec_d_j.clone()
    r_dtype = store.target_dtype_r
    sigma_val = 0.05
    sp = torch.full((M,), sigma_val, dtype=r_dtype, device=device)
    sq = torch.full((M,), sigma_val, dtype=r_dtype, device=device)
    R_full = 7
    per_full = CPSFPeriodization(kind=CPSFPeriodizationKind.FULL, max_radius=R_full)
    T_full = core.T_classic_full(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha,
        sigma_par=sp,
        sigma_perp=sq,
        offsets_iterator=per_full.iter_offsets,
        R_j=None,
        q_max=None,
    )
    assert T_full.shape == (S,)

    T_zero = core.T_classic_full(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha,
        sigma_par=sp,
        sigma_perp=sq,
        offsets_iterator=_offsets_iterator_zero(),
        R_j=None,
        q_max=None,
    )
    assert T_zero.shape == (S,)

    T_near = core.Tau_nearest(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha,
        sigma_par=sp,
        sigma_perp=sq,
        R_j=None,
        q_max=None,
    )
    assert T_near.shape == (S,)

    ref_mag = T_full.abs().max().item()
    floor = rtol * ref_mag + atol

    diff_near_zero = (T_near - T_zero).abs().max().item()
    assert diff_near_zero <= 10.0 * floor, (
        "Tau_nearest must equal the n=0 term in the interior regime: "
        f"max|near - zero|={diff_near_zero:.3e}, floor={floor:.3e}, "
        f"N={N}, S={S}, dtype={dtype}"
    )

    tail = (T_full - T_zero).abs().max().item()
    diff_near_full = (T_near - T_full).abs().max().item()
    assert diff_near_full <= tail + 10.0 * floor, (
        "Nearest-image error must be bounded by the true tail magnitude: "
        f"|near-full|={diff_near_full:.3e} vs tail={tail:.3e}, floor={floor:.3e}, "
        f"N={N}, S={S}, dtype={dtype}"
    )

    size_factor = max(1.0, float(S))
    assert tail <= 1e3 * floor * size_factor, (
        "Tail should be small in the interior (large Δq_min): "
        f"tail={tail:.3e}, floor={floor:.3e}, S={S}, dtype={dtype}"
    )


# ===> T08 — batch/broadcast: формы, батч-совместимость входов, согласование устройств
@pytest.mark.parametrize("impl_name,impl_fn", FIELD_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("S", S_VALS)
def test_T08_batch_and_broadcast_semantics(impl_name, impl_fn, dtype, N, S):
    """
    Что проверяем:
      - Работа с батчами запросов (разные batch_shape), широковещание параметров вкладов
      - Корректные выходные формы и согласование устройств при смешанных входах
    Шаги:
      - Сформировать несколько b_shape конфигураций
      - Проверить, что выход имеет ожидаемую форму и dtype/device
    """
    pass


# ===> T09 — CPU/GPU parity (опционально при наличии CUDA)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("impl_name,impl_fn", FIELD_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", [2, 8, 16])
@pytest.mark.parametrize("S", [1, 8])
def test_T09_device_propagation_and_cpu_gpu_parity(impl_name, impl_fn, dtype, N, S):
    """
    Что проверяем:
      - Совпадение результатов CPU/GPU и корректную пропагацию устройств
    Шаги:
      - Дублировать данные на cpu/cuda, сравнить результаты
    """
    pass


# ===> T10 — детерминизм при фиксированном SEED
@pytest.mark.parametrize("impl_name,impl_fn", FIELD_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("S", S_VALS)
def test_T10_determinism_with_fixed_seed(impl_name, impl_fn, dtype, N, S):
    """
    Что проверяем:
      - Повторяемость результата при фиксированном SEED (для генерации вкладов/запросов)
    Шаги:
      - Дважды построить одинаковые входы при одинаковом SEED и сравнить результаты
    """
    pass


# ===> T11 — согласованность чтения из CPSFContributionStore (snapshot/live/overlay, если применимо)
@pytest.mark.parametrize("impl_name,impl_fn", FIELD_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("S", S_VALS)
def test_T11_store_read_consistency_policies(impl_name, impl_fn, dtype, N, S):
    """
    Что проверяем:
      - Что выбор политики чтения из хранилища (если поддерживается флагами) не ломает корректность T
    Шаги:
      - Заполнить store; читать через разные режимы (напр. snapshot vs live)
      - Вызвать backend и убедиться в идентичности/ожидаемой эквивалентности результатов
    """
    pass


# ===> T12 — численная устойчивость (экстремальные sigma^{||}, sigma^{⊥})
@pytest.mark.parametrize("impl_name,impl_fn", FIELD_IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("S", S_VALS)
@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6])
def test_T12_numerical_stability_extreme_sigmas(impl_name, impl_fn, dtype, N, S, scale):
    """
    Что проверяем:
      - Устойчивость при экстремальных масштабах затуханий
    Шаги:
      - Масштабировать (sigma_par, sigma_perp) общей константой 'scale'
      - Сравнить разные реализации между собой и/или с эталонной (T_classic_full)
    """
    pass


# =========================
# Direct run help
# =========================
if __name__ == "__main__":
    print("\nUse pytest to run:")
    print("\tpytest -q ./test_CPSF_T_field.py\n")
