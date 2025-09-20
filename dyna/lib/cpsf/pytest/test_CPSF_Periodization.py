# Run as (example):
# > pytest -q .\dyna\lib\cpsf\pytest\test_CPSF_Periodization.py

import itertools
import pytest
import torch

from typing import Tuple, Set

from dyna.lib.cpsf.periodization import CPSFPeriodization


# =========================
# Global config
# =========================
TARGET_DEVICE = torch.device("cpu")
INT_DTYPES = [torch.int64, torch.int32]  # integer offsets only
NS = [1, 2, 3]  # complex dimensions (keep sizes manageable)
WS = [0, 1, 2]  # radii (window grows as (2W+1)^(2N))
SEED = 1337


def _as_set(x: torch.Tensor) -> Set[Tuple[int, ...]]:
    return set(tuple(map(int, row)) for row in x.to("cpu").tolist())


def _has_unique_rows(x: torch.Tensor) -> bool:
    if x.numel() == 0:
        return True
    uniq = torch.unique(x, dim=0)
    return uniq.shape[0] == x.shape[0]


# =========================
# P01 — ctor & dtype validation
# =========================
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_P01_ctor_and_dtype_ok(dtype):
    g = CPSFPeriodization(enable_cache=True, dtype=dtype)
    assert isinstance(g, CPSFPeriodization)
    assert g.dtype == dtype


@pytest.mark.parametrize(
    "bad_dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_P01b_ctor_rejects_non_integer_dtype(bad_dtype):
    with pytest.raises(ValueError):
        _ = CPSFPeriodization(dtype=bad_dtype)


# =========================
# P02 — shape/dtype/device & arg checks
# =========================
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", WS)
def test_P02_window_shell_basic_contract(dtype, N, W):
    dev = TARGET_DEVICE
    g = CPSFPeriodization(dtype=dtype)
    win = g.window(N=N, W=W, device=dev)
    sh = g.shell(N=N, W=W, device=dev)

    assert win.dtype == dtype and sh.dtype == dtype
    assert win.device.type == dev.type and sh.device.type == dev.type
    assert win.ndim == 2 and sh.ndim == 2
    assert win.shape[1] == 2 * N and sh.shape[1] == 2 * N

    expect_win = CPSFPeriodization.window_size(N=N, W=W)
    expect_sh = CPSFPeriodization.shell_size(N=N, W=W)
    assert win.shape[0] == expect_win
    assert sh.shape[0] == expect_sh

    if win.numel() > 0:
        assert int(win.abs().amax().item()) <= W
    if sh.numel() > 0:
        assert int(sh.abs().amax().item()) == W


def test_P02b_arg_validation_errors():
    g = CPSFPeriodization()
    with pytest.raises(ValueError):
        _ = g.window(N=0, W=1)
    with pytest.raises(ValueError):
        _ = g.window(N=1, W=-1)
    with pytest.raises(ValueError):
        _ = g.shell(N=0, W=0)
    with pytest.raises(ValueError):
        _ = g.shell(N=1, W=-2)
    with pytest.raises(ValueError):
        _ = g.iter_shells(N=0).__iter__().__next__()
    with pytest.raises(ValueError):
        _ = list(g.iter_shells(N=1, start_radius=-1, max_radius=0))
    with pytest.raises(ValueError):
        _ = list(g.iter_packed(N=1, target_points_per_pack=0))


# =========================
# P03 — sizes & formulas
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", WS)
def test_P03_size_formulas(N, W):
    w_ref = (2 * W + 1) ** (2 * N)
    assert CPSFPeriodization.window_size(N=N, W=W) == w_ref
    if W == 0:
        assert CPSFPeriodization.shell_size(N=N, W=W) == 1
    else:
        s_ref = (2 * W + 1) ** (2 * N) - (2 * W - 1) ** (2 * N)
        assert CPSFPeriodization.shell_size(N=N, W=W) == s_ref


# =========================
# P04 — no duplicates inside window/shell
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", WS)
def test_P04_no_duplicates(N, W):
    g = CPSFPeriodization()
    win = g.window(N=N, W=W, device=TARGET_DEVICE)
    sh = g.shell(N=N, W=W, device=TARGET_DEVICE)
    assert _has_unique_rows(win), f"Duplicates in window(N={N}, W={W})"
    assert _has_unique_rows(sh), f"Duplicates in shell(N={N}, W={W})"


# =========================
# P05 — window == disjoint union of shells 0..W
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", WS)
def test_P05_window_equals_union_of_shells(N, W):
    g = CPSFPeriodization()
    dev = TARGET_DEVICE
    win = g.window(N=N, W=W, device=dev)
    shells = [g.shell(N=N, W=w, device=dev) for w in range(W + 1)]

    tot = 0
    all_sets = []
    for sw in shells:
        tot += sw.shape[0]
        all_sets.append(_as_set(sw))
    disjoint = sum(len(s) for s in all_sets) == len(set().union(*all_sets))
    assert disjoint, "Shells overlap"

    assert tot == win.shape[0], "Sum of shell sizes != window size"

    union_set = set().union(*all_sets)
    win_set = _as_set(win)
    assert union_set == win_set, "Window content != union(shells)"


# =========================
# P06 — iter_shells correctness (start/max), equality to direct shells
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("maxr", [0, 2])
def test_P06_iter_shells_matches_direct(N, start, maxr):
    g = CPSFPeriodization()
    dev = TARGET_DEVICE
    seen = []
    for W, S in g.iter_shells(N=N, start_radius=start, max_radius=maxr, device=dev):
        assert W >= start and W <= maxr

        Sd = g.shell(N=N, W=W, device=dev)
        assert Sd.shape == S.shape
        assert _as_set(Sd) == _as_set(S)

        seen.append((W, S))

    assert len(seen) == (maxr - start + 1)
    assert [w for (w, _) in seen] == list(range(start, maxr + 1))


# =========================
# P07 — pack_offsets correctness (concat + lengths)
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("maxr", [0, 1, 2])
def test_P07_pack_offsets_concat_and_lengths(N, maxr):
    g = CPSFPeriodization()
    dev = TARGET_DEVICE
    offsets, lengths = g.pack_offsets(N=N, max_radius=maxr, device=dev)

    assert offsets.ndim == 2 and offsets.shape[1] == 2 * N
    assert lengths.shape == (maxr + 1,)

    for w in range(maxr + 1):
        assert lengths[w].item() == CPSFPeriodization.shell_size(N=N, W=w)

    assert offsets.shape[0] == CPSFPeriodization.window_size(N=N, W=maxr)

    shells = [g.shell(N=N, W=w, device=dev) for w in range(maxr + 1)]
    union_set = set().union(*[_as_set(s) for s in shells])
    off_set = _as_set(offsets)
    assert off_set == union_set


# =========================
# P08 — iter_packed respects target size and covers range
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("maxr", [1, 2])
@pytest.mark.parametrize("target", [100, 1000])
def test_P08_iter_packed_semantics(N, start, maxr, target):
    g = CPSFPeriodization()
    dev = TARGET_DEVICE

    packs = []
    covered = set()
    total = 0
    prev_end = None

    for w0, w1, P in g.iter_packed(
        N=N,
        target_points_per_pack=target,
        start_radius=start,
        max_radius=maxr,
        device=dev,
    ):
        assert w0 <= w1
        assert P.ndim == 2 and P.shape[1] == 2 * N
        assert P.shape[0] <= max(target, CPSFPeriodization.shell_size(N=N, W=w0))

        if prev_end is not None:
            assert w0 == prev_end + 1
        prev_end = w1

        for w in range(w0, w1 + 1):
            covered.add(w)
        total += P.shape[0]
        packs.append(P)

    expect_shells = set(range(start, maxr + 1))
    assert covered == expect_shells, "iter_packed did not cover expected shell radii"

    pack_set = set().union(*[_as_set(p) for p in packs]) if packs else set()
    shells = [g.shell(N=N, W=w, device=dev) for w in range(start, maxr + 1)]
    shell_set = set().union(*[_as_set(s) for s in shells]) if shells else set()

    assert pack_set == shell_set
    assert total == sum(
        CPSFPeriodization.shell_size(N=N, W=w) for w in range(start, maxr + 1)
    )


# =========================
# P09 — determinism across calls (content equality)
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", WS)
def test_P09_determinism(N, W):
    g = CPSFPeriodization()
    dev = TARGET_DEVICE

    w1 = g.window(N=N, W=W, device=dev)
    w2 = g.window(N=N, W=W, device=dev)
    s1 = g.shell(N=N, W=W, device=dev)
    s2 = g.shell(N=N, W=W, device=dev)

    assert _as_set(w1) == _as_set(w2)
    assert _as_set(s1) == _as_set(s2)


# =========================
# P10 — cache transparency (on/off yields identical content)
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", WS)
def test_P10_cache_transparency(N, W):
    dev = TARGET_DEVICE
    g_on = CPSFPeriodization(enable_cache=True)
    g_off = CPSFPeriodization(enable_cache=False)

    win_on = g_on.window(N=N, W=W, device=dev)
    win_off = g_off.window(N=N, W=W, device=dev)
    sh_on = g_on.shell(N=N, W=W, device=dev)
    sh_off = g_off.shell(N=N, W=W, device=dev)

    assert _as_set(win_on) == _as_set(win_off)
    assert _as_set(sh_on) == _as_set(sh_off)


# =========================
# P11 — CPU/GPU parity (optional)
# =========================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("N", [1, 3])
@pytest.mark.parametrize("W", [0, 2])
def test_P11_device_propagation_and_cpu_gpu_parity(N, W):
    g_cpu = CPSFPeriodization(dtype=torch.int64)
    g_gpu = CPSFPeriodization(dtype=torch.int64)

    win_cpu = g_cpu.window(N=N, W=W, device=torch.device("cpu"))
    sh_cpu = g_cpu.shell(N=N, W=W, device=torch.device("cpu"))
    win_gpu = g_gpu.window(N=N, W=W, device=torch.device("cuda"))
    sh_gpu = g_gpu.shell(N=N, W=W, device=torch.device("cuda"))

    assert win_cpu.device.type == "cpu" and sh_cpu.device.type == "cpu"
    assert win_gpu.device.type == "cuda" and sh_gpu.device.type == "cuda"
    assert _as_set(win_cpu) == _as_set(win_gpu.to("cpu"))
    assert _as_set(sh_cpu) == _as_set(sh_gpu.to("cpu"))


# =========================
# P12 — edge cases: W=0, N=1
# =========================
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_P12_edge_cases(dtype):
    dev = TARGET_DEVICE
    g = CPSFPeriodization(dtype=dtype)
    win = g.window(N=1, W=0, device=dev)
    sh = g.shell(N=1, W=0, device=dev)
    assert win.shape == (1, 2)
    assert sh.shape == (1, 2)
    assert int(win.abs().sum().item()) == 0
    assert int(sh.abs().sum().item()) == 0


@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("start,maxr", [(2, 1), (3, 2)])
def test_P13_empty_ranges_emit_nothing(N, start, maxr):
    g = CPSFPeriodization()
    dev = TARGET_DEVICE

    shells = list(g.iter_shells(N=N, start_radius=start, max_radius=maxr, device=dev))
    assert shells == []

    packs = list(
        g.iter_packed(
            N=N,
            target_points_per_pack=10,
            start_radius=start,
            max_radius=maxr,
            device=dev,
        )
    )
    assert packs == []


# =========================
# P14 — type validation: non-integer / invalid args rejected
# =========================
def test_P14_type_validation_non_integer_args():
    g = CPSFPeriodization()

    with pytest.raises(ValueError):
        _ = g.window(N=1.0, W=1)
    with pytest.raises(ValueError):
        _ = g.window(N=1, W=1.0)
    with pytest.raises(ValueError):
        _ = g.shell(N=1, W=-1)
    with pytest.raises(ValueError):
        _ = g.window(N=torch.tensor(1.0), W=1)
    with pytest.raises(ValueError):
        _ = g.shell(N=1, W=torch.tensor(1.0))
    with pytest.raises(ValueError):
        _ = g.window(N=torch.tensor(1, dtype=torch.int64), W=1)
    with pytest.raises(ValueError):
        _ = g.window(N=True, W=1)
    with pytest.raises(ValueError):
        _ = g.window(N=1, W=False)
    with pytest.raises(ValueError):
        _ = g.window(N=0, W=0)


# =========================
# P15 — iter_shells with max_radius=None: take first K shells and compare
# =========================
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("K", [1, 3])
def test_P15_iter_shells_unbounded_prefix(N, K):
    g = CPSFPeriodization()
    dev = TARGET_DEVICE
    it = g.iter_shells(N=N, start_radius=0, max_radius=None, device=dev)

    taken = list(itertools.islice(it, K))
    assert [w for (w, _) in taken] == list(range(0, K))
    for w, S in taken:
        Sd = g.shell(N=N, W=w, device=dev)
        assert _as_set(S) == _as_set(Sd)


# =========================
# P16 — lattice symmetry: x in set => -x in set (window & shell)
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", WS)
def test_P16_lattice_sign_symmetry(N, W):
    g = CPSFPeriodization()
    dev = TARGET_DEVICE
    win = g.window(N=N, W=W, device=dev)
    shl = g.shell(N=N, W=W, device=dev)

    win_set = _as_set(win)
    shl_set = _as_set(shl)

    for row in list(win_set)[: min(len(win_set), 1000)]:
        neg = tuple(-v for v in row)
        assert neg in win_set
    for row in list(shl_set)[: min(len(shl_set), 1000)]:
        neg = tuple(-v for v in row)
        assert neg in shl_set


# =========================
# P17 — pack_offsets cumulative lengths equal window_size
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", WS)
def test_P17_pack_offsets_cumulative(N, W):
    g = CPSFPeriodization()
    dev = TARGET_DEVICE
    offsets, lengths = g.pack_offsets(N=N, max_radius=W, device=dev)

    cum = 0
    for w in range(W + 1):
        cum += lengths[w].item()
        assert cum == CPSFPeriodization.window_size(N=N, W=w)


# =========================
# P18 — iter_packed extremes: tiny target -> many packs; huge target -> single pack
# =========================
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("W", [1, 2])
def test_P18_iter_packed_extremes(N, W):
    g = CPSFPeriodization()
    dev = TARGET_DEVICE

    packs_tiny = list(
        g.iter_packed(
            N=N,
            target_points_per_pack=1,
            start_radius=0,
            max_radius=W,
            device=dev,
        )
    )
    assert len(packs_tiny) >= (W + 1)

    covered_tiny = set()
    for w0, w1, P in packs_tiny:
        assert w0 <= w1
        for w in range(w0, w1 + 1):
            covered_tiny.add(w)
        assert P.shape[1] == 2 * N and P.shape[0] >= 1
    assert covered_tiny == set(range(0, W + 1))

    total = CPSFPeriodization.window_size(N=N, W=W)
    packs_huge = list(
        g.iter_packed(
            N=N,
            target_points_per_pack=total,
            start_radius=0,
            max_radius=W,
            device=dev,
        )
    )
    assert len(packs_huge) == 1

    w0, w1, P = packs_huge[0]
    assert w0 == 0 and w1 == W
    assert P.shape[0] == total


# =========================
# P19 — CPU/GPU cache separation (when CUDA is available)
# =========================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_P19_cache_separation_cpu_gpu():
    g = CPSFPeriodization(enable_cache=True)
    N, W = 2, 1

    cpu_win = g.window(N=N, W=W, device=torch.device("cpu"))
    gpu_win = g.window(N=N, W=W, device=torch.device("cuda"))

    assert cpu_win.device.type == "cpu"
    assert gpu_win.device.type == "cuda"
    assert _as_set(cpu_win) == _as_set(gpu_win.to("cpu"))


# =========================
# P20 — extra edge: N=3, W=0 returns a single zero row of width 6
# =========================
def test_P20_edge_N3_W0():
    g = CPSFPeriodization()
    dev = TARGET_DEVICE
    t = g.window(N=3, W=0, device=dev)
    assert t.shape == (1, 6)
    assert int(t.abs().sum().item()) == 0


# =========================
# P21 — ctor rejects torch.bool dtype
# =========================
def test_P21_ctor_rejects_bool_dtype():
    with pytest.raises(ValueError):
        _ = CPSFPeriodization(dtype=torch.bool)


# =========================
# P22 — ctor cache guard: too small max_cache_bytes_per_tensor
# =========================
def test_P22_ctor_cache_size_guard():
    with pytest.raises(ValueError):
        _ = CPSFPeriodization(
            enable_cache=True, max_cache_bytes_per_tensor=0, dtype=torch.int64
        )
    with pytest.raises(ValueError):
        _ = CPSFPeriodization(
            enable_cache=True, max_cache_bytes_per_tensor=1, dtype=torch.int32
        )


# =========================
# P23 — default device is CPU when not provided
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", WS)
def test_P23_default_device_cpu(N, W):
    g = CPSFPeriodization()
    win = g.window(N=N, W=W)  # device=None
    shl = g.shell(N=N, W=W)  # device=None
    assert win.device.type == "cpu"
    assert shl.device.type == "cpu"


# =========================
# P24 — contiguity of outputs (window/shell/pack)
# =========================
@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", WS)
def test_P24_contiguity_window_shell(N, W):
    g = CPSFPeriodization()
    win = g.window(N=N, W=W, device=TARGET_DEVICE)
    shl = g.shell(N=N, W=W, device=TARGET_DEVICE)
    assert win.is_contiguous()
    assert shl.is_contiguous()


@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("maxr", [0, 1, 2])
def test_P24b_contiguity_pack(N, maxr):
    g = CPSFPeriodization()
    off, lengths = g.pack_offsets(N=N, max_radius=maxr, device=TARGET_DEVICE)
    assert off.is_contiguous()
    assert lengths.is_contiguous()


@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("W", WS)
def test_P24c_contiguity_iter_packed(N, W):
    g = CPSFPeriodization()
    total = CPSFPeriodization.window_size(N=N, W=W)
    packs = list(
        g.iter_packed(
            N=N,
            target_points_per_pack=max(1, total // 3),
            start_radius=0,
            max_radius=W,
            device=TARGET_DEVICE,
        )
    )
    for _, _, P in packs:
        assert P.is_contiguous()


# =========================
# P25 — window_size/shell_size: type/arg guards (statics)
# =========================
def test_P25_window_shell_size_guards():
    # bad types
    with pytest.raises(ValueError):
        _ = CPSFPeriodization.window_size(N=1.0, W=0)
    with pytest.raises(ValueError):
        _ = CPSFPeriodization.window_size(N=1, W=0.0)
    with pytest.raises(ValueError):
        _ = CPSFPeriodization.shell_size(N=1.0, W=0)
    with pytest.raises(ValueError):
        _ = CPSFPeriodization.shell_size(N=1, W=0.0)
    # bad bounds
    with pytest.raises(ValueError):
        _ = CPSFPeriodization.window_size(N=0, W=0)
    with pytest.raises(ValueError):
        _ = CPSFPeriodization.shell_size(N=0, W=0)
    with pytest.raises(ValueError):
        _ = CPSFPeriodization.window_size(N=1, W=-1)
    with pytest.raises(ValueError):
        _ = CPSFPeriodization.shell_size(N=1, W=-1)


# =========================
# P26 — monotonicity of sizes over W (sanity invariant)
# =========================
@pytest.mark.parametrize("N", NS)
def test_P26_monotonic_sizes(N):
    prev_win = None
    prev_sh = None
    for W in range(0, 4):
        cur_win = CPSFPeriodization.window_size(N=N, W=W)
        cur_sh = CPSFPeriodization.shell_size(N=N, W=W)
        assert cur_sh >= 1
        if prev_win is not None:
            assert cur_win > prev_win
        if prev_sh is not None:
            assert cur_sh >= prev_sh or W <= 1
        prev_win, prev_sh = cur_win, cur_sh


# =========================
# P27 — CUDA: iter_shells parity & device propagation
# =========================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("maxr", [0, 1, 2])
def test_P27_cuda_iter_shells_parity(N, maxr):
    g = CPSFPeriodization()
    cpu = torch.device("cpu")
    gpu = torch.device("cuda")

    cpu_pairs = list(g.iter_shells(N=N, start_radius=0, max_radius=maxr, device=cpu))
    gpu_pairs = list(g.iter_shells(N=N, start_radius=0, max_radius=maxr, device=gpu))

    assert [w for (w, _) in cpu_pairs] == [w for (w, _) in gpu_pairs]
    for (_, Scpu), (_, Sgpu) in zip(cpu_pairs, gpu_pairs):
        assert Scpu.device.type == "cpu"
        assert Sgpu.device.type == "cuda"
        assert _as_set(Scpu) == _as_set(Sgpu.to("cpu"))


# =========================
# P28 — CUDA: pack_offsets parity & device propagation
# =========================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("W", [0, 1, 2])
def test_P28_cuda_pack_offsets_parity(N, W):
    g = CPSFPeriodization()
    cpu = torch.device("cpu")
    gpu = torch.device("cuda")

    off_cpu, len_cpu = g.pack_offsets(N=N, max_radius=W, device=cpu)
    off_gpu, len_gpu = g.pack_offsets(N=N, max_radius=W, device=gpu)

    assert off_cpu.device.type == "cpu" and len_cpu.device.type == "cpu"
    assert off_gpu.device.type == "cuda" and len_gpu.device.type == "cuda"
    assert _as_set(off_cpu) == _as_set(off_gpu.to("cpu"))
    assert torch.equal(len_cpu.cpu(), len_gpu.cpu())


# =========================
# P29 — CUDA: iter_packed parity (coverage & content)
# =========================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("W", [1, 2])
@pytest.mark.parametrize("target", [16, 1000])
def test_P29_cuda_iter_packed_parity(N, W, target):
    g = CPSFPeriodization()
    cpu = torch.device("cpu")
    gpu = torch.device("cuda")

    packs_cpu = list(
        g.iter_packed(
            N=N,
            target_points_per_pack=target,
            start_radius=0,
            max_radius=W,
            device=cpu,
        )
    )
    packs_gpu = list(
        g.iter_packed(
            N=N,
            target_points_per_pack=target,
            start_radius=0,
            max_radius=W,
            device=gpu,
        )
    )

    assert [(a, b) for (a, b, _) in packs_cpu] == [(a, b) for (a, b, _) in packs_gpu]

    cov_cpu = set()
    for a, b, _ in packs_cpu:
        cov_cpu.update(range(a, b + 1))
    cov_gpu = set()
    for a, b, _ in packs_gpu:
        cov_gpu.update(range(a, b + 1))
    assert cov_cpu == cov_gpu == set(range(0, W + 1))

    cpu_union = (
        set().union(*[_as_set(P) for (_, _, P) in packs_cpu]) if packs_cpu else set()
    )
    gpu_union = (
        set().union(*[_as_set(P.to("cpu")) for (_, _, P) in packs_gpu])
        if packs_gpu
        else set()
    )
    assert cpu_union == gpu_union

    for _, _, P in packs_gpu:
        assert P.device.type == "cuda"


# =========================
# Direct run help
# =========================
if __name__ == "__main__":
    print("\nUse pytest to run:")
    print("\tpytest -q ./test_CPSF_Periodization.py\n")
