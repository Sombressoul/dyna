# Run examples:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_classic --mode compare --N 4 --M 8 --S 16 --W 2 --batch 16 --dtype_z c64 --dtype_T c64 --device cuda --iters 50 --warmup 10
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_classic --mode all --N 4 --M 8 --S 16 --W 2 --batch 8 --dtype_z c128 --dtype_T c64 --device auto --iters 50 --warmup 10 --profile
#
# Throughput metrics:
#   eta-terms/s = (B * M * O_total) / (avg_time_seconds)
#   points/s    = B / (avg_time_seconds)
#
# Notes:
# - Inputs are prepared once outside timed loops.
# - "iter_shells" is fed to T_classic_full via a thin wrapper that yields (w,w,offsets).

import argparse, time, math, torch
from torch.profiler import profile, ProfilerActivity, schedule

from ..functional.core_math import T_classic_window, T_classic_full
from ..periodization import CPSFPeriodization


def _fmt_bytes(x: int) -> str:
    u = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    v = float(x)
    while v >= 1024 and i < len(u) - 1:
        v /= 1024
        i += 1
    return f"{v:.2f} {u[i]}"


def _pick_device(sel: str) -> torch.device:
    if sel == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(sel)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("WARN: CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return dev


def _sync(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _cuda_metrics_begin(dev: torch.device):
    if dev.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return (mem0, start, end)


def _cuda_metrics_end(dev: torch.device, token):
    if dev.type != "cuda":
        return None, None, None
    mem0, start, end = token
    end.record()
    torch.cuda.synchronize()
    dt_ms = start.elapsed_time(end)
    mem1 = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    alloc_delta = max(0, mem1 - mem0)
    return dt_ms, peak, alloc_delta


def _time_block(dev: torch.device, fn):
    if dev.type == "cuda":
        tok = _cuda_metrics_begin(dev)
        out = fn()
        dt_ms, peak, alloc = _cuda_metrics_end(dev, tok)
        return dt_ms, peak, alloc, out
    else:
        t0 = time.perf_counter()
        out = fn()
        dt_ms = (time.perf_counter() - t0) * 1e3
        return dt_ms, None, None, out


def _profile_block(dev: torch.device, steps: int, body):
    acts = [ProfilerActivity.CPU] + (
        [ProfilerActivity.CUDA] if dev.type == "cuda" else []
    )
    sch = schedule(wait=1, warmup=1, active=max(steps, 3), repeat=1)
    with profile(
        activities=acts, schedule=sch, record_shapes=False, profile_memory=True
    ) as prof:
        for _ in range(max(steps, 3)):
            body()
            prof.step()
    print(
        prof.key_averages().table(
            sort_by=(
                "self_cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
            ),
            row_limit=15,
        )
    )


def _window_size(N: int, W: int) -> int:
    if W == 0:
        return 1
    t = (W << 1) + 1
    return int(pow(t, 2 * N))


def _shell_size(N: int, W: int) -> int:
    if W == 0:
        return 1
    tw = (W << 1) + 1
    tm = (W << 1) - 1
    return int(pow(tw, 2 * N) - pow(tm, 2 * N))


def _make_cplx(
    B, *dims, dtype: torch.dtype, device: torch.device, seed: int, unitize: bool = False
):
    g = torch.Generator(device=device).manual_seed(seed)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    xr = torch.randn(B, *dims, generator=g, device=device, dtype=REAL)
    xi = torch.randn(B, *dims, generator=g, device=device, dtype=REAL)
    z = torch.complex(xr, xi).to(dtype)
    if unitize:
        n = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        n = torch.where(n == 0, torch.ones_like(n), n)
        z = z / n
    return z


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["window", "iter_shells", "iter_packed", "compare", "all"],
        default="compare",
    )
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument(
        "--W", type=int, default=2, help="L∞ window/shell radius; O=|window(N,W)|"
    )
    ap.add_argument(
        "--start", type=int, default=0, help="Start radius for streaming modes"
    )
    ap.add_argument(
        "--max",
        dest="maxr",
        type=int,
        default=None,
        help="Max radius for streaming modes (default: W)",
    )
    ap.add_argument(
        "--stream_pack_target",
        type=int,
        default=20000,
        help="Rows per pack target for iter_packed",
    )
    ap.add_argument("--batch", type=int, default=512, help="Number of target points B")
    ap.add_argument("--dtype_z", choices=["c64", "c128"], default="c64")
    ap.add_argument("--dtype_T", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--tol_abs", type=float, default=None, help="FULL early-stop tol_abs"
    )
    ap.add_argument(
        "--tol_rel", type=float, default=None, help="FULL early-stop tol_rel"
    )
    ap.add_argument(
        "--consecutive_below",
        type=int,
        default=1,
        help="FULL consecutive-below shells to stop",
    )
    ap.add_argument(
        "--qmax",
        type=float,
        default=None,
        help="Optional q_max clamp; None disables clamping",
    )
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--verify_devices", action="store_true")
    args = ap.parse_args()

    dev = _pick_device(args.device)
    try:
        torch.set_default_device(dev.type)
    except Exception:
        pass

    dtype_z = torch.complex64 if args.dtype_z == "c64" else torch.complex128
    dtype_T = torch.complex64 if args.dtype_T == "c64" else torch.complex128
    REAL_z = torch.float32 if dtype_z == torch.complex64 else torch.float64
    REAL_T = torch.float32 if dtype_T == torch.complex64 else torch.float64

    B, N, M, S, W = args.batch, args.N, args.M, args.S, args.W
    start = args.start
    maxr = args.maxr if args.maxr is not None else W

    if N < 2:
        raise SystemExit("CPSF requires N >= 2.")
    if W < 0 or start < 0 or (maxr is not None and maxr < 0):
        raise SystemExit("Radii must be >= 0.")
    if start > maxr:
        print("WARN: start > max; streaming ranges will be empty.")

    print(
        f"Device={dev.type}, dtype_z={dtype_z}, dtype_T={dtype_T}, "
        f"B={B}, M={M}, S={S}, N={N}, W={W}, start={start}, max={maxr}, "
        f"warmup={args.warmup}, iters={args.iters}, pack_target={args.stream_pack_target}, "
        f"tol_abs={args.tol_abs}, tol_rel={args.tol_rel}, consecutive_below={args.consecutive_below}, q_max={args.qmax}"
    )

    gen = CPSFPeriodization()

    # Offsets for window and streaming
    offsets_win = gen.window(N=N, W=W, device=dev, sorted=False)  # [O, 2N]
    O_win = int(offsets_win.shape[0])
    O_stream = (
        sum(_shell_size(N, w) for w in range(start, maxr + 1))
        if (maxr is not None and start <= maxr)
        else 0
    )
    print(
        f"window: O={O_win} rows | stream range: start={start}, max={maxr}, O_total={O_stream}"
    )

    # Inputs (outside timing)
    # Random complex inputs; normalize direction vectors
    z = _make_cplx(B, N, dtype=dtype_z, device=dev, seed=10, unitize=False)  # [B,N]
    z_j = _make_cplx(B * M, N, dtype=dtype_z, device=dev, seed=20, unitize=False).view(
        B, M, N
    )  # [B,M,N]
    vec_d = _make_cplx(B, N, dtype=dtype_z, device=dev, seed=30, unitize=True)  # [B,N]
    vec_d_j = _make_cplx(
        B * M, N, dtype=dtype_z, device=dev, seed=40, unitize=True
    ).view(
        B, M, N
    )  # [B,M,N]

    # Weights and kernel scales
    alpha_j = 0.2 + 1.3 * torch.rand(B, M, device=dev, dtype=REAL_T)  # [B,M] real
    sigma_par = torch.full((B, M), 0.9, device=dev, dtype=REAL_z)  # [B,M] real
    sigma_perp = torch.full((B, M), 0.5, device=dev, dtype=REAL_z)  # [B,M] real

    # Semantic channels (dtype_T)
    Tr = torch.randn(B, M, S, device=dev, dtype=REAL_T)
    Ti = torch.randn(B, M, S, device=dev, dtype=REAL_T)
    T_hat_j = torch.complex(Tr, Ti).to(dtype_T)  # [B,M,S]

    q_max = float(args.qmax) if args.qmax is not None else None

    # Streaming packs
    packs_packed = list(
        gen.iter_packed(
            N=N,
            target_points_per_pack=args.stream_pack_target,
            start_radius=start,
            max_radius=maxr,
            device=dev,
            sorted=False,
        )
    )

    def _packs_from_shells():
        for w, S_shell in gen.iter_shells(
            N=N, start_radius=start, max_radius=maxr, device=dev, sorted=False
        ):
            yield (w, w, S_shell)

    # Device verification
    if args.verify_devices:
        out_w = T_classic_window(
            z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp, offsets_win
        )
        print(
            f"verify.window: z.device={z.device}, offsets.device={offsets_win.device}, out.device={out_w.device}"
        )
        assert out_w.device == z.device
        out_f = T_classic_full(
            z,
            z_j,
            vec_d,
            vec_d_j,
            T_hat_j,
            alpha_j,
            sigma_par,
            sigma_perp,
            packs_packed,
            tol_abs=args.tol_abs,
            tol_rel=args.tol_rel,
            consecutive_below=args.consecutive_below,
        )
        assert out_f.device == z.device
        del out_w, out_f

    # Warmups
    def warmup_window():
        for _ in range(args.warmup):
            _ = T_classic_window(
                z,
                z_j,
                vec_d,
                vec_d_j,
                T_hat_j,
                alpha_j,
                sigma_par,
                sigma_perp,
                offsets_win,
                q_max=q_max,
            )
        _sync(dev)

    def warmup_full_packed():
        for _ in range(args.warmup):
            _ = T_classic_full(
                z,
                z_j,
                vec_d,
                vec_d_j,
                T_hat_j,
                alpha_j,
                sigma_par,
                sigma_perp,
                packs_packed,
                q_max=q_max,
                tol_abs=args.tol_abs,
                tol_rel=args.tol_rel,
                consecutive_below=args.consecutive_below,
            )
        _sync(dev)

    def warmup_full_shells():
        for _ in range(args.warmup):
            _ = T_classic_full(
                z,
                z_j,
                vec_d,
                vec_d_j,
                T_hat_j,
                alpha_j,
                sigma_par,
                sigma_perp,
                _packs_from_shells(),
                q_max=q_max,
                tol_abs=args.tol_abs,
                tol_rel=args.tol_rel,
                consecutive_below=args.consecutive_below,
            )
        _sync(dev)

    # Single-run profiles
    def profile_window():
        _profile_block(
            dev,
            steps=5,
            body=lambda: T_classic_window(
                z,
                z_j,
                vec_d,
                vec_d_j,
                T_hat_j,
                alpha_j,
                sigma_par,
                sigma_perp,
                offsets_win,
                q_max=q_max,
            ),
        )

    def profile_full_packed():
        _profile_block(
            dev,
            steps=3,
            body=lambda: T_classic_full(
                z,
                z_j,
                vec_d,
                vec_d_j,
                T_hat_j,
                alpha_j,
                sigma_par,
                sigma_perp,
                packs_packed,
                q_max=q_max,
                tol_abs=args.tol_abs,
                tol_rel=args.tol_rel,
                consecutive_below=args.consecutive_below,
            ),
        )

    def profile_full_shells():
        _profile_block(
            dev,
            steps=3,
            body=lambda: T_classic_full(
                z,
                z_j,
                vec_d,
                vec_d_j,
                T_hat_j,
                alpha_j,
                sigma_par,
                sigma_perp,
                _packs_from_shells(),
                q_max=q_max,
                tol_abs=args.tol_abs,
                tol_rel=args.tol_rel,
                consecutive_below=args.consecutive_below,
            ),
        )

    # Bench bodies
    def bench_window():
        warmup_window()
        if args.profile:
            profile_window()

        times, peak_max, alloc_max = [], 0, 0
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for _ in range(args.iters):
            dt_ms, peak, alloc, out = _time_block(
                dev,
                lambda: T_classic_window(
                    z,
                    z_j,
                    vec_d,
                    vec_d_j,
                    T_hat_j,
                    alpha_j,
                    sigma_par,
                    sigma_perp,
                    offsets_win,
                    q_max=q_max,
                ),
            )
            # prevent DCE and add sync:
            _ = out.real.sum().item()
            times.append(dt_ms)
            if peak:
                peak_max = max(peak_max, peak)
            if alloc:
                alloc_max = max(alloc_max, alloc)

        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
        secs = avg / 1e3
        terms = B * M * O_win
        thr_terms = terms / secs
        thr_points = B / secs

        print("\n=== T_classic_window (monolithic window) ===")
        print(f"O={O_win:,} offsets, B={B:,}, M={M:,}, S={S:,}")
        print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
        print(f"Throughput:    {thr_terms:,.0f} eta-terms/s   (B*M*O)")
        print(f"Per-target:    {thr_points:,.0f} points/s     (B)")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")

        return avg, std, thr_terms, thr_points, out  # out for cross-checks

    def _bench_full(packs, label: str, O_total: int):
        # choose matching warmup/profile
        if label == "iter_packed":
            warmup_full_packed()
            if args.profile:
                profile_full_packed()
        else:
            warmup_full_shells()
            if args.profile:
                profile_full_shells()

        times, peak_max, alloc_max = [], 0, 0
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for _ in range(args.iters):
            dt_ms, peak, alloc, out = _time_block(
                dev,
                lambda: T_classic_full(
                    z,
                    z_j,
                    vec_d,
                    vec_d_j,
                    T_hat_j,
                    alpha_j,
                    sigma_par,
                    sigma_perp,
                    packs,
                    q_max=q_max,
                    tol_abs=args.tol_abs,
                    tol_rel=args.tol_rel,
                    consecutive_below=args.consecutive_below,
                ),
            )
            _ = out.real.sum().item()
            times.append(dt_ms)
            if peak:
                peak_max = max(peak_max, peak)
            if alloc:
                alloc_max = max(alloc_max, alloc)

        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
        secs = avg / 1e3
        terms = B * M * O_total if O_total > 0 else 0
        thr_terms = (terms / secs) if O_total > 0 else 0.0
        thr_points = B / secs

        print(f"\n=== T_classic_full ({label}) ===")
        print(f"O_total={O_total:,} offsets (start..max), B={B:,}, M={M:,}, S={S:,}")
        print(f"Avg time/pass: {avg:.3f} ms  (± {std:.3f} ms)")
        print(f"Throughput:    {thr_terms:,.0f} eta-terms/s")
        print(f"Per-target:    {thr_points:,.0f} points/s")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")

        return avg, std, thr_terms, thr_points, out

    # Execute modes
    modes = (
        [args.mode] if args.mode != "all" else ["window", "iter_shells", "iter_packed"]
    )
    results = {}

    for m in modes:
        if m == "window":
            results["window"] = bench_window()
        elif m == "iter_shells":
            results["iter_shells"] = _bench_full(
                _packs_from_shells(), "iter_shells", O_stream
            )
        elif m == "iter_packed":
            results["iter_packed"] = _bench_full(packs_packed, "iter_packed", O_stream)
        elif m == "compare":
            # run WINDOW and ITER_PACKED, then compare numeric equivalence
            res_w = bench_window()
            res_p = _bench_full(packs_packed, "iter_packed", O_stream)
            results["window"] = res_w
            results["iter_packed"] = res_p

            out_w = res_w[-1]
            out_p = res_p[-1]

            with torch.no_grad():
                diff = out_w - out_p
                abs_diff = diff.abs()
                denom = out_p.abs() + 1e-30
                rel = abs_diff / denom

                max_abs = abs_diff.max().item()
                mean_abs = abs_diff.mean().item()
                max_rel = rel.max().item()
                mean_rel = rel.mean().item()

            print("\n=== WINDOW vs ITER_PACKED: numeric equivalence ===")
            print(
                f"Shapes: {tuple(out_w.shape)}; dtype: {out_w.dtype} vs {out_p.dtype}; device: {out_w.device} vs {out_p.device}"
            )
            print(f"Abs diff: max={max_abs:.3e}, mean={mean_abs:.3e}")
            print(f"Rel diff: max={max_rel:.3e}, mean={mean_rel:.3e}")
        else:
            raise ValueError(f"Unknown mode: {m}")

    return results


if __name__ == "__main__":
    main()
