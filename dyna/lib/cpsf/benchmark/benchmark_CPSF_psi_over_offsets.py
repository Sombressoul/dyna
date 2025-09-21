# Run as (examples):
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_psi_over_offsets --mode window --N 4 --M 8 --W 2 --batch 16 --dtype c64 --device cuda --iters 50 --warmup 10
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_psi_over_offsets --mode compare --stream iter_shells --N 4 --M 8 --W 2 --batch 16 --dtype c64 --device cuda --iters 50 --warmup 10
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_psi_over_offsets --mode compare --stream iter_packed --N 4 --M 8 --W 2 --batch 16 --dtype c64 --device cuda --iters 50 --warmup 10
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_psi_over_offsets --mode all --N 4 --M 8 --W 2 --batch 16 --dtype c64 --device cuda --iters 50 --warmup 10
#
# Throughput metric:
#   rho-terms/s = (B * M * O_total) / (avg_time_seconds)
#
# Notes:
# - vec_d and vec_d_j are shaped [B, M, N] to satisfy delta_vec_d’s exact-shape requirement.
# - z is [B, 1, N] to broadcast over M contributors.
# - sigma_par / sigma_perp are 0-D tensors (real scalars) for safe broadcasting.

import argparse, time, torch
from torch.profiler import profile, ProfilerActivity, schedule

from ..functional.core_math import psi_over_offsets
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
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    sch = schedule(wait=1, warmup=1, active=max(steps, 3), repeat=1)
    with profile(activities=acts, schedule=sch, record_shapes=False, profile_memory=True) as prof:
        for _ in range(max(steps, 3)):
            body()
            prof.step()
    print(
        prof.key_averages().table(
            sort_by=("self_cuda_time_total" if dev.type == "cuda" else "cpu_time_total"),
            row_limit=15,
        )
    )


def make_unit_batch(B: int, N: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    """Complex gaussian -> normalize to unit vectors, shape (B, N)."""
    g = torch.Generator(device=device).manual_seed(seed)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    xr = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    xi = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    v = (xr + 1j * xi).to(dtype)
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    n = torch.where(n.real == 0, torch.ones_like(n), n)
    return v / n


def make_complex_batch(B: int, N: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    """Unnormalized complex gaussian, shape (B, N)."""
    g = torch.Generator(device=device).manual_seed(seed)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    xr = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    xi = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    return (xr + 1j * xi).to(dtype)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["window", "iter_shells", "iter_packed", "compare", "all"], default="compare")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--W", type=int, default=2, help="L∞ window/shell radius; O=|window(N,W)|")
    ap.add_argument("--start", type=int, default=0, help="Start radius for streaming modes")
    ap.add_argument("--max", dest="maxr", type=int, default=None, help="Max radius for streaming modes (default: W)")
    ap.add_argument("--stream", choices=["iter_shells", "iter_packed"], default="iter_packed",
                    help="Streaming method used in 'compare' mode")
    ap.add_argument("--stream_pack_target", type=int, default=20000, help="Target rows per pack for iter_packed")
    ap.add_argument("--batch", type=int, default=4096, help="Number of target points B")
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--verify_devices", action="store_true")
    ap.add_argument("--qmax", type=float, default=None, help="Optional q_max clamp; None disables clamping")
    args = ap.parse_args()

    dev = _pick_device(args.device)
    try:
        torch.set_default_device(dev.type)
    except Exception:
        pass

    dtype = torch.complex64 if args.dtype == "c64" else torch.complex128
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    B, N, M, W = args.batch, args.N, args.M, args.W
    start = args.start
    maxr = args.maxr if args.maxr is not None else W
    if N < 2:
        raise SystemExit("CPSF requires N >= 2.")
    if W < 0 or start < 0 or (maxr is not None and maxr < 0):
        raise SystemExit("Radii must be >= 0.")
    if start > maxr:
        print("WARN: start > max; streaming ranges will be empty.")

    print(
        f"Device={dev.type}, dtype={dtype}, B={B}, M={M}, N={N}, W={W}, start={start}, max={maxr}, "
        f"warmup={args.warmup}, iters={args.iters}, stream={args.stream}, pack_target={args.stream_pack_target}, q_max={args.qmax}"
    )

    # Offsets generator
    gen = CPSFPeriodization()

    # Window offsets (monolithic)
    offsets_win = gen.window(N=N, W=W, device=dev, sorted=False)  # [O, 2N]
    O_win = int(offsets_win.shape[0])
    print(f"window: O={O_win} rows")

    # Streaming total O (start..maxr)
    def window_size(N, W):
        if W == 0:
            return 1
        tw = (W << 1) + 1
        return int(pow(tw, 2 * N))

    def shell_size(N, W):
        if W == 0:
            return 1
        tw = (W << 1) + 1
        tm = (W << 1) - 1
        return int(pow(tw, 2 * N) - pow(tm, 2 * N))

    O_stream = sum(shell_size(N, w) for w in range(start, maxr + 1)) if (maxr is not None and start <= maxr) else 0
    print(f"stream range: start={start}, max={maxr}, O_total={O_stream}")

    # Build inputs (outside timed loops)
    z = make_complex_batch(B, N, dtype, dev, seed=10).unsqueeze(1)                 # [B, 1, N]
    z_j = make_complex_batch(B * M, N, dtype, dev, seed=20).view(B, M, N)          # [B, M, N]
    vec_d_j = make_unit_batch(B * M, N, dtype, dev, seed=30).view(B, M, N)         # [B, M, N]
    vec_d_query = make_unit_batch(B, N, dtype, dev, seed=40)                        # [B, N]
    vec_d = vec_d_query.unsqueeze(1).expand(B, M, N).contiguous()                   # [B, M, N]
    sigma_par = torch.tensor(0.9, dtype=REAL, device=dev)                           # 0-D
    sigma_perp = torch.tensor(0.5, dtype=REAL, device=dev)                          # 0-D
    q_max = float(args.qmax) if args.qmax is not None else None

    # Sanity: device check
    if args.verify_devices:
        out = psi_over_offsets(
            z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
            sigma_par=sigma_par, sigma_perp=sigma_perp,
            offsets=offsets_win, R_j=None, q_max=q_max
        )
        print(f"verify: z.device={z.device}, offsets.device={offsets_win.device}, out.device={out.device}")
        assert out.device == z.device
        del out

    def warmup_window():
        for _ in range(args.warmup):
            _ = psi_over_offsets(
                z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
                sigma_par=sigma_par, sigma_perp=sigma_perp,
                offsets=offsets_win, R_j=None, q_max=q_max
            )
        _sync(dev)

    def warmup_iter_shells():
        for _ in range(args.warmup):
            acc = torch.zeros(B, M, dtype=REAL, device=dev) if z.dim() == 3 else None
            for w, S in gen.iter_shells(N=N, start_radius=start, max_radius=maxr, device=dev, sorted=False):
                if S.numel() == 0:
                    continue
                out = psi_over_offsets(
                    z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
                    sigma_par=sigma_par, sigma_perp=sigma_perp,
                    offsets=S, R_j=None, q_max=q_max
                )
                if acc is not None:
                    acc = acc + out
        _sync(dev)

    def warmup_iter_packed():
        for _ in range(args.warmup):
            acc = torch.zeros(B, M, dtype=REAL, device=dev) if z.dim() == 3 else None
            for a, b, P in gen.iter_packed(
                N=N, target_points_per_pack=args.stream_pack_target,
                start_radius=start, max_radius=maxr, device=dev, sorted=False
            ):
                if P.numel() == 0:
                    continue
                out = psi_over_offsets(
                    z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
                    sigma_par=sigma_par, sigma_perp=sigma_perp,
                    offsets=P, R_j=None, q_max=q_max
                )
                if acc is not None:
                    acc = acc + out
        _sync(dev)

    def bench_window():
        # Warmup
        warmup_window()

        # Optional profile
        if args.profile:
            _profile_block(dev, steps=5, body=lambda: psi_over_offsets(
                z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
                sigma_par=sigma_par, sigma_perp=sigma_perp,
                offsets=offsets_win, R_j=None, q_max=q_max
            ))

        times, peak_max, alloc_max = [], 0, 0
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for _ in range(args.iters):
            dt_ms, peak, alloc, out = _time_block(
                dev,
                lambda: psi_over_offsets(
                    z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
                    sigma_par=sigma_par, sigma_perp=sigma_perp,
                    offsets=offsets_win, R_j=None, q_max=q_max
                ),
            )
            _ = out.real.sum().item()
            times.append(dt_ms)
            if peak:  peak_max = max(peak_max, peak)
            if alloc: alloc_max = max(alloc_max, alloc)

        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
        secs = avg / 1e3
        terms = B * M * O_win
        thr_terms = terms / secs
        thr_points = B / secs

        print("\n=== psi_over_offsets — window (monolithic) ===")
        print(f"O={O_win:,} offsets, B={B:,}, M={M:,}")
        print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
        print(f"Throughput:    {thr_terms:,.0f} rho-terms/s   (B*M*O)")
        print(f"Per-target:    {thr_points:,.0f} points/s     (B)")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")

        return avg, std, thr_terms, thr_points

    def bench_iter_shells():
        # Warmup
        warmup_iter_shells()

        if args.profile:
            def _body():
                acc = torch.zeros(B, M, dtype=REAL, device=dev)
                for w, S in gen.iter_shells(N=N, start_radius=start, max_radius=maxr, device=dev, sorted=False):
                    if S.numel() == 0:
                        continue
                    acc = acc + psi_over_offsets(
                        z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
                        sigma_par=sigma_par, sigma_perp=sigma_perp,
                        offsets=S, R_j=None, q_max=q_max
                    )
                _ = acc.sum().item()
            _profile_block(dev, steps=3, body=_body)

        times, peak_max, alloc_max = [], 0, 0
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for _ in range(args.iters):
            def _run():
                acc = torch.zeros(M, dtype=REAL, device=dev)
                # Sum across shells
                for w, S in gen.iter_shells(N=N, start_radius=start, max_radius=maxr, device=dev, sorted=False):
                    if S.numel() == 0:
                        continue
                    acc = acc + psi_over_offsets(
                        z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
                        sigma_par=sigma_par, sigma_perp=sigma_perp,
                        offsets=S, R_j=None, q_max=q_max
                    )
                return acc

            dt_ms, peak, alloc, out = _time_block(dev, _run)
            _ = out.sum().item()
            times.append(dt_ms)
            if peak:  peak_max = max(peak_max, peak)
            if alloc: alloc_max = max(alloc_max, alloc)

        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
        secs = avg / 1e3
        terms = B * M * O_stream
        thr_terms = (terms / secs) if O_stream > 0 else 0.0
        thr_points = B / secs

        print("\n=== psi_over_offsets — streaming (iter_shells) ===")
        print(f"O_total={O_stream:,} offsets (start..max), B={B:,}, M={M:,}")
        print(f"Avg time/pass: {avg:.3f} ms  (± {std:.3f} ms)")
        print(f"Throughput:    {thr_terms:,.0f} rho-terms/s")
        print(f"Per-target:    {thr_points:,.0f} points/s")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")

        return avg, std, thr_terms, thr_points

    def bench_iter_packed():
        # Warmup
        warmup_iter_packed()

        if args.profile:
            def _body():
                acc = torch.zeros(B, M, dtype=REAL, device=dev)
                for a, b, P in gen.iter_packed(
                    N=N, target_points_per_pack=args.stream_pack_target,
                    start_radius=start, max_radius=maxr, device=dev, sorted=False
                ):
                    if P.numel() == 0:
                        continue
                    acc = acc + psi_over_offsets(
                        z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
                        sigma_par=sigma_par, sigma_perp=sigma_perp,
                        offsets=P, R_j=None, q_max=q_max
                    )
                _ = acc.sum().item()
            _profile_block(dev, steps=3, body=_body)

        times, peak_max, alloc_max = [], 0, 0
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for _ in range(args.iters):
            def _run():
                acc = torch.zeros(M, dtype=REAL, device=dev)
                for a, b, P in gen.iter_packed(
                    N=N, target_points_per_pack=args.stream_pack_target,
                    start_radius=start, max_radius=maxr, device=dev, sorted=False
                ):
                    if P.numel() == 0:
                        continue
                    acc = acc + psi_over_offsets(
                        z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
                        sigma_par=sigma_par, sigma_perp=sigma_perp,
                        offsets=P, R_j=None, q_max=q_max
                    )
                return acc

            dt_ms, peak, alloc, out = _time_block(dev, _run)
            _ = out.sum().item()
            times.append(dt_ms)
            if peak:  peak_max = max(peak_max, peak)
            if alloc: alloc_max = max(alloc_max, alloc)

        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
        secs = avg / 1e3
        terms = B * M * O_stream
        thr_terms = (terms / secs) if O_stream > 0 else 0.0
        thr_points = B / secs

        print("\n=== psi_over_offsets — streaming (iter_packed) ===")
        print(f"O_total={O_stream:,} offsets (start..max), B={B:,}, M={M:,}")
        print(f"Avg time/pass: {avg:.3f} ms  (± {std:.3f} ms)")
        print(f"Throughput:    {thr_terms:,.0f} rho-terms/s")
        print(f"Per-target:    {thr_points:,.0f} points/s")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")

        return avg, std, thr_terms, thr_points

    # Execute
    modes = [args.mode] if args.mode != "all" else ["window", "iter_shells", "iter_packed"]

    results = {}
    for m in modes:
        if m == "window":
            results["window"] = bench_window()
        elif m == "iter_shells":
            results["iter_shells"] = bench_iter_shells()
        elif m == "iter_packed":
            results["iter_packed"] = bench_iter_packed()
        elif m == "compare":
            # Always run window + selected streaming method
            results["window"] = bench_window()
            if args.stream == "iter_shells":
                results["iter_shells"] = bench_iter_shells()
            else:
                results["iter_packed"] = bench_iter_packed()

    # Simple side-by-side summary for compare/all
    if len(results) > 1:
        print("\n=== Summary (avg over iterations) ===")
        def _fmt(r):
            avg, std, thr_terms, thr_points = r
            return f"{avg:8.3f} ms  | {thr_terms:>11,.0f} terms/s | {thr_points:>10,.0f} pts/s"
        for k in ["window", "iter_shells", "iter_packed"]:
            if k in results:
                print(f"{k:>12}: {_fmt(results[k])}")


if __name__ == "__main__":
    main()
