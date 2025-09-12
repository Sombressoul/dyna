# Run as (examples):
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_classic_full --N 4 --M 256 --S 64 --batch 32 --dtype c64 --device cpu --iters 50 --warmup 10 --radius 3
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_classic_full --N 4 --M 256 --S 64 --batch 32 --dtype c64 --device cuda --iters 50 --warmup 10 --radius 3 --profile
#
# Notes:
# - Includes overhead of constructing CPSFPeriodization(FULL, max_radius=R) and streaming shells via iter_offsets.
# - Matches the calling contract used in tests: T_classic_full(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp, offsets_iterator, ...).  # see tests

# Results (reference):
#
# CPU (Ryzen 9 - 5950x):
#   > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_classic_full --N 4 --M 256 --S 64 --batch 32 --dtype c64 --device cpu --iters 50 --warmup 10 --radius 3
#       Device=cpu, dtype=torch.complex64, N=4, M=256, S=64, B=32, R=3, warmup=10, iters=50, sigma_par=1.25, sigma_perp=0.55, tol_abs=None, tol_rel=None, consecutive_below=1, cache_active=False
#
#       === T_classic_full Benchmark ===
#       Avg time/iter: 1823.058 ms  (± 106.329 ms)
#       Throughput:    18 fields/s  (each has S=64)
#       Offsets total (W<=R): ~2,401 points (theoretical (2R+1)^N)
#       Process RSS:   3.16 GB
#
# GPU (RTX 4090):
#   > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_classic_full --N 4 --M 256 --S 64 --batch 32 --dtype c64 --device cuda --iters 50 --warmup 10 --radius 3
#       Device=cuda, dtype=torch.complex64, N=4, M=256, S=64, B=32, R=3, warmup=10, iters=50, sigma_par=1.25, sigma_perp=0.55, tol_abs=None, tol_rel=None, consecutive_below=1, cache_active=False
#
#       === T_classic_full Benchmark ===
#       Avg time/iter: 91.012 ms  (± 14.145 ms)
#       Throughput:    352 fields/s  (each has S=64)
#       Offsets total (W<=R): ~2,401 points (theoretical (2R+1)^N)
#       CUDA peak mem (max):   4.13 GB
#       CUDA alloc Δ (max):    16.00 KB

import argparse, time, torch
from ..functional.core_math import T_classic_full
from ..periodization import CPSFPeriodization
from ..structures import CPSFPeriodizationKind


def _fmt_bytes(x: int) -> str:
    u = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    v = float(x)
    while v >= 1024 and i < len(u) - 1:
        v /= 1024
        i += 1
    return f"{v:.2f} {u[i]}"


def _real_dtype_of(cdtype: torch.dtype) -> torch.dtype:
    return torch.float32 if cdtype == torch.complex64 else torch.float64


def _make_unit_batch(
    B: int, N: int, dtype: torch.dtype, device: torch.device, seed: int
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    REAL = _real_dtype_of(dtype)
    xr = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    xi = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    v = (xr + 1j * xi).to(dtype)
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    n = torch.where(n.real == 0, torch.ones_like(n), n)
    return v / n


def _make_complex(shape, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    REAL = _real_dtype_of(dtype)
    xr = torch.randn(*shape, generator=g, device=device, dtype=REAL)
    xi = torch.randn(*shape, generator=g, device=device, dtype=REAL)
    return (xr + 1j * xi).to(dtype)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8, help="Torus dimension (>=2)")
    ap.add_argument("--M", type=int, default=2048, help="Number of contributions")
    ap.add_argument(
        "--S", type=int, default=256, help="Spectral dimension per contribution"
    )
    ap.add_argument("--batch", type=int, default=64, help="Batch of query rays (B)")
    ap.add_argument(
        "--radius",
        type=int,
        default=5,
        help="FULL periodization max radius R (sum shells W=0..R)",
    )
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sigma_par", type=float, default=1.25)
    ap.add_argument("--sigma_perp", type=float, default=0.55)
    ap.add_argument("--q_max", type=float, default=None)
    ap.add_argument("--tol_abs", type=float, default=None)
    ap.add_argument("--tol_rel", type=float, default=None)
    ap.add_argument("--consecutive_below", type=int, default=1)
    ap.add_argument(
        "--cache_active",
        action="store_true",
        help="Enable LRU cache inside CPSFPeriodization object",
    )
    ap.add_argument("--cache_limit", type=int, default=32)
    ap.add_argument("--cache_soft_limit_bytes", type=int, default=128 * 1024 * 1024)
    ap.add_argument("--verify_devices", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("WARN: CUDA not available, falling back to CPU.")
            dev = torch.device("cpu")

    try:
        torch.set_default_device(dev.type)
    except Exception:
        pass

    CDTYPE = torch.complex64 if args.dtype == "c64" else torch.complex128
    REAL = _real_dtype_of(CDTYPE)
    N, M, S, B = args.N, args.M, args.S, args.batch
    if N < 2:
        raise SystemExit("CPSF requires N >= 2.")
    if args.radius is None or args.radius < 0:
        raise SystemExit("--radius must be non-negative integer")

    print(
        f"Device={dev.type}, dtype={CDTYPE}, N={N}, M={M}, S={S}, B={B}, "
        f"R={args.radius}, warmup={args.warmup}, iters={args.iters}, "
        f"sigma_par={args.sigma_par}, sigma_perp={args.sigma_perp}, "
        f"tol_abs={args.tol_abs}, tol_rel={args.tol_rel}, consecutive_below={args.consecutive_below}, "
        f"cache_active={'True' if args.cache_active else 'False'}"
    )

    z = _make_complex((B, N), CDTYPE, dev, seed=args.seed + 1)
    vec_d = _make_unit_batch(B, N, CDTYPE, dev, seed=args.seed + 2)
    z_j = _make_complex((M, N), CDTYPE, dev, seed=args.seed + 3)
    vec_d_j = _make_unit_batch(M, N, CDTYPE, dev, seed=args.seed + 4)
    T_hat_j = _make_complex((M, S), CDTYPE, dev, seed=args.seed + 5)
    g_alpha = torch.Generator(device=dev).manual_seed(args.seed + 6)
    alpha_j = torch.rand(M, generator=g_alpha, device=dev, dtype=REAL)
    sp = torch.full((M,), float(args.sigma_par), dtype=REAL, device=dev)
    sq = torch.full((M,), float(args.sigma_perp), dtype=REAL, device=dev)
    z_b = z.unsqueeze(-2).expand(B, M, N)
    vec_d_b = vec_d.unsqueeze(-2).expand(B, M, N)
    z_j_b = z_j.unsqueeze(0).expand(B, M, N)
    vec_d_j_b = vec_d_j.unsqueeze(0).expand(B, M, N)
    T_hat_b = T_hat_j.unsqueeze(0)
    alpha_b = alpha_j.unsqueeze(0)
    sp_b = sp.unsqueeze(0)
    sq_b = sq.unsqueeze(0)

    if args.verify_devices:
        per0 = CPSFPeriodization(
            kind=CPSFPeriodizationKind.FULL,
            max_radius=int(args.radius),
            cache_active=bool(args.cache_active),
            cache_limit=int(args.cache_limit),
            cache_soft_limit_bytes=int(args.cache_soft_limit_bytes),
        )
        out = T_classic_full(
            z=z_b,
            z_j=z_j_b,
            vec_d=vec_d_b,
            vec_d_j=vec_d_j_b,
            T_hat_j=T_hat_b,
            alpha_j=alpha_b,
            sigma_par=sp_b,
            sigma_perp=sq_b,
            offsets_iterator=per0.iter_offsets,
            R_j=None,
            q_max=args.q_max,
            tol_abs=args.tol_abs,
            tol_rel=args.tol_rel,
            consecutive_below=int(args.consecutive_below),
        )
        print(
            f"verify: out.shape={tuple(out.shape)}, out.dtype={out.dtype}, out.device={out.device}"
        )
        assert out.shape == (B, S), f"Expected (B,S), got {tuple(out.shape)}"
        assert (
            out.dtype == CDTYPE and out.device.type == dev.type
        ), "dtype/device mismatch"
        assert (
            torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()
        ), "non-finite output"
        del out

    if dev.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(args.warmup):
        per = CPSFPeriodization(
            kind=CPSFPeriodizationKind.FULL,
            max_radius=int(args.radius),
            cache_active=bool(args.cache_active),
            cache_limit=int(args.cache_limit),
            cache_soft_limit_bytes=int(args.cache_soft_limit_bytes),
        )
        _ = T_classic_full(
            z=z_b,
            z_j=z_j_b,
            vec_d=vec_d_b,
            vec_d_j=vec_d_j_b,
            T_hat_j=T_hat_b,
            alpha_j=alpha_b,
            sigma_par=sp_b,
            sigma_perp=sq_b,
            offsets_iterator=per.iter_offsets,
            R_j=None,
            q_max=args.q_max,
            tol_abs=args.tol_abs,
            tol_rel=args.tol_rel,
            consecutive_below=int(args.consecutive_below),
        )
    if dev.type == "cuda":
        torch.cuda.synchronize()

    if args.profile:
        from torch.profiler import profile, ProfilerActivity, schedule

        activities = [ProfilerActivity.CPU] + (
            [ProfilerActivity.CUDA] if dev.type == "cuda" else []
        )
        sch = schedule(wait=1, warmup=1, active=4, repeat=1)
        with profile(
            activities=activities,
            schedule=sch,
            record_shapes=False,
            profile_memory=True,
        ) as prof:
            for _ in range(6):
                per = CPSFPeriodization(
                    kind=CPSFPeriodizationKind.FULL,
                    max_radius=int(args.radius),
                    cache_active=bool(args.cache_active),
                    cache_limit=int(args.cache_limit),
                    cache_soft_limit_bytes=int(args.cache_soft_limit_bytes),
                )
                _ = T_classic_full(
                    z=z_b,
                    z_j=z_j_b,
                    vec_d=vec_d_b,
                    vec_d_j=vec_d_j_b,
                    T_hat_j=T_hat_b,
                    alpha_j=alpha_b,
                    sigma_par=sp_b,
                    sigma_perp=sq_b,
                    offsets_iterator=per.iter_offsets,
                    R_j=None,
                    q_max=args.q_max,
                    tol_abs=args.tol_abs,
                    tol_rel=args.tol_rel,
                    consecutive_below=int(args.consecutive_below),
                )
                prof.step()
        print(
            prof.key_averages().table(
                sort_by=(
                    "self_cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
                ),
                row_limit=15,
            )
        )

    times = []
    peak_bytes_max = 0
    alloc_delta_max = 0
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(args.iters):
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            mem0 = torch.cuda.memory_allocated()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            per = CPSFPeriodization(
                kind=CPSFPeriodizationKind.FULL,
                max_radius=int(args.radius),
                cache_active=bool(args.cache_active),
                cache_limit=int(args.cache_limit),
                cache_soft_limit_bytes=int(args.cache_soft_limit_bytes),
            )
            out = T_classic_full(
                z=z_b,
                z_j=z_j_b,
                vec_d=vec_d_b,
                vec_d_j=vec_d_j_b,
                T_hat_j=T_hat_b,
                alpha_j=alpha_b,
                sigma_par=sp_b,
                sigma_perp=sq_b,
                offsets_iterator=per.iter_offsets,
                R_j=None,
                q_max=args.q_max,
                tol_abs=args.tol_abs,
                tol_rel=args.tol_rel,
                consecutive_below=int(args.consecutive_below),
            )
            _ = out.real.sum().item()
            end.record()
            torch.cuda.synchronize()
            dt = start.elapsed_time(end)
            mem1 = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            peak_bytes_max = max(peak_bytes_max, peak)
            alloc_delta_max = max(alloc_delta_max, max(0, mem1 - mem0))
        else:
            t0 = time.perf_counter()
            per = CPSFPeriodization(
                kind=CPSFPeriodizationKind.FULL,
                max_radius=int(args.radius),
                cache_active=bool(args.cache_active),
                cache_limit=int(args.cache_limit),
                cache_soft_limit_bytes=int(args.cache_soft_limit_bytes),
            )
            out = T_classic_full(
                z=z_b,
                z_j=z_j_b,
                vec_d=vec_d_b,
                vec_d_j=vec_d_j_b,
                T_hat_j=T_hat_b,
                alpha_j=alpha_b,
                sigma_par=sp_b,
                sigma_perp=sq_b,
                offsets_iterator=per.iter_offsets,
                R_j=None,
                q_max=args.q_max,
                tol_abs=args.tol_abs,
                tol_rel=args.tol_rel,
                consecutive_below=int(args.consecutive_below),
            )
            _ = out.real.sum().item()
            dt = (time.perf_counter() - t0) * 1e3
        times.append(dt)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
    thr = B / (avg / 1e3)
    R = int(args.radius)
    try:
        offset_count = (2 * R + 1) ** N
    except OverflowError:
        offset_count = float("inf")

    print("\n=== T_classic_full Benchmark ===")
    print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
    print(f"Throughput:    {thr:,.0f} fields/s  (each has S={S})")
    print(f"Offsets total (W<=R): ~{offset_count:,} points (theoretical (2R+1)^N)")
    if dev.type == "cuda":
        print(f"CUDA peak mem (max):   {_fmt_bytes(peak_bytes_max)}")
        print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_delta_max)}")
    else:
        try:
            import psutil, os

            rss = psutil.Process(os.getpid()).memory_info().rss
            print(f"Process RSS:   {_fmt_bytes(rss)}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
