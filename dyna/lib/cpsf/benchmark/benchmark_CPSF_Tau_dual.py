# Run as (examples):
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_Tau_dual --N 4 --M 256 --S 64 --batch 32 --K 4 --dtype c64 --device cpu --iters 50 --warmup 10
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_Tau_dual --N 4 --M 256 --S 64 --batch 32 --K 4 --dtype c64 --device cuda --iters 50 --warmup 10 --profile
#
# Notes:
# - Includes overhead of constructing the spectral lattice k (full cube |k|_inf<=K).
# - Shapes aligned exactly as required by delta_vec_d: expand to (B, M, N).
# - Output expected shape: (B, S).

# Results (reference):
#
# CPU (Ryzen 9 - 5950x):
#   > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_Tau_dual --N 4 --M 256 --S 64 --batch 32 --K 4 --dtype c64 --device cpu --iters 50 --warmup 10
#       Device=cpu, dtype=torch.complex64, N=4, M=256, S=64, B=32, K=4, warmup=10, iters=50, sigma_par=1.25, sigma_perp=0.55
#
#       === Tau_dual Benchmark ===
#       Avg time/iter: 1060.647 ms  (± 82.298 ms)
#       Throughput:    30 fields/s  (each has S=64)
#       k points (|k|_inf<=K): ~6,561
#       Process RSS:   2.85 GB
#
# GPU (RTX 4090):
#   > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_Tau_dual --N 4 --M 256 --S 64 --batch 32 --K 4 --dtype c64 --device cuda --iters 50 --warmup 10
#       Device=cuda, dtype=torch.complex64, N=4, M=256, S=64, B=32, K=4, warmup=10, iters=50, sigma_par=1.25, sigma_perp=0.55
#
#       === Tau_dual Benchmark ===
#       Avg time/iter: 21.567 ms  (± 7.158 ms)
#       Throughput:    1,484 fields/s  (each has S=64)
#       k points (|k|_inf<=K): ~6,561
#       CUDA peak mem (max):   2.42 GB
#       CUDA alloc Δ (max):    16.00 KB


import argparse, time, torch
from ..functional.core_math import Tau_dual


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


def _k_cube(K: int, N: int, device: torch.device) -> torch.Tensor:
    if K < 0:
        return torch.zeros(1, N, device=device, dtype=torch.int64)
    rng = torch.arange(-K, K + 1, device=device, dtype=torch.int64)
    grids = torch.meshgrid(*([rng] * N), indexing="ij")
    k = torch.stack(grids, dim=-1).reshape(-1, N)
    return k


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
        "--K", type=int, default=8, help="Spectral cube radius K (|k|_inf<=K)"
    )
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sigma_par", type=float, default=1.25)
    ap.add_argument("--sigma_perp", type=float, default=0.55)
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
    N, M, S, B, K = args.N, args.M, args.S, args.batch, args.K
    if N < 2:
        raise SystemExit("CPSF requires N >= 2.")
    if K is None or K < 0:
        raise SystemExit("--K must be integer >= 0")

    print(
        f"Device={dev.type}, dtype={CDTYPE}, N={N}, M={M}, S={S}, B={B}, "
        f"K={K}, warmup={args.warmup}, iters={args.iters}, "
        f"sigma_par={args.sigma_par}, sigma_perp={args.sigma_perp}"
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
    z_bm = z.unsqueeze(-2).expand(B, M, N)
    vec_d_bm = vec_d.unsqueeze(-2).expand(B, M, N)
    z_j_bm = z_j.unsqueeze(0).expand(B, M, N)
    vec_d_j_bm = vec_d_j.unsqueeze(0).expand(B, M, N)
    T_hat_b = T_hat_j.unsqueeze(0)
    alpha_b = alpha_j.unsqueeze(0)
    sp_b = sp.unsqueeze(0)
    sq_b = sq.unsqueeze(0)

    def _build_k():
        return _k_cube(int(K), int(N), dev)

    if args.verify_devices:
        k = _build_k()
        out = Tau_dual(
            z=z_bm,
            z_j=z_j_bm,
            vec_d=vec_d_bm,
            vec_d_j=vec_d_j_bm,
            T_hat_j=T_hat_b,
            alpha_j=alpha_b,
            sigma_par=sp_b,
            sigma_perp=sq_b,
            k=k,
            R_j=None,
        )
        print(
            f"verify: out.shape={tuple(out.shape)}, out.dtype={out.dtype}, out.device={out.device}, k.shape={tuple(k.shape)}"
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
        k = _build_k()
        _ = Tau_dual(
            z=z_bm,
            z_j=z_j_bm,
            vec_d=vec_d_bm,
            vec_d_j=vec_d_j_bm,
            T_hat_j=T_hat_b,
            alpha_j=alpha_b,
            sigma_par=sp_b,
            sigma_perp=sq_b,
            k=k,
            R_j=None,
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
                k = _build_k()
                _ = Tau_dual(
                    z=z_bm,
                    z_j=z_j_bm,
                    vec_d=vec_d_bm,
                    vec_d_j=vec_d_j_bm,
                    T_hat_j=T_hat_b,
                    alpha_j=alpha_b,
                    sigma_par=sp_b,
                    sigma_perp=sq_b,
                    k=k,
                    R_j=None,
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
            k = _build_k()
            out = Tau_dual(
                z=z_bm,
                z_j=z_j_bm,
                vec_d=vec_d_bm,
                vec_d_j=vec_d_j_bm,
                T_hat_j=T_hat_b,
                alpha_j=alpha_b,
                sigma_par=sp_b,
                sigma_perp=sq_b,
                k=k,
                R_j=None,
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
            k = _build_k()
            out = Tau_dual(
                z=z_bm,
                z_j=z_j_bm,
                vec_d=vec_d_bm,
                vec_d_j=vec_d_j_bm,
                T_hat_j=T_hat_b,
                alpha_j=alpha_b,
                sigma_par=sp_b,
                sigma_perp=sq_b,
                k=k,
                R_j=None,
            )
            _ = out.real.sum().item()
            dt = (time.perf_counter() - t0) * 1e3
        times.append(dt)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
    thr = B / (avg / 1e3)

    try:
        k_count = (2 * K + 1) ** N
    except OverflowError:
        k_count = float("inf")

    print("\n=== Tau_dual Benchmark ===")
    print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
    print(f"Throughput:    {thr:,.0f} fields/s  (each has S={S})")
    print(f"k points (|k|_inf<=K): ~{k_count:,}")
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
