# Run examples:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_PD_full_vs_T_classic_full --N 4 --M 8 --S 16 --W 2 --batch 8 --dtype_z c64 --dtype_T c64 --device cuda --iters 25 --warmup 5 --target_per_pack 4096
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_PD_full_vs_T_classic_full --N 4 --M 4 --S 16 --W 3 --batch 2 --dtype_z c64 --dtype_T c64 --device cuda --iters 25 --warmup 5 --target_per_pack 2048
#
# Notes:
# - Compares FULL/streaming modes: T_classic_full() vs T_PD_full() on equally-packed index sets.
# - "Packs" for classic are positional lattice shifts n in Z^{2N}; for PD they are dual modes k in Z^{2N}.
# - Throughput is reported as eta-terms/s = (B * M * O_total) / (avg_time_seconds),
#   where O_total = sum_k |pack_k|.
# - By default early stopping is DISABLED (tol_abs=None, tol_rel=None). You can enable it via flags.

import argparse, time, torch
from torch.profiler import profile, ProfilerActivity, schedule

from dyna.lib.cpsf.functional.core_math import T_classic_full
from dyna.lib.cpsf.functional.t_pd import T_PD_full
from dyna.lib.cpsf.periodization import CPSFPeriodization


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


def _time_block(dev: torch.device, fn):
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem0 = torch.cuda.memory_allocated()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        dt_ms = start.elapsed_time(end)
        mem1 = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        alloc = max(0, mem1 - mem0)
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
            sort_by=("self_cuda_time_total" if dev.type == "cuda" else "cpu_time_total"),
            row_limit=15,
        )
    )


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
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--W", type=int, default=2, help="Linf window radius; packs cover |n|<=W or |k|<=W")
    ap.add_argument("--batch", type=int, default=512, help="Number of target points B")
    ap.add_argument("--dtype_z", choices=["c64", "c128"], default="c64")
    ap.add_argument("--dtype_T", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    # Pack controls
    ap.add_argument("--target_per_pack", type=int, default=4096)
    ap.add_argument("--sorted", action="store_true", help="Sort offsets inside each pack")
    # PD-only knobs
    ap.add_argument("--t", type=float, default=1.0, help="Poisson/Ewald scale t>0 for PD")
    # Early stopping (applies to both; default disabled)
    ap.add_argument("--tol_abs", type=float, default=None)
    ap.add_argument("--tol_rel", type=float, default=None)
    ap.add_argument("--consecutive_below", type=int, default=1)
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
    if N < 2:
        raise SystemExit("CPSF requires N >= 2.")
    if W < 0:
        raise SystemExit("W must be >= 0.")
    if args.t <= 0.0:
        raise SystemExit("t must be > 0 for PD.")

    print(
        f"Device={dev.type}, dtype_z={dtype_z}, dtype_T={dtype_T}, "
        f"B={B}, M={M}, S={S}, N={N}, W={W}, "
        f"iters={args.iters}, warmup={args.warmup}, t={args.t}, "
        f"target_per_pack={args.target_per_pack}, sorted={args.sorted}, "
        f"tol_abs={args.tol_abs}, tol_rel={args.tol_rel}, consecutive_below={args.consecutive_below}"
    )

    gen = CPSFPeriodization()
    # Build identical pack layout for both methods
    packs = list(
        gen.iter_packed(
            N=N,
            target_points_per_pack=max(1, args.target_per_pack),
            start_radius=0,
            max_radius=W,
            device=dev,
            sorted=args.sorted,
        )
    )
    O_total = sum(int(off.shape[0]) for _, _, off in packs)
    print(f"total packed size O_total={O_total:,} rows (used as n-packs for classic and k-packs for PD)")

    # Inputs (outside timing)
    z = _make_cplx(B, N, dtype=dtype_z, device=dev, seed=10, unitize=False)  # [B,N]
    z_j = _make_cplx(B * M, N, dtype=dtype_z, device=dev, seed=20, unitize=False).view(B, M, N)  # [B,M,N]
    vec_d = _make_cplx(B, N, dtype=dtype_z, device=dev, seed=30, unitize=True)  # [B,N]
    vec_d_j = _make_cplx(B * M, N, dtype=dtype_z, device=dev, seed=40, unitize=True).view(B, M, N)  # [B,M,N]

    alpha_j = 0.2 + 1.3 * torch.rand(B, M, device=dev, dtype=REAL_T)                # [B,M] real
    sigma_par = torch.full((B, M), 0.9, device=dev, dtype=REAL_z)                   # [B,M] real
    sigma_perp = torch.full((B, M), 0.5, device=dev, dtype=REAL_z)                   # [B,M] real

    Tr = torch.randn(B, M, S, device=dev, dtype=REAL_T)
    Ti = torch.randn(B, M, S, device=dev, dtype=REAL_T)
    T_hat_j = torch.complex(Tr, Ti).to(dtype_T)                                     # [B,M,S]

    # Warmups
    def warmup_classic():
        for _ in range(args.warmup):
            _ = T_classic_full(
                z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j,
                sigma_par, sigma_perp,
                packs=packs,
                R_j=None,
                q_max=None,
                tol_abs=args.tol_abs,
                tol_rel=args.tol_rel,
                consecutive_below=args.consecutive_below,
            )
        _sync(dev)

    def warmup_pd():
        for _ in range(args.warmup):
            _ = T_PD_full(
                z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j, T_hat_j=T_hat_j, alpha_j=alpha_j,
                sigma_par=sigma_par, sigma_perp=sigma_perp,
                packs=packs, R_j=None, t=args.t,
                tol_abs=args.tol_abs, tol_rel=args.tol_rel, consecutive_below=args.consecutive_below,
            )
        _sync(dev)

    # Optional profiling (single run)
    def profile_classic():
        _profile_block(
            dev, steps=5,
            body=lambda: T_classic_full(
                z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j,
                sigma_par, sigma_perp, packs=packs,
                R_j=None, q_max=None,
                tol_abs=args.tol_abs, tol_rel=args.tol_rel, consecutive_below=args.consecutive_below,
            ),
        )

    def profile_pd():
        _profile_block(
            dev, steps=5,
            body=lambda: T_PD_full(
                z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j, T_hat_j=T_hat_j, alpha_j=alpha_j,
                sigma_par=sigma_par, sigma_perp=sigma_perp,
                packs=packs, R_j=None, t=args.t,
                tol_abs=args.tol_abs, tol_rel=args.tol_rel, consecutive_below=args.consecutive_below,
            ),
        )

    # Bench bodies
    def bench_classic():
        warmup_classic()
        # profile_classic()  # uncomment if needed
        times, peak_max, alloc_max = [], 0, 0
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(args.iters):
            dt_ms, peak, alloc, out = _time_block(
                dev,
                lambda: T_classic_full(
                    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j,
                    sigma_par, sigma_perp, packs=packs,
                    R_j=None, q_max=None,
                    tol_abs=args.tol_abs, tol_rel=args.tol_rel, consecutive_below=args.consecutive_below,
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
        thr_terms = (B * M * O_total) / secs
        thr_points = B / secs
        print("\n=== T_classic_full ===")
        print(f"O_total={O_total:,} offsets (in packs), B={B:,}, M={M:,}, S={S:,}")
        print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
        print(f"Throughput:    {thr_terms:,.0f} eta-terms/s   (B*M*O_total)")
        print(f"Per-target:    {thr_points:,.0f} points/s     (B)")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")
        return avg, std, thr_terms, thr_points, out

    def bench_pd():
        warmup_pd()
        # profile_pd()  # uncomment if needed
        times, peak_max, alloc_max = [], 0, 0
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(args.iters):
            dt_ms, peak, alloc, out = _time_block(
                dev,
                lambda: T_PD_full(
                    z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j, T_hat_j=T_hat_j, alpha_j=alpha_j,
                    sigma_par=sigma_par, sigma_perp=sigma_perp,
                    packs=packs, R_j=None, t=args.t,
                    tol_abs=args.tol_abs, tol_rel=args.tol_rel, consecutive_below=args.consecutive_below,
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
        thr_terms = (B * M * O_total) / secs
        thr_points = B / secs
        print("\n=== T_PD_full (dual positional modes, streaming) ===")
        print(f"O_total={O_total:,} k-modes (in packs), B={B:,}, M={M:,}, S={S:,}, t={args.t}")
        print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
        print(f"Throughput:    {thr_terms:,.0f} eta-terms/s   (B*M*O_total)")
        print(f"Per-target:    {thr_points:,.0f} points/s     (B)")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")
        return avg, std, thr_terms, thr_points, out

    # Execute both and summarize
    res_classic = bench_classic()
    res_pd = bench_pd()

    # quick numeric sanity (NOT equality): report norms
    out_c = res_classic[-1]
    out_p = res_pd[-1]
    with torch.no_grad():
        n_c = out_c.abs().mean().item()
        n_p = out_p.abs().mean().item()
        diff = (out_c - out_p).abs().mean().item()
    print("\n=== Quick numeric sanity (not an equality test) ===")
    print(f"mean|classic|={n_c:.4e}, mean|PD|={n_p:.4e}, mean|classic-PD|={diff:.4e}")

    # speedup summary
    t_c = res_classic[0]
    t_p = res_pd[0]
    speedup = t_c / max(1e-9, t_p)
    print("\n=== Summary ===")
    print(f"Classic avg: {t_c:.3f} ms | PD avg: {t_p:.3f} ms | speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
