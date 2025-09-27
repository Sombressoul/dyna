# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_Omega_vs_T_classic_window_ZeroFrame --N 4 --M 8 --S 16 --batch 8 --dtype_z c64 --dtype_T c64 --device cuda --iters 25 --warmup 5
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_Omega_vs_T_classic_window_ZeroFrame --N 32 --M 256 --S 16 --batch 256 --dtype_z c64 --dtype_T c64 --device cuda --iters 25 --warmup 5

import argparse, time, torch

from dyna.lib.cpsf.periodization import CPSFPeriodization
from dyna.lib.cpsf.functional.core_math import T_classic_window
from dyna.lib.cpsf.functional.t_omega import T_Omega


def TEST_ZERO_FRAME(*args, **kwargs) -> torch.Tensor:
    return T_Omega(return_components="zero", *args, **kwargs)


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
        z = z / torch.where(n == 0, torch.ones_like(n), n)
    return z


def _dtype_name(dt: torch.dtype) -> str:
    return "torch.complex64" if dt == torch.complex64 else "torch.complex128"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512, help="B")
    ap.add_argument("--dtype_z", choices=["c64", "c128"], default="c64")
    ap.add_argument("--dtype_T", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sp_mean", type=float, default=0.9, help="sigma_par mean")
    ap.add_argument("--sq_mean", type=float, default=0.5, help="sigma_perp mean")
    ap.add_argument(
        "--rtol", type=float, default=None, help="rtol for allclose (default by dtype)"
    )
    ap.add_argument(
        "--atol", type=float, default=None, help="atol for allclose (default by dtype)"
    )
    args = ap.parse_args()

    dev = _pick_device(args.device)

    dtype_z = torch.complex64 if args.dtype_z == "c64" else torch.complex128
    dtype_T = torch.complex64 if args.dtype_T == "c64" else torch.complex128
    REAL_z = torch.float32 if dtype_z == torch.complex64 else torch.float64
    REAL_T = torch.float32 if dtype_T == torch.complex64 else torch.float64

    B, N, M, S = args.batch, args.N, args.M, args.S
    if N < 2:
        raise SystemExit("CPSF requires N >= 2.")

    print(
        f"Device={dev.type}, dtype_z={_dtype_name(dtype_z)}, dtype_T={_dtype_name(dtype_T)}, "
        f"B={B}, M={M}, S={S}, N={N}, iters={args.iters}, warmup={args.warmup}"
    )
    print(f"sigmas: sigma_perp(mean)={args.sq_mean}, sigma_par(mean)={args.sp_mean}")

    gen = CPSFPeriodization()
    offsets = gen.window(N=N, W=0, device=dev, sorted=False)
    O = int(offsets.shape[0])
    print(f"window size (W=0) O={O} rows")

    z = _make_cplx(B, N, dtype=dtype_z, device=dev, seed=10, unitize=False)
    z_j = _make_cplx(B * M, N, dtype=dtype_z, device=dev, seed=20, unitize=False).view(
        B, M, N
    )
    vec_d = _make_cplx(B, N, dtype=dtype_z, device=dev, seed=30, unitize=True)
    vec_d_j = _make_cplx(
        B * M, N, dtype=dtype_z, device=dev, seed=40, unitize=True
    ).view(B, M, N)

    alpha_j = 0.2 + 1.3 * torch.rand(B, M, device=dev, dtype=REAL_T)
    sigma_perp = torch.full((B, M), float(args.sq_mean), device=dev, dtype=REAL_z)
    sigma_par = torch.full((B, M), float(args.sp_mean), device=dev, dtype=REAL_z)

    Tr = torch.randn(B, M, S, device=dev, dtype=REAL_T)
    Ti = torch.randn(B, M, S, device=dev, dtype=REAL_T)
    T_hat_j = torch.complex(Tr, Ti).to(dtype_T)

    # Тёплый старт
    def warmup_classic():
        for _ in range(args.warmup):
            _ = T_classic_window(
                z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp, offsets
            )
        _sync(dev)

    def warmup_omega():
        for _ in range(args.warmup):
            _ = TEST_ZERO_FRAME(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha_j,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
            )
        _sync(dev)

    def bench_classic():
        warmup_classic()
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
                    offsets,
                ),
            )
            _ = out.real.sum().item()
            times.append(dt_ms)
            if peak is not None:
                peak_max = max(peak_max, peak)
            if alloc is not None:
                alloc_max = max(alloc_max, alloc)
        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
        secs = avg / 1e3
        thr_terms = (B * M * O) / max(1e-12, secs)
        thr_points = B / max(1e-12, secs)
        print("\n=== T_classic_window (W=0) ===")
        print(f"O={O} offsets, B={B:,}, M={M:,}, S={S:,}")
        print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
        print(f"Throughput:    {thr_terms:,.0f} terms/s    (B*M*O)")
        print(f"Per-target:    {thr_points:,.0f} points/s  (B)")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")
        return avg, std, thr_terms, thr_points, out

    def bench_omega():
        warmup_omega()
        times, peak_max, alloc_max = [], 0, 0
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(args.iters):
            dt_ms, peak, alloc, out = _time_block(
                dev,
                lambda: TEST_ZERO_FRAME(
                    z=z,
                    z_j=z_j,
                    vec_d=vec_d,
                    vec_d_j=vec_d_j,
                    T_hat_j=T_hat_j,
                    alpha_j=alpha_j,
                    sigma_par=sigma_par,
                    sigma_perp=sigma_perp,
                ),
            )
            _ = out.real.sum().item()
            times.append(dt_ms)
            if peak is not None:
                peak_max = max(peak_max, peak)
            if alloc is not None:
                alloc_max = max(alloc_max, alloc)
        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
        secs = avg / 1e3
        thr_terms = (B * M) / max(1e-12, secs)
        thr_points = B / max(1e-12, secs)
        print("\n=== T_Omega (n=0 only) ===")
        print(f"O=1 term, B={B:,}, M={M:,}, S={S:,}")
        print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
        print(f"Throughput:    {thr_terms:,.0f} terms/s    (B*M)")
        print(f"Per-target:    {thr_points:,.0f} points/s  (B)")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")
        return avg, std, thr_terms, thr_points, out

    # RUN
    res_classic = bench_classic()
    res_omega = bench_omega()

    # NUMERIC CHECK
    out_c = res_classic[-1].to(dtype_T)
    out_o = res_omega[-1].to(dtype_T)

    abs_diff = (out_c - out_o).abs()
    rel_diff = abs_diff / out_c.abs().clamp_min(1e-32)
    amax = abs_diff.max().item()
    amean = abs_diff.mean().item()
    rmax = rel_diff.max().item()
    rmean = rel_diff.mean().item()
    default_rtol = 1e-5 if dtype_T == torch.complex64 else 1e-12
    default_atol = 1e-6 if dtype_T == torch.complex64 else 1e-12
    rtol = default_rtol if args.rtol is None else float(args.rtol)
    atol = default_atol if args.atol is None else float(args.atol)
    ok = torch.allclose(out_c, out_o, rtol=rtol, atol=atol)

    print("\n=== Numeric equivalence: T_classic_window(W=0) vs T_Omega (zero) ===")
    print(f"Abs diff: max={amax:.3e}, mean={amean:.3e}")
    print(f"Rel diff: max={rmax:.3e}, mean={rmean:.3e}")
    print(f"allclose(rtol={rtol}, atol={atol}) -> {ok}")

    # SUMMARY
    t_c = res_classic[0]
    t_o = res_omega[0]
    speedup = t_c / max(1e-9, t_o)
    print("\n=== Summary ===")
    print(
        f"Classic (W=0) avg: {t_c:.3f} ms | Omega (n=0) avg: {t_o:.3f} ms | speedup: {speedup:.2f}×"
    )


if __name__ == "__main__":
    main()
