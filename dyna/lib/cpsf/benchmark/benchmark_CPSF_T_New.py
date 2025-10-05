# Run examples:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_New --B 2048 --N 4 --M 64 --S 16 --dtype c64 --device cuda --iters 50 --warmup 10
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_New --B 256 --N 8 --M 128 --S 32 --dtype c128 --device cuda --iters 30 --warmup 10 --error_budget 1e-6 --q_order 9

import argparse, time, torch
from dyna.lib.cpsf.functional.t_new import T_New


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
        alloc_delta = max(0, mem1 - mem0)
        return dt_ms, peak, alloc_delta, out
    else:
        t0 = time.perf_counter()
        out = fn()
        dt_ms = (time.perf_counter() - t0) * 1e3
        return dt_ms, None, None, out


def _profile_block(dev: torch.device, steps: int, body):
    try:
        from torch.profiler import profile, ProfilerActivity, schedule
    except Exception:
        print("Profiler not available; skipping.")
        for _ in range(steps):
            body()
        return
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


def _rand_torus(
    B: int, N: int, dtype: torch.dtype, device: torch.device, seed: int
) -> torch.Tensor:
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    xr = torch.empty([B, N], device=device, dtype=REAL).uniform_(-0.5, 0.5)
    xi = torch.empty([B, N], device=device, dtype=REAL).uniform_(-0.5, 0.5)
    return (xr + 1j * xi).to(dtype)


def _rand_complex(
    B: int, N: int, dtype: torch.dtype, device: torch.device, seed: int
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    xr = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    xi = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    return (xr + 1j * xi).to(dtype)


def _rand_unit(
    B: int, N: int, dtype: torch.dtype, device: torch.device, seed: int
) -> torch.Tensor:
    v = _rand_complex(B, N, dtype, device, seed)
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    n = torch.where(n.real == 0, torch.ones_like(n), n)
    return v / n


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2048)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--error_budget", type=float, default=1.0e-5)
    ap.add_argument("--q_order", type=int, default=7)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--verify_devices", action="store_true")
    args = ap.parse_args()

    dev = _pick_device(args.device)
    try:
        torch.set_default_device(dev.type)
    except Exception:
        pass

    dtype = torch.complex64 if args.dtype == "c64" else torch.complex128
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    B, N, M, S = args.B, args.N, args.M, args.S
    if N < 2:
        raise SystemExit("CPSF requires N >= 2.")
    if M < 1:
        raise SystemExit("CPSF requires M >= 1.")
    print(
        f"Device={dev.type}, dtype={dtype}, B={B}, N={N}, M={M}, S={S}, "
        f"warmup={args.warmup}, iters={args.iters}, error_budget={args.error_budget:g}, q_order={args.q_order}"
    )

    z = _rand_torus(B, N, dtype, dev, seed=10)  # [B, N]
    z_j = _rand_torus(B * M, N, dtype, dev, seed=20).view(B, M, N)  # [B, M, N]
    vec_d = _rand_unit(B, N, dtype, dev, seed=30)  # [B, N]
    vec_d_j = _rand_unit(B * M, N, dtype, dev, seed=40).view(B, M, N)  # [B, M, N]
    T_hat_j = _rand_complex(B * M, S, dtype, dev, seed=50).view(B, M, S)  # [B, M, S]
    alpha_j = torch.empty([B, M], device=dev, dtype=REAL).uniform_(0.1, 2.0)  # [B, M]
    sigma_par = torch.empty([B, M], device=dev, dtype=REAL).uniform_(0.1, 1.5)  # [B, M]
    sigma_perp = torch.empty([B, M], device=dev, dtype=REAL).uniform_(
        0.1, 1.5
    )  # [B, M]

    if args.verify_devices:
        out = T_New(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            error_budget=float(args.error_budget),
            q_order=int(args.q_order),
        )
        print(
            f"verify: all inputs on {dev.type}, out.device={out.device}, out.shape={tuple(out.shape)}"
        )
        assert out.device == z.device and out.shape == (B, S)
        del out

    _sync(dev)
    for _ in range(args.warmup):
        _ = T_New(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            error_budget=float(args.error_budget),
            q_order=int(args.q_order),
        )
    _sync(dev)

    if args.profile:
        _profile_block(
            dev,
            steps=5,
            body=lambda: T_New(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha_j,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
                error_budget=float(args.error_budget),
                q_order=int(args.q_order),
            ),
        )

    times, peak_max, alloc_max = [], 0, 0
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(args.iters):
        dt_ms, peak, alloc, out = _time_block(
            dev,
            lambda: T_New(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha_j,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
                error_budget=float(args.error_budget),
                q_order=int(args.q_order),
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
    thr_terms = (B * M * S) / secs
    thr_points = B / secs

    print("\n=== T_New Benchmark ===")
    print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
    print(f"Throughput:    {thr_terms:,.0f} terms/s   (B*M*S)")
    print(f"Per-target:    {thr_points:,.0f} points/s  (B)")
    if dev.type == "cuda":
        print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
        print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")
    else:
        try:
            import psutil, os

            rss = psutil.Process(os.getpid()).memory_info().rss
            print(f"Process RSS:   {_fmt_bytes(rss)}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
