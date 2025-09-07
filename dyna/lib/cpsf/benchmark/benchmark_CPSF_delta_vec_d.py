# Run as (example):
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_delta_vec_d --N 16 --batch 65536 --dtype c64 --device cuda --iters 100 --warmup 10
# Options:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_delta_vec_d --profile --verify_devices --eps 1e-6

import argparse, time, torch
from ..functional.core_math import delta_vec_d


def _fmt_bytes(x: int) -> str:
    u = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    v = float(x)
    while v >= 1024 and i < len(u) - 1:
        v /= 1024
        i += 1
    return f"{v:.2f} {u[i]}"


def make_unit_batch(B, N, dtype, device, seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    xr = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    xi = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    v = (xr + 1j * xi).to(dtype)
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    # Avoid divide-by-zero (extremely unlikely)
    n = torch.where(n.real == 0, torch.ones_like(n), n)
    return v / n


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--eps", type=float, default=1.0e-6)
    ap.add_argument("--verify_devices", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    # Device pick
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("WARN: CUDA not available, falling back to CPU.")
            dev = torch.device("cpu")

    # Optional: default device so new tensors w/o explicit device land on dev
    try:
        torch.set_default_device(dev.type)
    except Exception:
        pass

    dtype = torch.complex64 if args.dtype == "c64" else torch.complex128
    REAL = torch.float32 if dtype == torch.complex64 else torch.float64
    B, N = args.batch, args.N
    print(
        f"Device={dev.type}, dtype={dtype}, batch={B}, N={N}, warmup={args.warmup}, iters={args.iters}, eps={args.eps}"
    )

    # Prepare inputs outside the timed loop
    d  = make_unit_batch(B, N, dtype, dev)   # (B, N)
    dj = make_unit_batch(B, N, dtype, dev)   # (B, N)

    if args.verify_devices:
        out = delta_vec_d(d, dj, args.eps)   # (B, N)
        print(f"in.device={d.device}, out.device={out.device}")
        assert (
            (dev.type == "cpu" and (not d.is_cuda) and (not out.is_cuda))
            or (dev.type == "cuda" and d.is_cuda and out.is_cuda)
        ), "Device mismatch!"
        # quick tangency sanity (not timed)
        ip = torch.sum(dj.conj() * out, dim=-1).abs().real.mean().item()
        print(f"avg |<dj, delta>|: {ip:.3e}")
        del out

    # Warmup
    if dev.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(args.warmup):
        _ = delta_vec_d(d, dj, args.eps)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Optional profiling (short)
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
                _ = delta_vec_d(d, dj, args.eps)
                prof.step()
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total" if dev.type == "cuda" else "cpu_time_total",
                row_limit=15,
            )
        )

    # Timed loop
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
            out = delta_vec_d(d, dj, args.eps)  # (B, N)
            # keep the work; avoid DCE; .item() syncs
            _ = out[..., 0].real.sum().item()
            end.record()
            torch.cuda.synchronize()
            dt = start.elapsed_time(end)  # ms
            mem1 = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            peak_bytes_max = max(peak_bytes_max, peak)
            alloc_delta_max = max(alloc_delta_max, max(0, mem1 - mem0))
        else:
            t0 = time.perf_counter()
            out = delta_vec_d(d, dj, args.eps)
            _ = out[..., 0].real.sum().item()
            dt = (time.perf_counter() - t0) * 1e3
        times.append(dt)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
    thr = B / (avg / 1e3)  # offsets/s

    print("\n=== delta_vec_d Benchmark ===")
    print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
    print(f"Throughput:    {thr:,.0f} offsets/s")
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
