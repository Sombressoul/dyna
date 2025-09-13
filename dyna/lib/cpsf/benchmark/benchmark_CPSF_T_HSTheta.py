# dyna/lib/cpsf/benchmark/benchmark_CPSF_T_HSTheta.py
# Run examples:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_HSTheta --N 256 --M 256 --S 128 --batch 128 --dtype c64 --device cuda --iters 50 --warmup 10 --n_chunk 256 --m_chunk 256 --quad_nodes 7

import argparse, time, math, torch

from ..functional.t_hs_theta import T_HSTheta


def _fmt_bytes(
    x: int,
) -> str:
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
    B: int,
    N: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    REAL = _real_dtype_of(dtype)
    xr = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    xi = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    v = (xr + 1j * xi).to(dtype)
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    n = torch.where(n.real == 0, torch.ones_like(n), n)
    return v / n


def _make_complex(
    shape: tuple[int],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
):
    g = torch.Generator(device=device).manual_seed(seed)
    REAL = _real_dtype_of(dtype)
    xr = torch.randn(*shape, generator=g, device=device, dtype=REAL)
    xi = torch.randn(*shape, generator=g, device=device, dtype=REAL)
    return (xr + 1j * xi).to(dtype)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256, help="Torus dimension (>=2)")
    ap.add_argument("--M", type=int, default=256, help="Number of contributions")
    ap.add_argument(
        "--S", type=int, default=128, help="Spectral dimension per contribution"
    )
    ap.add_argument("--batch", type=int, default=128, help="Batch of query rays (B)")
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    # HS-Theta controls
    ap.add_argument("--quad_nodes", type=int, default=7)
    ap.add_argument("--eps_total", type=float, default=1.0e-3)
    ap.add_argument("--n_chunk", type=int, default=256)
    ap.add_argument("--m_chunk", type=int, default=256)
    # misc
    ap.add_argument("--verify_devices", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Device pick
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("WARN: CUDA not available, falling back to CPU.")
            dev = torch.device("cpu")

    # Optional: let new tensors without explicit device land on dev
    try:
        torch.set_default_device(dev.type)
    except Exception:
        pass

    CDTYPE = torch.complex64 if args.dtype == "c64" else torch.complex128
    REAL = _real_dtype_of(CDTYPE)
    N, M, S, B = args.N, args.M, args.S, args.batch
    if N < 2:
        raise SystemExit("CPSF requires N >= 2.")
    if min(N, M, S, B) < 1:
        raise SystemExit("Invalid sizes (N,M,S,B must be >=1).")

    print(
        f"Device={dev.type}, dtype={CDTYPE}, N={N}, M={M}, S={S}, B={B}, "
        f"warmup={args.warmup}, iters={args.iters}"
    )
    print(
        f"HS-Theta: quad_nodes={args.quad_nodes}, eps_total={args.eps_total}, "
        f"n_chunk={args.n_chunk}, m_chunk={args.m_chunk}"
    )

    # --------- inputs generation (outside timed loop) ---------
    z = _make_complex((B, N), CDTYPE, dev, seed=args.seed + 1)
    vec_d = _make_unit_batch(B, N, CDTYPE, dev, seed=args.seed + 2)
    z_j = _make_complex((M, N), CDTYPE, dev, seed=args.seed + 3)
    vec_d_j = _make_unit_batch(M, N, CDTYPE, dev, seed=args.seed + 4)
    T_hat_j = _make_complex((M, S), CDTYPE, dev, seed=args.seed + 5)

    g_alpha = torch.Generator(device=dev).manual_seed(args.seed + 6)
    alpha_j = torch.rand(M, generator=g_alpha, device=dev, dtype=REAL)

    # mildly anisotropic, positive, with a gap sp > sq
    sp = torch.empty(M, device=dev, dtype=REAL)
    sq = torch.empty(M, device=dev, dtype=REAL)
    g_sig = torch.Generator(device=dev).manual_seed(args.seed + 7)
    sq.uniform_(0.4, 1.2, generator=g_sig)
    sp.uniform_(1.0, 2.0, generator=g_sig)
    sp = torch.maximum(sp, sq + 1e-3)

    if args.verify_devices:
        out = T_HSTheta(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sp,
            sigma_perp=sq,
            quad_nodes=args.quad_nodes,
            eps_total=args.eps_total,
            n_chunk=args.n_chunk,
            m_chunk=args.m_chunk,
            dtype_override=CDTYPE,
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

    # --------- warmup ---------
    if dev.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(args.warmup):
        _ = T_HSTheta(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sp,
            sigma_perp=sq,
            quad_nodes=args.quad_nodes,
            eps_total=args.eps_total,
            n_chunk=args.n_chunk,
            m_chunk=args.m_chunk,
            dtype_override=CDTYPE,
        )
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # --------- optional short profiling ---------
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
                _ = T_HSTheta(
                    z=z,
                    z_j=z_j,
                    vec_d=vec_d,
                    vec_d_j=vec_d_j,
                    T_hat_j=T_hat_j,
                    alpha_j=alpha_j,
                    sigma_par=sp,
                    sigma_perp=sq,
                    quad_nodes=args.quad_nodes,
                    eps_total=args.eps_total,
                    n_chunk=args.n_chunk,
                    m_chunk=args.m_chunk,
                    dtype_override=CDTYPE,
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

    # --------- timed loop ---------
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
            out = T_HSTheta(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha_j,
                sigma_par=sp,
                sigma_perp=sq,
                quad_nodes=args.quad_nodes,
                eps_total=args.eps_total,
                n_chunk=args.n_chunk,
                m_chunk=args.m_chunk,
                dtype_override=CDTYPE,
            )
            # prevent DCE; .item() syncs
            _ = out.real.sum().item()
            end.record()
            torch.cuda.synchronize()
            dt = start.elapsed_time(end)  # ms
            mem1 = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            peak_bytes_max = max(peak_bytes_max, peak)
            alloc_delta_max = max(alloc_delta_max, max(0, mem1 - mem0))
        else:
            t0 = time.perf_counter()
            out = T_HSTheta(
                z=z,
                z_j=z_j,
                vec_d=vec_d,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha_j,
                sigma_par=sp,
                sigma_perp=sq,
                quad_nodes=args.quad_nodes,
                eps_total=args.eps_total,
                n_chunk=args.n_chunk,
                m_chunk=args.m_chunk,
                dtype_override=CDTYPE,
            )
            _ = out.real.sum().item()
            dt = (time.perf_counter() - t0) * 1e3
        times.append(dt)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
    thr = B / (avg / 1e3)  # fields/s (each has S outputs)

    print("\n=== T_HS_theta Benchmark ===")
    print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
    print(f"Throughput:    {thr:,.0f} fields/s  (each has S={S})")
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
