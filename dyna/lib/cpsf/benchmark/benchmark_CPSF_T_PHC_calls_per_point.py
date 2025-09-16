# dyna/lib/cpsf/benchmark/benchmark_CPSF_T_PHC_calls_per_point.py
# Examples:
#   CPU:  python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_PHC_calls_per_point --N 256 --M 256 --S 128 --calls 256 --dtype c64 --device cpu
#   CUDA: python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_PHC_calls_per_point --N 256 --M 256 --S 128 --calls 256 --dtype c64 --device cuda --quad_nodes 7 --eps_total 1e-3

import argparse, time, torch
from ..functional.t_phc import T_PHC

def _real_dtype_of(cdtype: torch.dtype) -> torch.dtype:
    return torch.float32 if cdtype == torch.complex64 else torch.float64

def _make_unit(shape, cdtype, device, g):
    REAL = _real_dtype_of(cdtype)
    xr = torch.randn(*shape, generator=g, device=device, dtype=REAL)
    xi = torch.randn(*shape, generator=g, device=device, dtype=REAL)
    v = (xr + 1j * xi).to(cdtype)
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    n = torch.where(n.real == 0, torch.ones_like(n), n)
    return v / n

def _make_complex(shape, cdtype, device, g):
    REAL = _real_dtype_of(cdtype)
    xr = torch.randn(*shape, generator=g, device=device, dtype=REAL)
    xi = torch.randn(*shape, generator=g, device=device, dtype=REAL)
    return (xr + 1j * xi).to(cdtype)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--calls", type=int, default=256, help="How many single-point calls to run")
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--quad_nodes", type=int, default=7)
    ap.add_argument("--eps_total", type=float, default=1.0e-3)
    ap.add_argument("--n_chunk", type=int, default=256)
    ap.add_argument("--m_chunk", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--iters", type=int, default=128, help="Timed iterations (<= calls)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--unique_contribs", action="store_true", help="Use unique contribution pool per call")
    args = ap.parse_args()

    # device
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("WARN: CUDA not available, falling back to CPU.")
            dev = torch.device("cpu")

    CDTYPE = torch.complex64 if args.dtype == "c64" else torch.complex128
    REAL = _real_dtype_of(CDTYPE)
    N, M, S = args.N, args.M, args.S

    print(f"Device={dev.type}, dtype={CDTYPE}, N={N}, M={M}, S={S}")
    print(f"PHC: quad_nodes={args.quad_nodes}, eps_total={args.eps_total}, n_chunk={args.n_chunk}, m_chunk={args.m_chunk}")
    print(f"Mode: single-point calls, calls={args.calls}, warmup={args.warmup}, iters={args.iters}, unique_contribs={args.unique_contribs}")

    # generators
    g0 = torch.Generator(device=dev).manual_seed(args.seed)

    # Prepare per-call inputs
    # Query is always B=1
    z_list, vd_list = [], []
    for i in range(args.calls):
        gi = torch.Generator(device=dev).manual_seed(args.seed + 10 + i)
        z_list.append(_make_complex((1, N), CDTYPE, dev, gi))
        vd_list.append(_make_unit((1, N), CDTYPE, dev, gi))

    # Contributions: either shared for all calls, or unique per call
    if not args.unique_contribs:
        gC = torch.Generator(device=dev).manual_seed(args.seed + 1000)
        z_j = _make_complex((M, N), CDTYPE, dev, gC)
        vd_j = _make_unit((M, N), CDTYPE, dev, gC)
        T_hat = _make_complex((M, S), CDTYPE, dev, gC)
        alpha = torch.rand(M, generator=gC, device=dev, dtype=REAL)
        sq = torch.empty(M, device=dev, dtype=REAL); sp = torch.empty(M, device=dev, dtype=REAL)
        sq.uniform_(0.4, 1.2, generator=gC); sp.uniform_(1.0, 2.0, generator=gC); sp = torch.maximum(sp, sq + 1e-3)
        contribs = (z_j, vd_j, T_hat, alpha, sp, sq)
    else:
        contribs = []
        for i in range(args.calls):
            gC = torch.Generator(device=dev).manual_seed(args.seed + 1000 + i)
            z_j = _make_complex((M, N), CDTYPE, dev, gC)
            vd_j = _make_unit((M, N), CDTYPE, dev, gC)
            T_hat = _make_complex((M, S), CDTYPE, dev, gC)
            alpha = torch.rand(M, generator=gC, device=dev, dtype=REAL)
            sq = torch.empty(M, device=dev, dtype=REAL); sp = torch.empty(M, device=dev, dtype=REAL)
            sq.uniform_(0.4, 1.2, generator=gC); sp.uniform_(1.0, 2.0, generator=gC); sp = torch.maximum(sp, sq + 1e-3)
            contribs.append((z_j, vd_j, T_hat, alpha, sp, sq))

    # Warmup
    if dev.type == "cuda":
        torch.cuda.synchronize()
    for i in range(min(args.warmup, args.calls)):
        z = z_list[i]; vd = vd_list[i]
        if args.unique_contribs:
            z_j, vd_j, T_hat, alpha, sp, sq = contribs[i]
        else:
            z_j, vd_j, T_hat, alpha, sp, sq = contribs
        _ = T_PHC(
            z=z, vec_d=vd,
            z_j=z_j, vec_d_j=vd_j,
            T_hat_j=T_hat, alpha_j=alpha,
            sigma_par_j=sp, sigma_perp_j=sq,
            quad_nodes=args.quad_nodes, eps_total=args.eps_total,
            n_chunk=args.n_chunk, m_chunk=args.m_chunk,
            dtype_override=CDTYPE,
        )
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Timed loop (up to iters)
    iters = min(args.iters, args.calls)
    times_ms = []

    if dev.type == "cuda":
        for i in range(iters):
            z = z_list[i]; vd = vd_list[i]
            if args.unique_contribs:
                z_j, vd_j, T_hat, alpha, sp, sq = contribs[i]
            else:
                z_j, vd_j, T_hat, alpha, sp, sq = contribs
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = T_PHC(
                z=z, vec_d=vd,
                z_j=z_j, vec_d_j=vd_j,
                T_hat_j=T_hat, alpha_j=alpha,
                sigma_par_j=sp, sigma_perp_j=sq,
                quad_nodes=args.quad_nodes, eps_total=args.eps_total,
                n_chunk=args.n_chunk, m_chunk=args.m_chunk,
                dtype_override=CDTYPE,
            )
            _ = out.real.sum().item()
            end.record(); torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    else:
        for i in range(iters):
            z = z_list[i]; vd = vd_list[i]
            if args.unique_contribs:
                z_j, vd_j, T_hat, alpha, sp, sq = contribs[i]
            else:
                z_j, vd_j, T_hat, alpha, sp, sq = contribs
            t0 = time.perf_counter()
            out = T_PHC(
                z=z, vec_d=vd,
                z_j=z_j, vec_d_j=vd_j,
                T_hat_j=T_hat, alpha_j=alpha,
                sigma_par_j=sp, sigma_perp_j=sq,
                quad_nodes=args.quad_nodes, eps_total=args.eps_total,
                n_chunk=args.n_chunk, m_chunk=args.m_chunk,
                dtype_override=CDTYPE,
            )
            _ = out.real.sum().item()
            times_ms.append((time.perf_counter() - t0) * 1e3)

    avg = sum(times_ms) / len(times_ms)
    std = (sum((t - avg) ** 2 for t in times_ms) / max(1, len(times_ms) - 1)) ** 0.5
    calls_per_sec = 1000.0 / avg

    print("\n=== T_PHC single-point call benchmark ===")
    print(f"Avg time/call: {avg:.3f} ms  (± {std:.3f} ms)")
    print(f"Throughput:    {calls_per_sec:,.1f} calls/s  (B=1, each returns S={S} values)")

if __name__ == "__main__":
    main()
