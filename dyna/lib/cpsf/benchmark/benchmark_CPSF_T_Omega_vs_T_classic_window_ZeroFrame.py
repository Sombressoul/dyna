# Run examples:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_Omega_vs_T_classic_window_ZeroFrame --N 4 --M 8 --S 16 --batch 8 --dtype_z c64 --dtype_T c64 --device cuda --iters 25 --warmup 5
#
# Notes:
# - Compares: T_Omega (n=0 only) vs T_classic_window with W=0 (exact n=0 lattice term).
# - Throughput is reported as:
#     * terms/s  : B*M for T_Omega, and B*M*O (O=1) for classic W=0  — so identical numerically here
#     * points/s : B per second

import argparse, time, torch

from dyna.lib.cpsf.functional.core_math import (
    T_classic_window,  # [B,S]
    delta_vec_d,
)
from dyna.lib.cpsf.functional.t_omega import (
    _t_omega_zero_frame,  # [B,S] — n=0 only (no tail), per your definition
)
from dyna.lib.cpsf.periodization import CPSFPeriodization


def _wrap_real(xr: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "floor":  # center to [-0.5, 0.5)
        return xr - torch.floor(xr + 0.5)
    elif mode == "round":  # symmetric wrap via nearest integer
        return xr - torch.round(xr)
    else:
        raise ValueError("wrap mode must be 'floor' or 'round'")


def _T_Omega_variant_core(
    *,
    z,
    z_j,
    vec_d,
    vec_d_j,
    T_hat_j,
    alpha_j,
    sigma_par,
    sigma_perp,
    proj_mode: str,  # "re2" or "abs2"
    b_scale: float,  # 1.0 or 2.0
    wrap_mode: str,  # "floor" or "round"
    apply_dir: bool,  # True or False
):
    # positional delta: wrap ONLY Re-part
    x = z.unsqueeze(1) - z_j  # [B,M,N] complex
    xr = _wrap_real(x.real, wrap_mode)
    x = torch.complex(xr, x.imag)

    # metric coeffs (complex convention: 1/sigma, no squaring)
    inv_sp = torch.reciprocal(sigma_par)  # [B,M]
    inv_sq = torch.reciprocal(sigma_perp)  # [B,M]
    a = inv_sq
    b = b_scale * (inv_sp - inv_sq)  # scaled b

    # quadratic form
    norm2_x = (x.conj() * x).real.sum(dim=-1)  # [B,M]
    inner_h = (vec_d_j.conj() * x).sum(dim=-1)  # [B,M] complex
    if proj_mode == "re2":
        proj_term = (inner_h.real) ** 2
    elif proj_mode == "abs2":
        proj_term = (inner_h.conj() * inner_h).real
    else:
        raise ValueError("proj_mode must be 're2' or 'abs2'")

    q_pos = a * norm2_x + b * proj_term  # [B,M]
    A_pos = torch.exp(-torch.pi * q_pos)  # [B,M]

    # directional factor (optional): isotropic with sigma_perp
    if apply_dir:
        dv = delta_vec_d(vec_d.unsqueeze(1).expand_as(vec_d_j), vec_d_j)  # [B,M,N]
        norm2_dv = (dv.conj() * dv).real.sum(dim=-1)  # [B,M]
        q_dir = inv_sq * norm2_dv
        A_dir = torch.exp(-torch.pi * q_dir)
    else:
        A_dir = torch.ones_like(A_pos)

    gain = (alpha_j * A_pos * A_dir).unsqueeze(-1)  # [B,M,1]
    return (gain * T_hat_j).sum(dim=1)  # [B,S] complex


def scan_variants(
    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp, ref_out
):
    combos = []
    for proj_mode in ("re2", "abs2"):
        for b_scale in (1.0, 2.0):
            for wrap_mode in ("floor", "round"):
                for apply_dir in (True, False):
                    combos.append((proj_mode, b_scale, wrap_mode, apply_dir))

    print("\n=== Variant scan vs T_classic_window(W=0) ===")
    for proj_mode, b_scale, wrap_mode, apply_dir in combos:
        out = _T_Omega_variant_core(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            proj_mode=proj_mode,
            b_scale=b_scale,
            wrap_mode=wrap_mode,
            apply_dir=apply_dir,
        )
        abs_diff = (ref_out - out).abs()
        rel_diff = abs_diff / ref_out.abs().clamp_min(1e-32)
        amax = abs_diff.max().item()
        amean = abs_diff.mean().item()
        rmax = rel_diff.max().item()
        rmean = rel_diff.mean().item()
        ok = torch.allclose(ref_out, out, rtol=1e-5, atol=1e-6)
        flag = "OK " if ok else "   "
        print(
            f"{flag} proj={proj_mode:<4}  b×={b_scale:<3}  wrap={wrap_mode:<5}  dir={'on ' if apply_dir else 'off'}"
            f"  |  Abs max={amax:.3e}, mean={amean:.3e} ; Rel max={rmax:.3e}, mean={rmean:.3e}"
        )


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
        n = torch.where(n == 0, torch.ones_like(n), n)
        z = z / n
    return z


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512, help="Number of target points B")
    ap.add_argument("--dtype_z", choices=["c64", "c128"], default="c64")
    ap.add_argument("--dtype_T", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
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

    B, N, M, S = args.batch, args.N, args.M, args.S
    if N < 2:
        raise SystemExit("CPSF requires N >= 2.")

    print(
        f"Device={dev.type}, dtype_z={dtype_z}, dtype_T={dtype_T}, B={B}, M={M}, S={S}, N={N}, iters={args.iters}, warmup={args.warmup}"
    )

    # For classic W=0 we still form offsets via CPSFPeriodization (O=1, the origin)
    gen = CPSFPeriodization()
    offsets = gen.window(N=N, W=0, device=dev, sorted=False)  # [1, 2N]
    O = int(offsets.shape[0])
    print(f"window size (W=0) O={O} rows")

    # Inputs (outside timing)
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

    alpha_j = 0.2 + 1.3 * torch.rand(B, M, device=dev, dtype=REAL_T)  # [B,M] real
    sigma_par = torch.full((B, M), 0.9, device=dev, dtype=REAL_z)  # [B,M] real > 0
    sigma_perp = torch.full((B, M), 0.5, device=dev, dtype=REAL_z)  # [B,M] real > 0

    Tr = torch.randn(B, M, S, device=dev, dtype=REAL_T)
    Ti = torch.randn(B, M, S, device=dev, dtype=REAL_T)
    T_hat_j = torch.complex(Tr, Ti).to(dtype_T)  # [B,M,S]

    # WARMUPS
    def warmup_classic():
        for _ in range(args.warmup):
            _ = T_classic_window(
                z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp, offsets
            )
        _sync(dev)

    def warmup_omega():
        for _ in range(args.warmup):
            _ = _t_omega_zero_frame(
                z=z.unsqueeze(1).expand_as(z_j),
                z_j=z_j,
                vec_d=vec_d.unsqueeze(1).expand_as(vec_d_j),
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha_j,
                sigma_par=sigma_par,
                sigma_perp=sigma_perp,
            )
        _sync(dev)

    # BENCH: classic W=0
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
            if peak:
                peak_max = max(peak_max, peak)
            if alloc:
                alloc_max = max(alloc_max, alloc)
        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
        secs = avg / 1e3
        thr_terms = (B * M * O) / secs
        thr_points = B / secs
        print("\n=== T_classic_window (W=0) ===")
        print(f"O={O} offsets, B={B:,}, M={M:,}, S={S:,}")
        print(f"Avg time/iter: {avg:.3f} ms  (± {std:.3f} ms)")
        print(f"Throughput:    {thr_terms:,.0f} terms/s    (B*M*O)")
        print(f"Per-target:    {thr_points:,.0f} points/s  (B)")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")
        return avg, std, thr_terms, thr_points, out

    # BENCH: T_Omega (n=0 only)
    def bench_omega():
        warmup_omega()
        times, peak_max, alloc_max = [], 0, 0
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(args.iters):
            dt_ms, peak, alloc, out = _time_block(
                dev,
                lambda: _t_omega_zero_frame(
                    z=z.unsqueeze(1).expand_as(z_j),
                    z_j=z_j,
                    vec_d=vec_d.unsqueeze(1).expand_as(vec_d_j),
                    vec_d_j=vec_d_j,
                    T_hat_j=T_hat_j,
                    alpha_j=alpha_j,
                    sigma_par=sigma_par,
                    sigma_perp=sigma_perp,
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
        thr_terms = (B * M) / secs  # single n=0 term per (B,M)
        thr_points = B / secs
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

    # NUMERIC EQUIVALENCE CHECK
    out_c = res_classic[-1]
    out_o = res_omega[-1]
    with torch.no_grad():
        abs_diff = (out_c - out_o).abs()
        rel_diff = abs_diff / (out_c.abs().clamp_min(1e-32))
        amax = abs_diff.max().item()
        amean = abs_diff.mean().item()
        rmax = rel_diff.max().item()
        rmean = rel_diff.mean().item()
        # dtype-based tolerances
        rtol = 1e-5 if dtype_T == torch.complex64 else 1e-12
        atol = 1e-6 if dtype_T == torch.complex64 else 1e-12
        ok = torch.allclose(out_c, out_o, rtol=rtol, atol=atol)

    print("\n=== Numeric equivalence: T_classic_window(W=0) vs T_Omega ===")
    print(f"Abs diff: max={amax:.3e}, mean={amean:.3e}")
    print(f"Rel diff: max={rmax:.3e}, mean={rmean:.3e}")
    print(f"allclose(rtol={rtol}, atol={atol}) -> {ok}")

    # scan_variants(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp, out_c)

    # SPEED SUMMARY
    t_c = res_classic[0]
    t_o = res_omega[0]
    speedup = t_c / max(1e-9, t_o)
    print("\n=== Summary ===")
    print(
        f"Classic (W=0) avg: {t_c:.3f} ms | Omega avg: {t_o:.3f} ms | speedup: {speedup:.2f}×"
    )


if __name__ == "__main__":
    main()
