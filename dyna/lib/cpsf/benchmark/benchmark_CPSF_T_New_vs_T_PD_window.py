# Save as:
# dyna/lib/cpsf/benchmark/benchmark_CPSF_T_New_vs_T_PD_window.py
#
# Examples:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_New_vs_T_PD_window --N 2 --M 16 --S 16 --W 11 --batch 16 --dtype_z c64 --dtype_T c64 --device cuda --iters 100 --warmup 5 --etol_abs 1e-5 --etol_rel 1e-5 --per_iter
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_New_vs_T_PD_window --N 3 --M 16 --S 16 --W 7 --batch 1 --dtype_z c64 --dtype_T c64 --device cuda --iters 100 --warmup 5 --etol_abs 1e-5 --etol_rel 1e-5 --per_iter
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_New_vs_T_PD_window --N 4 --M 16 --S 16 --W 3 --batch 2 --dtype_z c64 --dtype_T c64 --device cuda --iters 100 --warmup 5 --etol_abs 1e-5 --etol_rel 1e-5 --per_iter
#
# Examples for q_order and error_budget checks:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_New_vs_T_PD_window --N 2 --M 16 --S 16 --W 11 --batch 16 --dtype_z c64 --dtype_T c64 --device cuda --iters 100 --warmup 5 --etol_abs 1e-5 --etol_rel 1e-5 --per_iter --error_budget 1.0e-7 --q_order 12
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_T_New_vs_T_PD_window --N 2 --M 16 --S 16 --W 11 --batch 16 --dtype_z c64 --dtype_T c64 --device cuda --iters 100 --warmup 5 --etol_abs 1e-5 --etol_rel 1e-5 --per_iter --error_budget 1.0e-3 --q_order 5

import argparse, math
import torch

# --- Imports of target functions ---
from dyna.lib.cpsf.functional.t_pd import T_PD_window # pd reference
from dyna.lib.cpsf.functional.t_new import T_New  # candidate under test
from dyna.lib.cpsf.periodization import CPSFPeriodization


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


def _stats_mag(x: torch.Tensor):
    a = x.abs()
    return a.mean().item(), a.std(unbiased=False).item(), a.min().item(), a.max().item()


def _snr_db(ref: torch.Tensor, err: torch.Tensor, eps: float = 1e-30) -> float:
    # 20 * log10( ||ref||_2 / ||err||_2 )
    num = torch.linalg.vector_norm(ref).item()
    den = max(eps, torch.linalg.vector_norm(err).item())
    return 20.0 * math.log10(max(eps, num) / den)


def _sample_sigmas(
    B: int,
    M: int,
    *,
    device: torch.device,
    rtype: torch.dtype,
    seed: int,
    sigma_min: float,
    sigma_max: float,
):
    """Uniform sampling per-j in [sigma_min, sigma_max], independent for par/perp."""
    g = torch.Generator(device=device).manual_seed(seed)
    u_par = torch.rand(B, M, device=device, dtype=rtype, generator=g)
    u_perp = torch.rand(B, M, device=device, dtype=rtype, generator=g)
    span = float(sigma_max - sigma_min)
    sigma_par = sigma_min + span * u_par
    sigma_perp = sigma_min + span * u_perp
    # strictly positive by construction; no extra clamps needed
    return sigma_par, sigma_perp


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8, help="space dimension")
    ap.add_argument("--M", type=int, default=64, help="number of contributions")
    ap.add_argument("--S", type=int, default=8, help="semantic vector size")
    ap.add_argument(
        "--W", type=int, default=2, help="L∞ window radius for T_PD_window"
    )
    ap.add_argument("--batch", type=int, default=128, help="B: points per iteration")
    ap.add_argument("--dtype_z", choices=["c64", "c128"], default="c64")
    ap.add_argument("--dtype_T", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--iters", type=int, default=25, help="number of fresh batches")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--per_iter", action="store_true", help="print per-iteration metrics"
    )

    # NEW: random kernel scales per j
    ap.add_argument(
        "--sigma_min", type=float, default=0.1, help="min of uniform range for sigma"
    )
    ap.add_argument(
        "--sigma_max", type=float, default=1.5, help="max of uniform range for sigma"
    )

    # numeric tolerances
    ap.add_argument(
        "--etol_abs", type=float, default=1e-6, help="absolute error tolerance (max)"
    )
    ap.add_argument(
        "--etol_rel", type=float, default=1e-6, help="relative error tolerance (max)"
    )

    # algoalgorithm params
    ap.add_argument(
        "--error_budget", type=float, default=1e-5, help="error budget"
    )
    ap.add_argument(
        "--q_order", type=int, default=7, help="GH Q order"
    )

    args = ap.parse_args()

    if args.sigma_min <= 0.0 or args.sigma_max <= 0.0:
        raise SystemExit("sigma_min and sigma_max must be positive.")
    if not (args.sigma_min < args.sigma_max):
        raise SystemExit("Require sigma_min < sigma_max.")

    dev = _pick_device(args.device)
    try:
        torch.set_default_device("cuda" if dev.type == "cuda" else "cpu")
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

    print(
        f"Device={dev.type}, dtype_z={dtype_z}, dtype_T={dtype_T}, "
        f"B={B}, N={N}, M={M}, S={S}, W={W}, iters={args.iters}, warmup={args.warmup}, "
        f"sigma_min={args.sigma_min}, sigma_max={args.sigma_max}, "
        f"etol_abs={args.etol_abs}, etol_rel={args.etol_rel}, "
        f"error_budget={args.error_budget}, q_order={args.q_order}"
    )

    # Offsets for PD window
    gen = CPSFPeriodization()
    offsets = gen.window(N=N, W=W, device=dev, sorted=False)  # [O, 2N]
    O = int(offsets.shape[0])
    print(f"window size O={O:,} offsets for T_PD_window")

    # Seed baseline
    seed0 = 12345

    # --- Warmup (smaller synthetic batch but with random sigmas per j) ---
    z = _make_cplx(
        B, N, dtype=dtype_z, device=dev, seed=seed0 + 1, unitize=False
    )  # [B,N]
    z_j = _make_cplx(
        B, M, N, dtype=dtype_z, device=dev, seed=seed0 + 2, unitize=False
    )  # [M,N]
    vec_d = _make_cplx(
        B, N, dtype=dtype_z, device=dev, seed=seed0 + 3, unitize=True
    )  # [B,N]
    vec_d_j = _make_cplx(
        B, M, N, dtype=dtype_z, device=dev, seed=seed0 + 4, unitize=True
    )  # [M,N]
    Tr = torch.randn(
        B,
        M,
        S,
        device=dev,
        dtype=REAL_T,
        generator=torch.Generator(device=dev).manual_seed(seed0 + 5),
    )
    Ti = torch.randn(
        B,
        M,
        S,
        device=dev,
        dtype=REAL_T,
        generator=torch.Generator(device=dev).manual_seed(seed0 + 6),
    )
    T_hat_j = torch.complex(Tr, Ti).to(dtype_T)  # [M,S]
    alpha_j = 0.2 + 1.3 * torch.rand(
        B,
        M,
        device=dev,
        dtype=REAL_T,
        generator=torch.Generator(device=dev).manual_seed(seed0 + 7),
    )  # [M]

    # sample random sigmas per j for warmup
    sigma_par, sigma_perp = _sample_sigmas(
        B,
        M,
        device=dev,
        rtype=REAL_z,
        seed=seed0 + 8,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )

    for _ in range(args.warmup):
        _ = T_New(
            z=z,
            vec_d=vec_d,
            z_j=z_j,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            error_budget=args.error_budget,
            q_order=args.q_order,
        )
        _ = T_PD_window(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            offsets=offsets,
        )
    _sync(dev)

    # --- Aggregated metrics across iters ---
    acc_abs_mean, acc_abs_max = [], []
    acc_rel_mean, acc_rel_max = [], []
    acc_snr = []
    acc_stats_NEW, acc_stats_cls = [], []
    failures = 0

    # --- Per-iter loop with fresh random data each time ---
    for it in range(args.iters):
        seed = seed0 + 1000 + it * 10

        z = _make_cplx(B, N, dtype=dtype_z, device=dev, seed=seed + 1, unitize=False)
        z_j = _make_cplx(
            B, M, N, dtype=dtype_z, device=dev, seed=seed + 2, unitize=False
        )
        vec_d = _make_cplx(B, N, dtype=dtype_z, device=dev, seed=seed + 3, unitize=True)
        vec_d_j = _make_cplx(
            B, M, N, dtype=dtype_z, device=dev, seed=seed + 4, unitize=True
        )

        Tr = torch.randn(
            B,
            M,
            S,
            device=dev,
            dtype=REAL_T,
            generator=torch.Generator(device=dev).manual_seed(seed + 5),
        )
        Ti = torch.randn(
            B,
            M,
            S,
            device=dev,
            dtype=REAL_T,
            generator=torch.Generator(device=dev).manual_seed(seed + 6),
        )
        T_hat_j = torch.complex(Tr, Ti).to(dtype_T)

        alpha_j = 0.2 + 1.3 * torch.rand(
            B,
            M,
            device=dev,
            dtype=REAL_T,
            generator=torch.Generator(device=dev).manual_seed(seed + 7),
        )

        # NEW: sample sigmas per j each iteration
        sigma_par, sigma_perp = _sample_sigmas(
            B,
            M,
            device=dev,
            rtype=REAL_z,
            seed=seed + 8,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )

        # Evaluate both
        out_NEW = T_New(
            z=z,
            vec_d=vec_d,
            z_j=z_j,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            error_budget=args.error_budget,
            q_order=args.q_order,
        )  # [B,S] complex

        out_cls = T_PD_window(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            offsets=offsets,
        )  # [B,S] complex

        # Numeric comparison
        diff = out_NEW - out_cls
        abs_diff = diff.abs()
        denom = out_cls.abs().clamp_min(1e-30)
        rel = abs_diff / denom

        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()
        max_rel = rel.max().item()
        mean_rel = rel.mean().item()
        snr = _snr_db(out_cls, diff)

        s_NEW = _stats_mag(out_NEW)
        s_cls = _stats_mag(out_cls)

        acc_abs_mean.append(mean_abs)
        acc_abs_max.append(max_abs)
        acc_rel_mean.append(mean_rel)
        acc_rel_max.append(max_rel)
        acc_snr.append(snr)
        acc_stats_NEW.append(s_NEW)
        acc_stats_cls.append(s_cls)

        ok = (max_abs <= args.etol_abs) and (max_rel <= args.etol_rel)
        failures += 0 if ok else 1

        if args.per_iter:
            print(
                f"[iter {it+1:03d}] abs: mean={mean_abs:.3e} max={max_abs:.3e} | "
                f"rel: mean={mean_rel:.3e} max={max_rel:.3e} | snr={snr:.2f} dB | "
                f"|NEW| mean={s_NEW[0]:.3e} std={s_NEW[1]:.3e} min={s_NEW[2]:.3e} max={s_NEW[3]:.3e} | "
                f"|CLS| mean={s_cls[0]:.3e} std={s_cls[1]:.3e} min={s_cls[2]:.3e} max={s_cls[3]:.3e} | "
                f"{'OK' if ok else 'FAIL'}"
            )

    # Aggregate summary
    def _mean(x):
        return sum(x) / max(1, len(x))

    def _std(x):
        m = _mean(x)
        return (
            ((sum((v - m) ** 2 for v in x) / max(1, len(x) - 1)) ** 0.5)
            if len(x) > 1
            else 0.0
        )

    print("\n=== Numeric equivalence: NEW vs PD_Window ===")
    print(
        f"iters={args.iters}, failures={failures} (criteria: max_abs <= {args.etol_abs}, max_rel <= {args.etol_rel})"
    )
    print(
        f"Abs error: mean={_mean(acc_abs_mean):.3e} ±{_std(acc_abs_mean):.3e} | max over iters={max(acc_abs_max):.3e}"
    )
    print(
        f"Rel error: mean={_mean(acc_rel_mean):.3e} ±{_std(acc_rel_mean):.3e} | max over iters={max(acc_rel_max):.3e}"
    )
    print(
        f"SNR (dB): mean={_mean(acc_snr):.2f} ±{_std(acc_snr):.2f} | min over iters={min(acc_snr):.2f}"
    )

    # Aggregate output stats (averaged over iters)
    mNEW = tuple(_mean([s[i] for s in acc_stats_NEW]) for i in range(4))
    mCLS = tuple(_mean([s[i] for s in acc_stats_cls]) for i in range(4))
    print("\n--- Output magnitude stats (averaged over iters) ---")
    print(
        f"|NEW| mean={mNEW[0]:.3e} std={mNEW[1]:.3e} min={mNEW[2]:.3e} max={mNEW[3]:.3e}"
    )
    print(
        f"|CLS| mean={mCLS[0]:.3e} std={mCLS[1]:.3e} min={mCLS[2]:.3e} max={mCLS[3]:.3e}"
    )


if __name__ == "__main__":
    main()
