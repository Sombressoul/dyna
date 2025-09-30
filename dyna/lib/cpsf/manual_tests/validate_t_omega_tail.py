# > python -m dyna.lib.cpsf.manual_tests.validate_t_omega_tail --ref classic --N 2 --M 16 --S 8 --W 11 --B 1 --iters 128 --dtype_z c128 --dtype_T c128 --device cuda --q_theta 24 --q_rad 128 --guards
import math
import argparse
import torch
from typing import Dict, Tuple, Literal

from dyna.lib.cpsf.periodization import CPSFPeriodization
from dyna.lib.cpsf.functional.core_math import T_classic_window
from dyna.lib.cpsf.functional.t_pd import T_PD_window
from dyna.lib.cpsf.functional.t_omega import T_Omega, T_Omega_Components


# -------------------------
# Helpers
# -------------------------

@torch.no_grad()
def _pick_device(sel: str) -> torch.device:
    if sel == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(sel)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("WARN: CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return dev


@torch.no_grad()
def _rand_complex(shape, dtype: torch.dtype, device, rng: torch.Generator, scale: float = 1.0):
    assert dtype.is_complex
    rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
    re = torch.randn(shape, dtype=rdtype, device=device, generator=rng) * scale
    im = torch.randn(shape, dtype=rdtype, device=device, generator=rng) * scale
    return torch.complex(re, im).to(dtype)


@torch.no_grad()
def _rand_unit_complex(shape, dtype: torch.dtype, device, rng: torch.Generator):
    x = _rand_complex(shape, dtype, device, rng)
    n = torch.sqrt(torch.clamp((x.real**2 + x.imag**2).sum(dim=-1, keepdim=True),
                               min=torch.finfo(x.real.dtype).tiny))
    return x / n


@torch.no_grad()
def _sample_batch(
    B: int,
    N: int,
    M: int,
    S: int,
    dtype_z: torch.dtype,
    dtype_T: torch.dtype,
    device,
    rng: torch.Generator,
    sigma_min: float,
    sigma_max: float,
    anisotropy_range: Tuple[float, float],
):
    rdz = torch.float64 if dtype_z == torch.complex128 else torch.float32
    rdT = torch.float64 if dtype_T == torch.complex128 else torch.float32

    z = torch.complex(
        torch.rand(B, N, dtype=rdz, device=device, generator=rng),
        torch.rand(B, N, dtype=rdz, device=device, generator=rng),
    ).to(dtype_z)
    z_j = torch.complex(
        torch.rand(B, M, N, dtype=rdz, device=device, generator=rng),
        torch.rand(B, M, N, dtype=rdz, device=device, generator=rng),
    ).to(dtype_z)

    vec_d   = _rand_unit_complex((B, N),     dtype_z, device, rng)
    vec_d_j = _rand_unit_complex((B, M, N),  dtype_z, device, rng)

    T_hat_j = _rand_complex((B, M, S), dtype_T, device, rng, scale=1.0)

    alpha_j = torch.exp(torch.randn(B, M, dtype=rdT, device=device, generator=rng) * 0.25)

    u = torch.rand(B, M, dtype=rdT, device=device, generator=rng)
    log_sig_perp = math.log(sigma_min) + (math.log(sigma_max) - math.log(sigma_min)) * u
    sigma_perp = torch.exp(log_sig_perp)

    a0, a1 = anisotropy_range
    v = torch.rand(B, M, dtype=rdT, device=device, generator=rng)
    log_ratio = math.log(a0) + (math.log(a1) - math.log(a0)) * v
    ratio = torch.exp(log_ratio)
    sigma_par = sigma_perp * ratio

    return z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp


@torch.no_grad()
def _call_ref_total(
    ref: Literal["pd", "classic"],
    *,
    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp,
    offsets: torch.Tensor,
    t_PD: float,
):
    if ref == "pd":
        return T_PD_window(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            offsets=offsets,
            t=t_PD,
        )
    elif ref == "classic":
        return T_classic_window(
            z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp, offsets
        )
    else:
        raise ValueError(f"Unknown ref='{ref}' (expected 'pd' or 'classic').")


@torch.no_grad()
def _update_metrics(
    acc: Dict[str, float],
    tail_ref: torch.Tensor,       # [B, S] complex
    tail_omega: torch.Tensor,     # [B, S] complex
    tau_abs: float,
    tau_rel: float,
):
    diff = (tail_omega - tail_ref)
    abs_err = diff.abs()
    abs_ref = tail_ref.abs()

    small = abs_ref < tau_abs
    large = ~small

    if large.any():
        denom = torch.maximum(abs_ref[large], torch.as_tensor(tau_rel, dtype=abs_ref.dtype, device=abs_ref.device))
        rel = (abs_err[large] / denom).to(torch.float64)
        acc["rel_sum"] += rel.sum().item()
        acc["rel_cnt"] += rel.numel()
        acc["rel_max"] = max(acc["rel_max"], float(rel.max().item()))

    if small.any():
        abs_small = abs_err[small].to(torch.float64)
        acc["abs_small_max"] = max(acc["abs_small_max"], float(abs_small.max().item()))
        fp = (tail_omega.abs()[small] >= tau_abs)
        acc["fp_small_cnt"] += int(fp.sum().item())
        acc["small_cnt"] += int(small.sum().item())

    acc["snr_num"] += float((abs_ref ** 2).sum().item())
    acc["snr_den"] += float((abs_err ** 2).sum().item())


# -------------------------
# Validator
# -------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Tail-to-tail validator: T_Omega (TAIL) vs windowed reference minus ZERO.")
    ap.add_argument("--ref", choices=["pd", "classic"], default="pd", help="Reference: PD or classic window")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=123)

    # shapes
    ap.add_argument("--B", type=int, default=1, help="Batch size per iteration")
    ap.add_argument("--iters", type=int, default=128, help="Number of iterations (streaming)")
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--S", type=int, default=8)

    # dtypes
    ap.add_argument("--dtype_z", choices=["c64", "c128"], default="c128")
    ap.add_argument("--dtype_T", choices=["c64", "c128"], default="c128")

    # window + PD
    ap.add_argument("--W", type=int, default=11, help="L∞ window radius for CPSFPeriodization.window")
    ap.add_argument("--t_PD", type=float, default=1.0, help="Ewald/Poisson scale t>0 for PD reference")

    # Omega quadratures
    ap.add_argument("--q_theta", type=int, default=24)
    ap.add_argument("--q_rad", type=int, default=128)
    ap.add_argument("--guards", action="store_true", help="Enable T_Omega numerical guards")

    # generation ranges
    ap.add_argument("--sigma_min", type=float, default=0.08)
    ap.add_argument("--sigma_max", type=float, default=3.0)
    ap.add_argument("--anisotropy_min", type=float, default=0.25, help="min ratio = sigma_par/sigma_perp")
    ap.add_argument("--anisotropy_max", type=float, default=4.0,   help="max ratio = sigma_par/sigma_perp")

    # thresholds
    ap.add_argument("--tau_abs_scale", type=float, default=200.0, help="abs threshold = scale * eps * tau_abs_scale")
    ap.add_argument("--tau_rel_scale", type=float, default=1000.0, help="rel denom floor = scale * eps * tau_rel_scale")

    args = ap.parse_args()

    dev = _pick_device(args.device)
    dtype_z = torch.complex64 if args.dtype_z == "c64" else torch.complex128
    dtype_T = torch.complex64 if args.dtype_T == "c64" else torch.complex128
    REAL_z = torch.float32 if dtype_z == torch.complex64 else torch.float64
    REAL_T = torch.float32 if dtype_T == torch.complex64 else torch.float64

    if args.N < 2:
        raise SystemExit("CPSF requires N >= 2.")
    if args.W < 0:
        raise SystemExit("W must be >= 0.")
    if args.ref == "pd" and args.t_PD <= 0.0:
        raise SystemExit("t_PD must be > 0 for PD reference.")

    # RNG
    try:
        torch.set_default_device(dev.type)
    except Exception:
        pass
    g = torch.Generator(device=dev).manual_seed(int(args.seed))

    # Offsets via CPSFPeriodization
    gen = CPSFPeriodization()
    offsets = gen.window(N=args.N, W=args.W, device=dev, sorted=False)  # [O, 2N]
    O = int(offsets.shape[0])
    print(f"[info] device={dev.type}, dtype_z={dtype_z}, dtype_T={dtype_T}, B={args.B}, iters={args.iters}, "
          f"N={args.N}, M={args.M}, S={args.S}, W={args.W}, O={O}, ref={args.ref}, t_PD={args.t_PD}, "
          f"q_theta={args.q_theta}, q_rad={args.q_rad}, guards={args.guards}")

    z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp = _sample_batch(
        args.B, args.N, args.M, args.S, dtype_z, dtype_T, dev, g,
        args.sigma_min, args.sigma_max, (args.anisotropy_min, args.anisotropy_max)
    )

    scale_alpha = alpha_j.abs().median().to(REAL_T)
    scale_That  = T_hat_j.abs().amax(dim=-1).median().to(REAL_T)
    scale = (scale_alpha * scale_That).item()
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    eps = torch.finfo(REAL_T).eps
    tau_abs = float(args.tau_abs_scale * eps * scale)
    tau_rel = float(args.tau_rel_scale * eps * scale)
    print(f"[info] thresholds: tau_abs={tau_abs:.3e}, tau_rel={tau_rel:.3e}, scale~{scale:.3e}")

    acc = dict(
        rel_sum=0.0, rel_cnt=0, rel_max=0.0,
        abs_small_max=0.0, fp_small_cnt=0, small_cnt=0,
        snr_num=0.0, snr_den=0.0,
        covered_samples=0,
    )

    for it in range(args.iters):
        z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp = _sample_batch(
            args.B, args.N, args.M, args.S, dtype_z, dtype_T, dev, g,
            args.sigma_min, args.sigma_max, (args.anisotropy_min, args.anisotropy_max)
        )

        T_zero = T_Omega(
            z=z, z_j=z_j,
            vec_d=vec_d, vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j.to(REAL_T),
            sigma_par=sigma_par.to(REAL_z),
            sigma_perp=sigma_perp.to(REAL_z),
            return_components=T_Omega_Components.ZERO,
            guards=args.guards,
            q_theta=args.q_theta,
            q_rad=args.q_rad,
        )

        T_tail_omega = T_Omega(
            z=z, z_j=z_j,
            vec_d=vec_d, vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j.to(REAL_T),
            sigma_par=sigma_par.to(REAL_z),
            sigma_perp=sigma_perp.to(REAL_z),
            return_components=T_Omega_Components.TAIL,
            guards=args.guards,
            q_theta=args.q_theta,
            q_rad=args.q_rad,
        )

        T_ref_total = _call_ref_total(
            args.ref,
            z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
            T_hat_j=T_hat_j, alpha_j=alpha_j.to(REAL_T),
            sigma_par=sigma_par.to(REAL_z), sigma_perp=sigma_perp.to(REAL_z),
            offsets=offsets, t_PD=args.t_PD,
        )

        T_tail_ref = T_ref_total - T_zero  # [B, S]

        _update_metrics(acc, T_tail_ref, T_tail_omega, tau_abs=tau_abs, tau_rel=tau_rel)
        acc["covered_samples"] += int(T_tail_ref.numel())

        if (it + 1) % max(1, args.iters // 10) == 0:
            mean_rel = (acc["rel_sum"] / acc["rel_cnt"]) if acc["rel_cnt"] > 0 else float("nan")
            print(f"[{it+1}/{args.iters}] mean_rel={mean_rel:.3e}, rel_max={acc['rel_max']:.3e}, "
                  f"max_abs_small={acc['abs_small_max']:.3e}, fp_rate_small="
                  f"{(acc['fp_small_cnt']/max(1,acc['small_cnt'])):.3e}")

    mean_rel = (acc["rel_sum"] / acc["rel_cnt"]) if acc["rel_cnt"] > 0 else float("nan")
    max_rel = acc["rel_max"]
    max_abs_small = acc["abs_small_max"]
    fp_rate_small = (acc["fp_small_cnt"] / acc["small_cnt"]) if acc["small_cnt"] > 0 else 0.0
    if acc["snr_den"] > 0.0:
        snr_db = 10.0 * math.log10(acc["snr_num"] / max(1e-300, acc["snr_den"]))
    else:
        snr_db = float("inf")

    print("\n=== Tail-to-tail validation summary ===")
    print(f"ref={args.ref}, device={dev.type}, dtype_z={dtype_z}, dtype_T={dtype_T}")
    print(f"B={args.B}, iters={args.iters}, N={args.N}, M={args.M}, S={args.S}, W={args.W}, O={O}")
    print(f"q_theta={args.q_theta}, q_rad={args.q_rad}, guards={args.guards}, t_PD={args.t_PD if args.ref=='pd' else 'n/a'}")
    print(f"tau_abs={tau_abs:.3e}, tau_rel={tau_rel:.3e}, covered_samples={acc['covered_samples']}")
    print(f"mean_rel={mean_rel:.6e}, rel_max={max_rel:.6e}, max_abs_small={max_abs_small:.6e}, "
          f"fp_rate_small={fp_rate_small:.6e}, snr_db={snr_db:.2f}")


if __name__ == "__main__":
    main()
