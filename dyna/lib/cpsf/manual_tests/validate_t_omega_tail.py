# > python -m dyna.lib.cpsf.manual_tests.validate_t_omega_tail --ref classic --N 2 --M 16 --S 8 --W 11 --B 1 --iters 128 --dtype_z c128 --dtype_T c128 --device cuda --seed 42
# > python -m dyna.lib.cpsf.manual_tests.validate_t_omega_tail --ref pd --N 3 --M 8 --S 8 --W 7 --B 1 --iters 128 --dtype_z c128 --dtype_T c128 --device cuda --seed 42
import os
import math
import argparse
from typing import Tuple

import torch

from dyna.lib.cpsf.periodization import CPSFPeriodization
from dyna.lib.cpsf.functional.core_math import T_classic_window
from dyna.lib.cpsf.functional.t_pd import T_PD_window
from dyna.lib.cpsf.functional.t_omega import T_Omega, T_Omega_Components


# ----------------------------
# Determinism helpers
# ----------------------------

def _configure_determinism(seed: int, device: torch.device, set_cublas_env: bool = True):
    # set Python-side torch RNGs too (for any internal ops that might not accept generator)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        # Strong deterministic mode in PyTorch
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Disable TF32 (to avoid tiny numeric jitter)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        # For cuBLAS determinism in GEMM reductions
        if set_cublas_env and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            # Valid options: ":16:8" or ":4096:8"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)


# ----------------------------
# RNG / types
# ----------------------------

def _real_dtype_of_complex(dtype_c: torch.dtype) -> torch.dtype:
    if dtype_c == torch.complex64:
        return torch.float32
    if dtype_c == torch.complex128:
        return torch.float64
    raise ValueError(f"Unsupported complex dtype: {dtype_c}")

def _rand_unit_complex(shape: Tuple[int, ...], device, dtype_c, g: torch.Generator) -> torch.Tensor:
    """Random complex unit vectors (normalize along last dim) using generator g."""
    dtype_r = _real_dtype_of_complex(dtype_c)
    x = torch.randn(shape, device=device, dtype=dtype_r, generator=g)
    y = torch.randn(shape, device=device, dtype=dtype_r, generator=g)
    v = torch.complex(x, y)
    nrm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(torch.finfo(dtype_r).tiny)
    return v / nrm

def _sample_batch(
    B, N, M, S, dtype_z, dtype_T, device, g: torch.Generator,
    sigma_min: float, sigma_max: float,
    anisotropy: Tuple[float, float]
):
    """Synthesize a random batch consistent with CPSF inputs (fully driven by generator g)."""
    # Positions
    z   = _rand_unit_complex((B, N),     device, dtype_z, g)      # [B,N]
    z_j = _rand_unit_complex((B, M, N),  device, dtype_z, g)      # [B,M,N]

    # Directions (unit)
    vec_d   = _rand_unit_complex((B, N),    device, dtype_z, g)   # [B,N]
    vec_d_j = _rand_unit_complex((B, M, N), device, dtype_z, g)   # [B,M,N]

    # Components in span S (keep imaginary part zero to match earlier setup, but still deterministic)
    REAL_T = _real_dtype_of_complex(dtype_T)
    T_hat_real = torch.randn((B, M, S), device=device, dtype=REAL_T, generator=g)
    T_hat_j = torch.complex(T_hat_real, torch.zeros_like(T_hat_real)).to(dtype_T)  # [B,M,S]

    # Positive weights
    alpha_j = torch.rand((B, M), device=device, dtype=REAL_T, generator=g) + 0.5  # [B,M]

    # Sigma sampling: 1/sigma convention (scale, not variance)
    sigma_perp = torch.empty((B, M), device=device, dtype=REAL_T)
    sigma_par  = torch.empty_like(sigma_perp)
    sigma_perp.uniform_(sigma_min, sigma_max, generator=g)
    sigma_par.uniform_(sigma_min, sigma_max, generator=g)

    # Optional controlled anisotropy (if anisotropy != (1,1))
    an_min, an_max = anisotropy
    if (an_min > 0.0) and (an_max >= an_min) and not (math.isclose(an_min, 1.0) and math.isclose(an_max, 1.0)):
        r = torch.empty_like(sigma_par).uniform_(an_min, an_max, generator=g)
        sigma_par = sigma_perp * r

    return z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp


# ----------------------------
# Main validation loop
# ----------------------------

def main():
    p = argparse.ArgumentParser("Validate T_Omega tail vs PD/classic reference (tail-to-tail).")
    p.add_argument("--ref", choices=["pd", "classic"], default="pd",
                   help="Reference algorithm for the full windowed sum (we subtract T_zero from it to get tail).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype_z", choices=["c64", "c128"], default="c128")
    p.add_argument("--dtype_T", choices=["c64", "c128"], default="c128")

    p.add_argument("--B", type=int, default=1)
    p.add_argument("--iters", type=int, default=128)
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--S", type=int, default=8)

    p.add_argument("--W", type=int, default=7, help="Window radius for the periodization window.")
    p.add_argument("--t_PD", type=float, default=1.0, help="Scaling parameter for T_PD_window (if used).")

    p.add_argument("--q_theta", type=int, default=24)
    p.add_argument("--q_rad", type=int, default=128)
    p.add_argument("--guards", action="store_true", default=False)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Enable strict deterministic execution on CUDA (recommended).")

    # sigma and anisotropy control
    p.add_argument("--sigma_min", type=float, default=0.25)
    p.add_argument("--sigma_max", type=float, default=2.0)
    p.add_argument("--anisotropy_min", type=float, default=1.0)
    p.add_argument("--anisotropy_max", type=float, default=1.0)

    # thresholds (scaled by eps*scale)
    p.add_argument("--tau_abs_scale", type=float, default=2.0)
    p.add_argument("--tau_rel_scale", type=float, default=10.0)

    args = p.parse_args()

    dev = torch.device(args.device)
    if args.deterministic:
        _configure_determinism(args.seed, dev)

    dtype_z = torch.complex64 if args.dtype_z == "c64" else torch.complex128
    dtype_T = torch.complex64 if args.dtype_T == "c64" else torch.complex128
    REAL_T = _real_dtype_of_complex(dtype_T)

    # Single local generator drives all randomness in this script
    g = torch.Generator(device=dev).manual_seed(int(args.seed))

    # Offsets via CPSFPeriodization (canonical)
    gen = CPSFPeriodization()
    offsets = gen.window(N=args.N, W=args.W, device=dev, sorted=False)  # [O, 2N]
    O = int(offsets.shape[0])

    print(
        f"[info] device={dev.type}, dtype_z={dtype_z}, dtype_T={dtype_T}, "
        f"B={args.B}, iters={args.iters}, N={args.N}, M={args.M}, S={args.S}, "
        f"W={args.W}, O={O}, ref={args.ref}, t_PD={args.t_PD}, "
        f"q_theta={args.q_theta}, q_rad={args.q_rad}, guards={args.guards}"
    )

    with torch.no_grad():
        # Pre-sample ONCE to estimate scale for thresholds (driven by the same generator g)
        z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp = _sample_batch(
            args.B, args.N, args.M, args.S, dtype_z, dtype_T, dev, g,
            args.sigma_min, args.sigma_max, (args.anisotropy_min, args.anisotropy_max),
        )
        scale_alpha = alpha_j.abs().median().to(REAL_T)
        scale_That  = T_hat_j.abs().amax(dim=-1).median().to(REAL_T)
        scale = float((scale_alpha * scale_That).clamp_min(torch.finfo(REAL_T).tiny).item())

        eps = torch.finfo(REAL_T).eps
        tau_abs = float(args.tau_abs_scale * eps * scale)
        tau_rel = float(args.tau_rel_scale * eps * scale)
        print(f"[info] thresholds: tau_abs={tau_abs:.3e}, tau_rel={tau_rel:.3e}, scale~{scale:.3e}")

        # Accumulators
        acc = dict(
            rel_err_sum=0.0, rel_err_count=0, rel_err_max=0.0,     # relative error stats
            abs_err_sum=0.0, abs_err_count=0, abs_err_max=0.0,     # absolute error stats
            abs_err_small_max=0.0, fp_small_count=0, small_count=0, # small-tail diagnostics
            snr_num=0.0, snr_den=0.0,                              # SNR accumulators
            covered_samples=0,                                     # coverage
        )

        def _call_ref_total(_z, _zj, _vd, _vdj, _Tj, _a, _sp, _spp):
            if args.ref == "pd":
                return T_PD_window(
                    z=_z, z_j=_zj, vec_d=_vd, vec_d_j=_vdj,
                    T_hat_j=_Tj, alpha_j=_a,
                    sigma_par=_sp, sigma_perp=_spp,
                    offsets=offsets, t=args.t_PD,
                )
            else:
                return T_classic_window(
                    z=_z, z_j=_zj, vec_d=_vd, vec_d_j=_vdj,
                    T_hat_j=_Tj, alpha_j=_a,
                    sigma_par=_sp, sigma_perp=_spp,
                    offsets=offsets,
                )

        for it in range(1, args.iters + 1):
            # fresh batch each iter (driven by the same generator g -> deterministic sequence)
            z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp = _sample_batch(
                args.B, args.N, args.M, args.S, dtype_z, dtype_T, dev, g,
                args.sigma_min, args.sigma_max, (args.anisotropy_min, args.anisotropy_max),
            )

            # omega tail and zero components
            T_tail_omega = T_Omega(
                z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
                T_hat_j=T_hat_j, alpha_j=alpha_j,
                sigma_par=sigma_par, sigma_perp=sigma_perp,
                return_components=T_Omega_Components.TAIL,
                guards=args.guards, q_theta=args.q_theta, q_rad=args.q_rad,
            )  # [B,S]
            T_zero_omega = T_Omega(
                z=z, z_j=z_j, vec_d=vec_d, vec_d_j=vec_d_j,
                T_hat_j=T_hat_j, alpha_j=alpha_j,
                sigma_par=sigma_par, sigma_perp=sigma_perp,
                return_components=T_Omega_Components.ZERO,
                guards=args.guards, q_theta=args.q_theta, q_rad=args.q_rad,
            )  # [B,S]

            # reference full windowed sum
            T_ref_total = _call_ref_total(z, z_j, vec_d, vec_d_j, T_hat_j, alpha_j, sigma_par, sigma_perp)  # [B,S]
            # reference tail = total - zero (tail-to-tail comparison)
            T_ref_tail = T_ref_total - T_zero_omega

            # errors
            delta = T_tail_omega - T_ref_tail
            abs_err = delta.abs()
            denom = torch.maximum(T_ref_tail.abs(), torch.as_tensor(tau_abs, device=dev, dtype=REAL_T))
            rel_err = (abs_err / denom).to(REAL_T)

            # accumulate
            acc["rel_err_sum"]  += float(rel_err.sum().item())
            acc["rel_err_count"] += int(rel_err.numel())
            acc["rel_err_max"]   = max(acc["rel_err_max"], float(rel_err.max().item()))

            acc["abs_err_sum"]  += float(abs_err.sum().item())
            acc["abs_err_count"] += int(abs_err.numel())
            acc["abs_err_max"]   = max(acc["abs_err_max"], float(abs_err.max().item()))

            # small-tail diagnostics (reference tail numerically zero)
            small_mask = (T_ref_tail.abs() <= tau_abs)
            if small_mask.any():
                abs_err_small = abs_err[small_mask]
                if abs_err_small.numel() > 0:
                    acc["abs_err_small_max"] = max(acc["abs_err_small_max"], float(abs_err_small.max().item()))
                    # false positive = our |tail| exceeds tau_abs where ref ~ 0
                    fp_mask = (T_tail_omega.abs() > tau_abs) & small_mask
                    acc["fp_small_count"] += int(fp_mask.sum().item())
                    acc["small_count"]     += int(small_mask.sum().item())

            # SNR = 10*log10( ||ref_tail||^2 / ||delta||^2 )
            acc["snr_num"] += float((T_ref_tail.abs() ** 2).sum().item())
            acc["snr_den"] += float((delta.abs()     ** 2).sum().item())

            acc["covered_samples"] += int(T_ref_tail.numel())

            if (it % 12) == 0 or it == args.iters:
                mean_rel_err = acc["rel_err_sum"] / max(1, acc["rel_err_count"])
                mean_abs_err = acc["abs_err_sum"] / max(1, acc["abs_err_count"])
                snr_db = 10.0 * math.log10(acc["snr_num"] / max(1e-300, acc["snr_den"])) if acc["snr_den"] > 0 else float("inf")
                fp_rate_small = (acc["fp_small_count"] / max(1, acc["small_count"])) if acc["small_count"] > 0 else 0.0
                print(
                    f"[{it}/{args.iters}] "
                    f"mean_rel_err={mean_rel_err:.3e}, max_rel_err={acc['rel_err_max']:.3e}, "
                    f"mean_abs_err={mean_abs_err:.3e}, max_abs_err={acc['abs_err_max']:.3e}, "
                    f"max_abs_err_small={acc['abs_err_small_max']:.3e}, fp_rate_small={fp_rate_small:.3e}, "
                    f"snr_db={snr_db:.2f}"
                )

        # final summary
        mean_rel_err = acc["rel_err_sum"] / max(1, acc["rel_err_count"])
        mean_abs_err = acc["abs_err_sum"] / max(1, acc["abs_err_count"])
        snr_db = 10.0 * math.log10(acc["snr_num"] / max(1e-300, acc["snr_den"])) if acc["snr_den"] > 0 else float("inf")
        fp_rate_small = (acc["fp_small_count"] / max(1, acc["small_count"])) if acc["small_count"] > 0 else 0.0

        print("\n=== Tail-to-tail validation summary ===")
        print(
            f"ref={args.ref}, device={dev.type}, dtype_z={dtype_z}, dtype_T={dtype_T}\n"
            f"B={args.B}, iters={args.iters}, N={args.N}, M={args.M}, S={args.S}, W={args.W}, O={O}\n"
            f"q_theta={args.q_theta}, q_rad={args.q_rad}, guards={args.guards}, t_PD={args.t_PD if args.ref=='pd' else 'n/a'}\n"
            f"tau_abs={tau_abs:.3e}, tau_rel={tau_rel:.3e}, covered_samples={acc['covered_samples']}\n"
            f"mean_rel_err={mean_rel_err:.6e}, max_rel_err={acc['rel_err_max']:.6e}, "
            f"mean_abs_err={mean_abs_err:.6e}, max_abs_err={acc['abs_err_max']:.6e}, "
            f"max_abs_err_small={acc['abs_err_small_max']:.6e}, fp_rate_small={fp_rate_small:.6f}, "
            f"snr_db={snr_db:.2f}"
        )


if __name__ == "__main__":
    main()
