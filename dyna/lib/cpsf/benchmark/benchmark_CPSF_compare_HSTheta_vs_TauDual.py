# dyna/lib/cpsf/benchmark/benchmark_CPSF_compare_HSTheta_vs_TauDual.py
# Run examples:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_compare_HSTheta_vs_TauDual --N 4 --M 128 --S 64 --batch 16 --dtype c64 --device cpu --K 7 --quad_nodes 12

import argparse
import math
import torch

from ..functional.core_math import Tau_dual
from ..functional.t_hs_theta import T_HS_theta


def _real_dtype_of(cdtype: torch.dtype) -> torch.dtype:
    return torch.float32 if cdtype == torch.complex64 else torch.float64


def _make_unit_batch(B: int, N: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    REAL = _real_dtype_of(dtype)
    xr = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    xi = torch.randn(B, N, generator=g, device=device, dtype=REAL)
    v = (xr + 1j * xi).to(dtype)
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    n = torch.where(n.real == 0, torch.ones_like(n), n)
    return v / n


def _make_complex(shape, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    REAL = _real_dtype_of(dtype)
    xr = torch.randn(*shape, generator=g, device=device, dtype=REAL)
    xi = torch.randn(*shape, generator=g, device=device, dtype=REAL)
    return (xr + 1j * xi).to(dtype)


def _k_cube(K: int, N: int, device: torch.device) -> torch.Tensor:
    if K < 0:
        return torch.zeros(1, N, device=device, dtype=torch.int64)
    rng = torch.arange(-K, K + 1, device=device, dtype=torch.int64)
    grids = torch.meshgrid(*([rng] * N), indexing="ij")
    k = torch.stack(grids, dim=-1).reshape(-1, N)
    return k


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    # HS-Theta params
    ap.add_argument("--quad_nodes", type=int, default=12, choices=[8, 12, 16])
    ap.add_argument("--theta_mode", type=str, default="auto", choices=["auto", "direct", "poisson"])
    ap.add_argument("--eps_total", type=float, default=1.0e-3)
    ap.add_argument("--a_threshold", type=float, default=1.0)
    ap.add_argument("--n_chunk", type=int, default=64)
    ap.add_argument("--m_chunk", type=int, default=65536)
    # misc
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("WARN: CUDA not available, falling back to CPU.")
            dev = torch.device("cpu")

    CDTYPE = torch.complex64 if args.dtype == "c64" else torch.complex128
    REAL = _real_dtype_of(CDTYPE)

    N, M, S, B, K = args.N, args.M, args.S, args.batch, args.K
    if N < 1 or M < 1 or S < 1 or B < 1:
        raise SystemExit("Invalid sizes.")
    if (2 * K + 1) ** N > 2_000_000:
        print(f"WARNING: k-grid size {(2*K+1)**N:,} is large; consider reducing N or K.")

    print(f"Device={dev.type}, dtype={CDTYPE}, N={N}, M={M}, S={S}, B={B}, K={K}")
    print(f"HS-Theta: quad_nodes={args.quad_nodes}, theta_mode={args.theta_mode}, eps_total={args.eps_total}")

    z = _make_complex((B, N), CDTYPE, dev, seed=args.seed + 1)
    vec_d = _make_unit_batch(B, N, CDTYPE, dev, seed=args.seed + 2)

    z_j = _make_complex((M, N), CDTYPE, dev, seed=args.seed + 3)
    vec_d_j = _make_unit_batch(M, N, CDTYPE, dev, seed=args.seed + 4)
    T_hat_j = _make_complex((M, S), CDTYPE, dev, seed=args.seed + 5)

    g_alpha = torch.Generator(device=dev).manual_seed(args.seed + 6)
    alpha_j = torch.rand(M, generator=g_alpha, device=dev, dtype=REAL)

    sp = torch.empty(M, device=dev, dtype=REAL)
    sq = torch.empty(M, device=dev, dtype=REAL)
    g_sig = torch.Generator(device=dev).manual_seed(args.seed + 7)
    sq.uniform_(0.4, 1.2, generator=g_sig)
    sp.uniform_(1.0, 2.0, generator=g_sig)
    sp = torch.maximum(sp, sq + 1e-3)

    # HS-Theta
    T_hs = T_HS_theta(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par=sp,
        sigma_perp=sq,
        quad_nodes=args.quad_nodes,
        theta_mode=args.theta_mode,
        eps_total=args.eps_total,
        a_threshold=args.a_threshold,
        n_chunk=args.n_chunk,
        m_chunk=args.m_chunk,
        dtype_override=CDTYPE,
    )

    # Tau_dual
    z_bm = z.unsqueeze(-2).expand(B, M, N)
    z_j_bm = z_j.unsqueeze(0).expand(B, M, N)
    vec_d_bm = vec_d.unsqueeze(-2).expand(B, M, N)
    vec_d_j_bm = vec_d_j.unsqueeze(0).expand(B, M, N)
    T_hat_b = T_hat_j.unsqueeze(0)
    alpha_b = alpha_j.unsqueeze(0)
    sp_b = sp.unsqueeze(0)
    sq_b = sq.unsqueeze(0)
    k = _k_cube(K, N, dev)
    T_dual = Tau_dual(
        z=z_bm,
        z_j=z_j_bm,
        vec_d=vec_d_bm,
        vec_d_j=vec_d_j_bm,
        T_hat_j=T_hat_b,
        alpha_j=alpha_b,
        sigma_par=sp_b,
        sigma_perp=sq_b,
        k=k,
        R_j=None,
    )

    eps = torch.as_tensor(1e-12, device=dev, dtype=REAL)
    diff = T_hs - T_dual
    l2_dual = torch.linalg.vector_norm(T_dual, dim=-1)
    l2_diff = torch.linalg.vector_norm(diff, dim=-1)
    rel_l2 = (l2_diff / torch.clamp(l2_dual, min=eps)).to(REAL)
    mae_elem = diff.abs().to(REAL).mean()
    mse_elem = (diff.abs().to(REAL) ** 2).mean()
    rel_elem = (diff.abs() / torch.clamp(T_dual.abs(), min=eps)).to(REAL)
    rel_elem_mean = rel_elem.mean()
    rel_elem_p95 = torch.quantile(rel_elem.flatten(), 0.95)
    rel_l2_mean = rel_l2.mean()
    rel_l2_median = torch.median(rel_l2)
    rel_l2_p95 = torch.quantile(rel_l2, 0.95)
    rel_l2_max = rel_l2.max()

    # после вычисления T_hs и T_dual:
    num = (T_dual.conj() * T_hs).sum(dim=-1).real  # (B,)
    den = (T_dual.conj() * T_dual).sum(dim=-1).real + 1e-30
    s_star_per = (num / den).clamp_min(1e-12)
    s_star = s_star_per.mean()  # глобальная оценка масштаба
    T_hs_cal = s_star * T_hs
    diff_cal = T_hs_cal - T_dual
    rel_l2_cal = torch.linalg.vector_norm(diff_cal, dim=-1) / torch.clamp(torch.linalg.vector_norm(T_dual, dim=-1), min=1e-12)
    print(f"\nScale diagnostic: s*={s_star.item():.8f}, "
        f"var={s_star_per.var().sqrt().item():.3e}, "
        f"relL2(mean) after scale={rel_l2_cal.mean().item():.6e}")


    print("\n=== Accuracy: T_HS_theta vs Tau_dual ===")
    print(f"Per-sample relative L2: mean={rel_l2_mean:.6e}, median={rel_l2_median:.6e}, p95={rel_l2_p95:.6e}, max={rel_l2_max:.6e}")
    print(f"Elementwise: MAE={mae_elem:.6e}, MSE={mse_elem:.6e}, mean(|Δ|/|T_dual|)={rel_elem_mean:.6e}, p95(|Δ|/|T_dual|)={rel_elem_p95:.6e}")

    if torch.isnan(rel_l2_mean) or torch.isinf(rel_l2_mean):
        raise SystemExit("Numerical instability detected.")

    # Optional quick consistency assertion for typical small N,K
    if N <= 4 and K <= 5:
        assert rel_l2_mean < 1e-2, "Mean relative L2 seems high; adjust HS-Theta params or investigate."


if __name__ == "__main__":
    main()
