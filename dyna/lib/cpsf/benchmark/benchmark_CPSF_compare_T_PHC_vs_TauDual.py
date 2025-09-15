# dyna/lib/cpsf/benchmark/benchmark_CPSF_compare_T_PHC_vs_TauDual.py
# Run examples:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_compare_T_PHC_vs_TauDual --N 4 --M 128 --S 64 --batch 16 --dtype c64 --device cpu --K 7 --quad_nodes 7 --phase_scale 1.0 --eps_total 1.0e-3 --seed 1337

import argparse
import torch

from ..functional.core_math import Tau_dual
from ..functional.t_phc import T_PHC


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
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    # HS-Theta params
    ap.add_argument("--quad_nodes", type=int, default=7)
    ap.add_argument("--eps_total", type=float, default=1.0e-3)
    ap.add_argument("--n_chunk", type=int, default=256)
    ap.add_argument("--m_chunk", type=int, default=256)
    # misc
    ap.add_argument("--phase_scale", type=float, default=0.0)
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
        print(
            f"WARNING: k-grid size {(2*K+1)**N:,} is large; consider reducing N or K."
        )

    print(f"Device={dev.type}, dtype={CDTYPE}, N={N}, M={M}, S={S}, B={B}, K={K}")
    print(f"HS-Theta: quad_nodes={args.quad_nodes}, eps_total={args.eps_total}")

    z = _make_complex((B, N), CDTYPE, dev, seed=args.seed + 1)
    z.imag = z.imag * args.phase_scale
    vec_d = _make_unit_batch(B, N, CDTYPE, dev, seed=args.seed + 2)

    z_j = _make_complex((M, N), CDTYPE, dev, seed=args.seed + 3)
    z_j.imag = z_j.imag * args.phase_scale
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
    T_hs = T_PHC(
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

    # ===== Diagnostics (scale, masking, cos-sim) =====
    def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    def _energy2(x_flat: torch.Tensor) -> torch.Tensor:
        # squared L2 per sample
        return (x_flat.real**2 + x_flat.imag**2).sum(dim=1)

    def _safe_stats(x: torch.Tensor, name: str):
        if x.numel() == 0:
            print(f"{name}: <no data>")
            return
        q95 = torch.quantile(x, torch.tensor(0.95, device=x.device))
        print(
            f"{name}: mean={x.mean().item():.6e}, median={x.median().item():.6e}, p95={q95.item():.6e}, max={x.max().item():.6e}"
        )

    REAL = _real_dtype_of(CDTYPE)

    td = _flatten_batch(T_dual)
    th = _flatten_batch(T_hs)

    e2 = _energy2(td)
    # Energy threshold: small but dtype-aware
    energy_tau = 1.0e-12 if REAL == torch.float32 else 1.0e-24
    mask = e2 > energy_tau
    num_live = int(mask.sum().item())
    num_total = int(e2.numel())

    print("\nEnergy diagnostic (||T_dual|| per-sample):")
    _safe_stats(e2.sqrt(), "||T_dual||_2")
    print(f"Live samples (>tau={energy_tau:g}): {num_live}/{num_total}")

    if num_live == 0:
        print(
            "".join(
                [
                    "\n\t=== ENERGY COLLAPSE DETECTED ===",
                    "\n\t=== Zero live samples.       ===",
                ]
            )
        )
    else:
        # Global scale (energy-weighted)
        num = (td.conj() * th).sum().real
        den = (td.abs() ** 2).sum().real
        s_star_global = (num / den).item() if den > 0 else float("nan")

        # Cosine similarity (global, scale-invariant)
        norm_td = den.sqrt()
        norm_th = ((th.abs() ** 2).sum().real).sqrt()
        cos_global = (
            (num.abs() / (norm_td * norm_th + 1e-30)).item()
            if (norm_td > 0 and norm_th > 0)
            else float("nan")
        )
        cos_global = min(cos_global, 1.0)

        print("\nScale & alignment:")
        print(f"s*_global={s_star_global:.8f}")
        print(f"cos(angle) (global)={cos_global:.12f}")

    # Per-sample relative L2 (masked by energy)
    diff = th - td
    rel_l2 = diff.abs().pow(2).sum(dim=1).sqrt()
    den_l2 = e2.sqrt()
    # no energy = NaN... avoid statistics corruption
    rel_l2 = torch.where(
        mask,
        rel_l2 / den_l2.clamp_min(1e-30),
        torch.tensor(float("nan"), device=rel_l2.device, dtype=rel_l2.dtype),
    )
    rel_l2_masked = rel_l2[mask]

    print("\n=== Accuracy: T_HS_theta vs Tau_dual (masked by energy) ===")
    if rel_l2_masked.numel() == 0:
        print("Per-sample relative L2: <no live samples>")
        rel_l2_mean = torch.tensor(0.0, dtype=REAL, device=td.device)
    else:
        q95 = torch.quantile(
            rel_l2_masked, torch.tensor(0.95, device=rel_l2_masked.device)
        )
        print(
            f"Per-sample relative L2: mean={rel_l2_masked.mean().item():.6e}, median={rel_l2_masked.median().item():.6e}, p95={q95.item():.6e}, max={rel_l2_masked.max().item():.6e}"
        )
        rel_l2_mean = rel_l2_masked.mean()

    # Elementwise metrics (avoid div-by-zero by tiny denom)
    abs_diff = (th - td).abs()
    abs_td = td.abs()
    elem_mae = abs_diff.mean()
    elem_mse = (abs_diff**2).mean()
    elem_rel = abs_diff / (abs_td + 1e-30)
    q95_elem_rel = torch.quantile(elem_rel, torch.tensor(0.95, device=elem_rel.device))
    print(
        f"Elementwise: MAE={elem_mae.item():.6e}, MSE={elem_mse.item():.6e}, mean(|Δ|/|T_dual|)={elem_rel.mean().item():.6e}, p95(|Δ|/|T_dual|)={q95_elem_rel.item():.6e}"
    )

    if torch.isnan(rel_l2_mean) or torch.isinf(rel_l2_mean):
        raise SystemExit("Numerical instability detected.")

    # Optional quick consistency assertion for typical small N,K
    if N <= 4 and K <= 5:
        assert (
            rel_l2_mean < 1e-2
        ), "Mean relative L2 seems high; adjust HS-Theta params or investigate."


if __name__ == "__main__":
    main()
