# dyna/lib/cpsf/benchmark/benchmark_CPSF_compare_T_PHC_Fused_vs_T_PHC_Batched.py
# Run examples:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_compare_T_PHC_Fused_vs_T_PHC_Batched --N 4 --M 128 --S 256 --batch 32 --dtype c64 --device cuda --quad_nodes 7 --eps_total 1.0e-3 --phase_scale 1.0 --seed 1337

import argparse
import torch

from ..functional.t_phc_fused import T_PHC_Fused
from ..functional.t_phc_batched import T_PHC_Batched


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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    # PHC params
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

    N, M, S, B = args.N, args.M, args.S, args.batch
    if N < 1 or M < 1 or S < 1 or B < 1:
        raise SystemExit("Invalid sizes.")

    print(f"Device={dev.type}, dtype={CDTYPE}, N={N}, M={M}, S={S}, B={B}")
    print(f"PHC: quad_nodes={args.quad_nodes}, eps_total={args.eps_total}")

    # Queries
    z = _make_complex((B, N), CDTYPE, dev, seed=args.seed + 1)
    z.imag = z.imag * args.phase_scale
    vec_d = _make_unit_batch(B, N, CDTYPE, dev, seed=args.seed + 2)

    # Per-query pools (B,M,...)
    z_j = _make_complex((B, M, N), CDTYPE, dev, seed=args.seed + 3)
    z_j.imag = z_j.imag * args.phase_scale

    g_vdj = torch.Generator(device=dev).manual_seed(args.seed + 4)
    xr = torch.randn(B, M, N, generator=g_vdj, device=dev, dtype=REAL)
    xi = torch.randn(B, M, N, generator=g_vdj, device=dev, dtype=REAL)
    vec_d_j = (xr + 1j * xi).to(CDTYPE)
    norm_vdj = torch.linalg.vector_norm(vec_d_j, dim=-1, keepdim=True)
    norm_vdj = torch.where(norm_vdj.real == 0, torch.ones_like(norm_vdj), norm_vdj)
    vec_d_j = vec_d_j / norm_vdj

    T_hat_j = _make_complex((B, M, S), CDTYPE, dev, seed=args.seed + 5)

    g_alpha = torch.Generator(device=dev).manual_seed(args.seed + 6)
    alpha_j = torch.rand(B, M, generator=g_alpha, device=dev, dtype=REAL)

    sp = torch.empty(B, M, device=dev, dtype=REAL)
    sq = torch.empty(B, M, device=dev, dtype=REAL)
    g_sig = torch.Generator(device=dev).manual_seed(args.seed + 7)
    sq.uniform_(0.4, 1.2, generator=g_sig)
    sp.uniform_(1.0, 2.0, generator=g_sig)
    sp = torch.maximum(sp, sq + 1e-3)

    # === Test: batched evaluator on [B,M,...]
    T_phc_batched = T_PHC_Batched(
        z=z,
        vec_d=vec_d,
        z_j=z_j,
        vec_d_j=vec_d_j,
        T_hat_j=T_hat_j,
        alpha_j=alpha_j,
        sigma_par_j=sp,
        sigma_perp_j=sq,
        quad_nodes=args.quad_nodes,
        eps_total=args.eps_total,
        n_chunk=args.n_chunk,
        m_chunk=args.m_chunk,
        dtype_override=CDTYPE,
    )

    print(
        "".join([
            f"\nT_PHC_Batched diag:",
            f"\n\tStdDev:   \t{T_phc_batched.std().item()}",
            f"\n\tVariance: \t{T_phc_batched.var().item()}",
            f"\n\tMean:     \t{T_phc_batched.mean().item()}",
            f"\n\tAbsMean:  \t{T_phc_batched.abs().mean().item()}",
            f"\n\tAbsMin:   \t{T_phc_batched.abs().min().item()}",
            f"\n\tAbsMax:   \t{T_phc_batched.abs().max().item()}",
        ])
    )

    # === Reference: emulate "point -> own pool" by calling T_PHC_Fused() B times with B=1
    T_phc_fused_list = []
    for b in range(B):
        Tb = T_PHC_Fused(
            z=z[b : b + 1],  # [1,N]
            vec_d=vec_d[b : b + 1],  # [1,N]
            z_j=z_j[b],  # [M,N]
            vec_d_j=vec_d_j[b],  # [M,N]
            T_hat_j=T_hat_j[b],  # [M,S]
            alpha_j=alpha_j[b],  # [M]
            sigma_par_j=sp[b],  # [M]
            sigma_perp_j=sq[b],  # [M]
            quad_nodes=args.quad_nodes,
            eps_total=args.eps_total,
            n_chunk=args.n_chunk,
            m_chunk=args.m_chunk,
            dtype_override=CDTYPE,
        )  # -> [1,S]
        T_phc_fused_list.append(Tb)
    T_phc_fused = torch.cat(T_phc_fused_list, dim=0)  # [B,S]

    # ===== Diagnostics & accuracy =====
    def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    def _energy2(x_flat: torch.Tensor) -> torch.Tensor:
        return (x_flat.real**2 + x_flat.imag**2).sum(dim=1)

    def _safe_stats(x: torch.Tensor, name: str):
        if x.numel() == 0:
            print(f"{name}: <no data>")
            return
        q95 = torch.quantile(x, torch.tensor(0.95, device=x.device))
        print(
            f"{name}: mean={x.mean().item():.6e}, median={x.median().item():.6e}, p95={q95.item():.6e}, max={x.max().item():.6e}"
        )

    tref = _flatten_batch(T_phc_fused)
    tbat = _flatten_batch(T_phc_batched)

    e2 = _energy2(tref)
    energy_tau = 1.0e-12 if REAL == torch.float32 else 1.0e-24
    mask = e2 > energy_tau
    num_live = int(mask.sum().item())
    num_total = int(e2.numel())

    print("\nEnergy diagnostic (||T_ref|| per-sample):")
    _safe_stats(e2.sqrt(), "||T_ref||_2")
    print(f"Live samples (>tau={energy_tau:g}): {num_live}/{num_total}")

    diff = tbat - tref
    rel_l2 = diff.abs().pow(2).sum(dim=1).sqrt()
    den_l2 = e2.sqrt()
    rel_l2 = torch.where(
        mask,
        rel_l2 / den_l2.clamp_min(1e-30),
        torch.tensor(float("nan"), device=rel_l2.device, dtype=rel_l2.dtype),
    )
    rel_l2_masked = rel_l2[mask]

    print("\n=== Accuracy: T_PHC_Batched vs T_PHC_Fused (masked by energy) ===")
    if rel_l2_masked.numel() == 0:
        print("Per-sample relative L2: <no live samples>")
        rel_l2_mean = torch.tensor(0.0, dtype=REAL, device=tref.device)
    else:
        q95 = torch.quantile(
            rel_l2_masked, torch.tensor(0.95, device=rel_l2_masked.device)
        )
        print(
            f"Per-sample relative L2: mean={rel_l2_masked.mean().item():.6e}, median={rel_l2_masked.median().item():.6e}, p95={q95.item():.6e}, max={rel_l2_masked.max().item():.6e}"
        )
        rel_l2_mean = rel_l2_masked.mean()

    # Elementwise metrics
    abs_diff = (tbat - tref).abs()
    abs_ref = tref.abs()
    elem_mae = abs_diff.mean()
    elem_mse = (abs_diff**2).mean()
    elem_rel = abs_diff / (abs_ref + 1e-30)
    q95_elem_rel = torch.quantile(elem_rel, torch.tensor(0.95, device=elem_rel.device))
    print(
        f"Elementwise: MAE={elem_mae.item():.6e}, MSE={elem_mse.item():.6e}, mean(|Δ|/|T_ref|)={elem_rel.mean().item():.6e}, p95(|Δ|/|T_ref|)={q95_elem_rel.item():.6e}"
    )

    if torch.isnan(rel_l2_mean) or torch.isinf(rel_l2_mean):
        raise SystemExit("Numerical instability detected.")

    # Optional quick consistency assertion for typical small sizes
    if N <= 4:
        assert (
            rel_l2_mean < 1e-6
        ), "Mean relative L2 seems high; investigate batched path."


if __name__ == "__main__":
    main()
