# run:
# > python -m dyna.lib.cpsf.manual_tests.test_PHC_tile_invariance --device cuda --dtype c64 --B 32 --N 8 --M 128 --S 64 --quad_nodes 7 --eps_total 1.0e-3 --seed 1337

import argparse, torch

from ..functional.t_phc_fused import T_PHC_Fused
from ..functional.t_phc_batched import T_PHC_Batched


def real_dtype_of(cdtype: torch.dtype) -> torch.dtype:
    return torch.float32 if cdtype == torch.complex64 else torch.float64


def make_unit(B, N, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    R = real_dtype_of(dtype)
    xr = torch.randn(B, N, generator=g, device=device, dtype=R)
    xi = torch.randn(B, N, generator=g, device=device, dtype=R)
    v = (xr + 1j * xi).to(dtype)
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    n = torch.where(n.real == 0, torch.ones_like(n), n)
    return v / n


def make_unit_bmn(B, M, N, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    R = real_dtype_of(dtype)
    xr = torch.randn(B, M, N, generator=g, device=device, dtype=R)
    xi = torch.randn(B, M, N, generator=g, device=device, dtype=R)
    v = (xr + 1j * xi).to(dtype)
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    n = torch.where(n.real == 0, torch.ones_like(n), n)
    return v / n


def make_complex(shape, dtype, device, seed, scale_im=1.0):
    g = torch.Generator(device=device).manual_seed(seed)
    R = real_dtype_of(dtype)
    xr = torch.randn(*shape, generator=g, device=device, dtype=R)
    xi = torch.randn(*shape, generator=g, device=device, dtype=R) * float(scale_im)
    return (xr + 1j * xi).to(dtype)


def rel_l2_max(A: torch.Tensor, B: torch.Tensor) -> float:
    num = torch.linalg.vector_norm(A - B, dim=-1)
    den = torch.linalg.vector_norm(A, dim=-1).clamp_min(1e-30)
    return float((num / den).max())


def thresholds(dtype: torch.dtype):
    return 2e-6 if dtype == torch.complex64 else 5e-12


@torch.no_grad()
def run_once(
    dev,
    dtype,
    B,
    N,
    M,
    S,
    n_chunk_a,
    n_chunk_b,
    m_chunk_a,
    m_chunk_b,
    quad_nodes,
    eps_total,
    scale_im,
    seed,
):
    R = real_dtype_of(dtype)

    z = make_complex((B, N), dtype, dev, seed + 1, scale_im=scale_im)
    vec_d = make_unit(B, N, dtype, dev, seed + 2)

    z_j_f = make_complex((M, N), dtype, dev, seed + 3, scale_im=scale_im)
    vec_d_j_f = make_unit(M, N, dtype, dev, seed + 4)
    T_hat_j_f = make_complex((M, S), dtype, dev, seed + 5)
    g_alpha = torch.Generator(device=dev).manual_seed(seed + 6)
    alpha_j_f = torch.rand(M, generator=g_alpha, device=dev, dtype=R)

    g_sig = torch.Generator(device=dev).manual_seed(seed + 7)
    sigma_perp_f = torch.empty(M, device=dev, dtype=R).uniform_(
        0.4, 1.2, generator=g_sig
    )
    sigma_par_f = torch.empty(M, device=dev, dtype=R).uniform_(
        1.0, 2.0, generator=g_sig
    )
    sigma_par_f = torch.maximum(sigma_par_f, sigma_perp_f + 1.0e-3)

    z_j_b = z_j_f.unsqueeze(0).expand(B, M, N).contiguous()
    vec_d_j_b = vec_d_j_f.unsqueeze(0).expand(B, M, N).contiguous()
    T_hat_j_b = T_hat_j_f.unsqueeze(0).expand(B, M, S).contiguous()
    alpha_j_b = alpha_j_f.unsqueeze(0).expand(B, M).contiguous()
    sigma_perp_b = sigma_perp_f.unsqueeze(0).expand(B, M).contiguous()
    sigma_par_b = sigma_par_f.unsqueeze(0).expand(B, M).contiguous()

    T_fused_A = T_PHC_Fused(
        z=z,
        vec_d=vec_d,
        z_j=z_j_f,
        vec_d_j=vec_d_j_f,
        T_hat_j=T_hat_j_f,
        alpha_j=alpha_j_f,
        sigma_par_j=sigma_par_f,
        sigma_perp_j=sigma_perp_f,
        quad_nodes=quad_nodes,
        eps_total=eps_total,
        n_chunk=n_chunk_a,
        m_chunk=m_chunk_a,
        dtype_override=dtype,
    )
    T_fused_B = T_PHC_Fused(
        z=z,
        vec_d=vec_d,
        z_j=z_j_f,
        vec_d_j=vec_d_j_f,
        T_hat_j=T_hat_j_f,
        alpha_j=alpha_j_f,
        sigma_par_j=sigma_par_f,
        sigma_perp_j=sigma_perp_f,
        quad_nodes=quad_nodes,
        eps_total=eps_total,
        n_chunk=n_chunk_a,
        m_chunk=m_chunk_b,
        dtype_override=dtype,
    )
    thr = thresholds(dtype)
    r1 = rel_l2_max(T_fused_A, T_fused_B)
    print(f"[fused]   m_chunk invariance (scale_im={scale_im}): max rel L2 = {r1:.3e}")
    assert r1 < thr, f"Fused m_chunk invariance failed: {r1:.3e} >= {thr:.1e}"

    T_fused_C = T_PHC_Fused(
        z=z,
        vec_d=vec_d,
        z_j=z_j_f,
        vec_d_j=vec_d_j_f,
        T_hat_j=T_hat_j_f,
        alpha_j=alpha_j_f,
        sigma_par_j=sigma_par_f,
        sigma_perp_j=sigma_perp_f,
        quad_nodes=quad_nodes,
        eps_total=eps_total,
        n_chunk=n_chunk_b,
        m_chunk=m_chunk_a,
        dtype_override=dtype,
    )
    r2 = rel_l2_max(T_fused_A, T_fused_C)
    print(f"[fused]   n_chunk invariance (scale_im={scale_im}): max rel L2 = {r2:.3e}")
    assert r2 < thr, f"Fused n_chunk invariance failed: {r2:.3e} >= {thr:.1e}"

    T_batched_A = T_PHC_Batched(
        z=z,
        vec_d=vec_d,
        z_j=z_j_b,
        vec_d_j=vec_d_j_b,
        T_hat_j=T_hat_j_b,
        alpha_j=alpha_j_b,
        sigma_par_j=sigma_par_b,
        sigma_perp_j=sigma_perp_b,
        quad_nodes=quad_nodes,
        eps_total=eps_total,
        n_chunk=n_chunk_a,
        m_chunk=m_chunk_a,
        dtype_override=dtype,
    )
    T_batched_B = T_PHC_Batched(
        z=z,
        vec_d=vec_d,
        z_j=z_j_b,
        vec_d_j=vec_d_j_b,
        T_hat_j=T_hat_j_b,
        alpha_j=alpha_j_b,
        sigma_par_j=sigma_par_b,
        sigma_perp_j=sigma_perp_b,
        quad_nodes=quad_nodes,
        eps_total=eps_total,
        n_chunk=n_chunk_a,
        m_chunk=m_chunk_b,
        dtype_override=dtype,
    )
    r3 = rel_l2_max(T_batched_A, T_batched_B)
    print(f"[batched] m_chunk invariance (scale_im={scale_im}): max rel L2 = {r3:.3e}")
    assert r3 < thr, f"Batched m_chunk invariance failed: {r3:.3e} >= {thr:.1e}"

    T_batched_C = T_PHC_Batched(
        z=z,
        vec_d=vec_d,
        z_j=z_j_b,
        vec_d_j=vec_d_j_b,
        T_hat_j=T_hat_j_b,
        alpha_j=alpha_j_b,
        sigma_par_j=sigma_par_b,
        sigma_perp_j=sigma_perp_b,
        quad_nodes=quad_nodes,
        eps_total=eps_total,
        n_chunk=n_chunk_b,
        m_chunk=m_chunk_a,
        dtype_override=dtype,
    )
    r4 = rel_l2_max(T_batched_A, T_batched_C)
    print(f"[batched] n_chunk invariance (scale_im={scale_im}): max rel L2 = {r4:.3e}")
    assert r4 < thr, f"Batched n_chunk invariance failed: {r4:.3e} >= {thr:.1e}"

    r5 = rel_l2_max(T_fused_A, T_batched_A)
    print(f"[fused vs batched] equality (scale_im={scale_im}): max rel L2 = {r5:.3e}")
    assert r5 < thr, f"Fused vs Batched mismatch: {r5:.3e} >= {thr:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--quad_nodes", type=int, default=7)
    ap.add_argument("--eps_total", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    dev = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    dtype = torch.complex64 if args.dtype == "c64" else torch.complex128

    n_chunk_a = args.N
    n_chunk_b = max(1, args.N // 3)
    m_chunk_a = args.M
    m_chunk_b = max(1, args.M // 4)

    if dev.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"Device={dev.type}, dtype={dtype}, B={args.B}, N={args.N}, M={args.M}, S={args.S}"
    )
    print(f"quad_nodes={args.quad_nodes}, eps_total={args.eps_total}")
    print(f"n_chunk: {n_chunk_a} vs {n_chunk_b} | m_chunk: {m_chunk_a} vs {m_chunk_b}")

    run_once(
        dev,
        dtype,
        args.B,
        args.N,
        args.M,
        args.S,
        n_chunk_a,
        n_chunk_b,
        m_chunk_a,
        m_chunk_b,
        args.quad_nodes,
        args.eps_total,
        scale_im=0.0,
        seed=args.seed,
    )

    run_once(
        dev,
        dtype,
        args.B,
        args.N,
        args.M,
        args.S,
        n_chunk_a,
        n_chunk_b,
        m_chunk_a,
        m_chunk_b,
        args.quad_nodes,
        args.eps_total,
        scale_im=1.0,
        seed=args.seed + 1000,
    )

    print("\nALL TESTS PASSED ✔")


if __name__ == "__main__":
    main()
