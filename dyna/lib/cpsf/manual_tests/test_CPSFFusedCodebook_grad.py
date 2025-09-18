# run:
# > python -m dyna.lib.cpsf.manual_tests.test_CPSFFusedCodebook_grad --device cuda --dtype c64 --B 1 --N 16 --M 128 --S 128 --quad_nodes 7 --eps_total 1.0e-3

import argparse, torch

from dyna.lib.cpsf.fused_codebook import CPSFFusedCodebook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--quad_nodes", type=int, default=7)
    ap.add_argument("--eps_total", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    dtype = torch.complex64 if args.dtype == "c64" else torch.complex128
    B = args.B
    N = args.N
    M = args.M
    S = args.S
    quad_nodes = args.quad_nodes
    eps_total = args.eps_total

    print(f"Device={device.type}, dtype={dtype}, B={B}, N={N}, M={M}, S={S}")
    print(f"quad_nodes={quad_nodes}, eps_total={eps_total}")

    codebook = CPSFFusedCodebook(
        N=N,
        M=M,
        S=S,
        quad_nodes=quad_nodes,
        n_chunk=N,
        m_chunk=M,
        autonorm_vec_d=True,
        autonorm_vec_d_j=True,
        eps_total=eps_total,
        c_dtype=dtype,
    ).to(device=device)

    z = torch.complex(
        real=torch.empty([B, N], dtype=torch.float32).uniform_(-0.5, +0.5),
        imag=torch.empty([B, N], dtype=torch.float32).uniform_(-0.5, +0.5),
    ).to(device=device, dtype=dtype)
    vec_d = torch.complex(
        real=torch.empty([B, N], dtype=torch.float32).uniform_(-0.5, +0.5),
        imag=torch.empty([B, N], dtype=torch.float32).uniform_(-0.5, +0.5),
    ).to(device=device, dtype=dtype)

    x = codebook(z, vec_d)

    print("x result:")
    print(f"\t{x.real.min()=}")
    print(f"\t{x.real.max()=}")
    print(f"\t{x.real.abs().min()=}")
    print(f"\t{x.real.abs().max()=}")
    print(f"\t{x.real.std()=}")

    xr = x
    yr = torch.randn_like(xr, device=device, dtype=dtype).detach()

    l = ((xr - yr) ** 2).mean().real
    l.backward()

    for name, param in codebook.named_parameters():
        if param.grad is not None:
            if torch.is_complex(param.grad):
                print(
                    "".join(
                        [
                            "\n",
                            f"\nGrads for '{name}' - complex:",
                            f"\n\tStd: {param.grad.std()}",
                            f"\n\tMin (real): {param.grad.real.min()}",
                            f"\n\tMax (real): {param.grad.real.max()}",
                            f"\n\tAbsMin (real): {param.grad.real.abs().min()}",
                            f"\n\tAbsMax (real): {param.grad.real.abs().max()}",
                            f"\n\tMin (imag): {param.grad.imag.min()}",
                            f"\n\tMax (imag): {param.grad.imag.max()}",
                            f"\n\tAbsMin (imag): {param.grad.imag.abs().min()}",
                            f"\n\tAbsMax (imag): {param.grad.imag.abs().max()}",
                        ]
                    )
                )
            else:
                print(
                    "".join(
                        [
                            "\n",
                            f"\nGrads for '{name}' - real:",
                            f"\n\tStd: {param.grad.std()}",
                            f"\n\tMin: {param.grad.min()}",
                            f"\n\tMax: {param.grad.max()}",
                            f"\n\tAbsMin: {param.grad.abs().min()}",
                            f"\n\tAbsMax: {param.grad.abs().max()}",
                        ]
                    )
                )
        else:
            print(f"Param '{name}' has no grad.")


if __name__ == "__main__":
    main()
