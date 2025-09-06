import torch


def R(
    d: torch.Tensor,
    # smoothing & stability
    kappa: float = 1.0e-3,  # soft de-clumping of J_E weights
    sigma: float = 1.0e-3,  # smoothing for anchor mixing
    jitter: float = 1.0e-6,  # SPD jitter in the polar normalization
    # mixing profile
    p: float = 2.0,  # exponent for J_E weights
    qmix: float = 2.0,  # "temperature" of energy-based anchor mixing
) -> torch.Tensor:
    if d.dim() < 1:
        raise ValueError(f"R(d): expected [..., N], got {tuple(d.shape)}")
    if not torch.is_complex(d):
        raise TypeError(
            "R(d): CPSF canon expects a complex input (torch.complex64/torch.complex128)."
        )

    *B, N = d.shape
    dtype, device = d.dtype, d.device

    finfo = torch.finfo(d.real.dtype)
    tiny = torch.sqrt(torch.tensor(finfo.eps, dtype=d.real.dtype, device=device))

    dn = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
    b = d / torch.clamp(dn, min=tiny)

    if N == 1:
        return b.unsqueeze(-1)

    I = torch.eye(N, dtype=dtype, device=device)
    if B:
        I = I.expand(*B, N, N).clone()
    P_perp = I - b.unsqueeze(-1) @ b.conj().unsqueeze(-2)

    E = I[..., :, 1:]
    absb2 = (b.conj() * b).real
    wE_real = (
        1.0 - absb2[..., 1:] + torch.tensor(kappa, dtype=d.real.dtype, device=device)
    ) ** p
    DE = torch.diag_embed(wE_real.to(dtype))
    J_E = E @ DE

    reals = d.real.dtype
    i = torch.arange(N, device=device, dtype=reals).unsqueeze(-1)
    k = torch.arange(1, N, device=device, dtype=reals).unsqueeze(0)
    phase1 = 2.0 * torch.pi * i * k / float(N)
    phase2 = 2.0 * torch.pi * i * (k + 0.5) / float(N)
    J_F1 = torch.exp(1j * phase1).to(dtype)
    J_F2 = torch.exp(1j * phase2).to(dtype)
    if B:
        J_F1 = J_F1.expand(*B, N, N - 1).clone()
        J_F2 = J_F2.expand(*B, N, N - 1).clone()

    Z1 = P_perp @ J_E
    Z2 = P_perp @ J_F1
    Z3 = P_perp @ J_F2

    def frob2(Z: torch.Tensor) -> torch.Tensor:
        return (Z.conj() * Z).real.sum(dim=(-2, -1), keepdim=True)

    s1 = frob2(Z1)
    s2 = frob2(Z2)
    s3 = frob2(Z3)

    sig = torch.tensor(sigma, dtype=s1.dtype, device=device)
    a1 = (s1 + sig) ** qmix
    a2 = (s2 + sig) ** qmix
    a3 = (s3 + sig) ** qmix
    asum = a1 + a2 + a3
    a1 = (a1 / asum).to(dtype)
    a2 = (a2 / asum).to(dtype)
    a3 = (a3 / asum).to(dtype)

    Z = a1 * Z1 + a2 * Z2 + a3 * Z3

    H = Z.mH @ Z
    I_nm1 = torch.eye(N - 1, dtype=H.dtype, device=device)
    tr = H.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1, keepdim=True).unsqueeze(-1)
    scale = tr / float(N - 1)
    jitter_eff = (torch.tensor(jitter, dtype=H.real.dtype, device=device) + tiny) * (
        scale + 1.0
    )
    H_spd = H + jitter_eff.to(H.dtype) * I_nm1

    evals, U = torch.linalg.eigh(H_spd)
    inv_sqrt = (evals.clamp_min(tiny) ** -0.5).diag_embed().to(H_spd.dtype)
    UH = U.mH
    H_inv_half = U @ inv_sqrt @ UH
    Q_perp = Z @ H_inv_half

    Q_perp = P_perp @ Q_perp
    S = Q_perp.mH @ Q_perp
    evals2, U2 = torch.linalg.eigh(S)
    inv_sqrt2 = (evals2.clamp_min(tiny) ** -0.5).diag_embed().to(S.dtype)
    U2H = U2.mH
    S_inv_half = U2 @ inv_sqrt2 @ U2H
    Q_perp = Q_perp @ S_inv_half

    R_out = torch.cat([b.unsqueeze(-1), Q_perp], dim=-1)

    return R_out
