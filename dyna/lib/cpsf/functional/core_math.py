import torch


def R(
    d: torch.Tensor,
    kappa: float = 1.0e-3,
    sigma: float = 1.0e-3,
    jitter: float = 1.0e-6,
) -> torch.Tensor:
    if d.dim() < 1:
        raise ValueError(f"R(d): expected [..., N], got {tuple(d.shape)}")

    *Bsz, N = d.shape
    dtype, device = d.dtype, d.device
    is_c = torch.is_complex(d)

    finfo = torch.finfo(d.real.dtype) if is_c else torch.finfo(d.dtype)
    tiny = torch.sqrt(
        torch.tensor(finfo.eps, dtype=(d.real if is_c else d).dtype, device=device)
    )
    dn = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
    b = d / torch.clamp(dn, min=tiny)

    if N == 1:
        return b.unsqueeze(-1)

    I = torch.eye(N, dtype=dtype, device=device)
    if Bsz:
        I = I.expand(*Bsz, N, N).clone()
    bbH = b.unsqueeze(-1) @ ((b.conj() if is_c else b).unsqueeze(-2))
    P_perp = I - bbH

    E = I[..., :, 1:]
    absb2 = (b.conj() * b).real if is_c else (b * b)
    kappa = torch.tensor(kappa, dtype=absb2.dtype, device=device)
    p = 2.0
    wE_real = (1.0 - absb2[..., 1:] + kappa) ** p
    wE = wE_real.to(dtype)
    DE = torch.diag_embed(wE)
    J_E = E @ DE

    if is_c:
        i = torch.arange(N, device=device, dtype=torch.float64).unsqueeze(-1)
        k = torch.arange(1, N, device=device, dtype=torch.float64).unsqueeze(0)
        phase1 = 2.0 * torch.pi * i * k / float(N)
        phase2 = 2.0 * torch.pi * i * (k + 0.5) / float(N)
        J_F1 = torch.exp(1j * phase1).to(dtype)
        J_F2 = torch.exp(1j * phase2).to(dtype)
    else:
        i = torch.arange(N, device=device, dtype=torch.float64).unsqueeze(-1)
        k = torch.arange(1, N, device=device, dtype=torch.float64).unsqueeze(0)
        phase1 = 2.0 * torch.pi * i * k / float(N)
        phase2 = 2.0 * torch.pi * i * (k + 0.5) / float(N)
        J_F1 = torch.sin(phase1).to(dtype)
        J_F2 = torch.sin(phase2).to(dtype)

    if Bsz:
        J_E = J_E.expand(*Bsz, N, N - 1).clone()
        J_F1 = J_F1.expand(*Bsz, N, N - 1).clone()
        J_F2 = J_F2.expand(*Bsz, N, N - 1).clone()

    Z1 = P_perp @ J_E
    Z2 = P_perp @ J_F1
    Z3 = P_perp @ J_F2

    def frob2(Z):
        return (Z.conj() * Z).real.sum(dim=(-2, -1), keepdim=True)

    s1 = frob2(Z1)
    s2 = frob2(Z2)
    s3 = frob2(Z3)
    sigma = torch.tensor(sigma, dtype=s1.dtype, device=device)
    qmix = 2.0
    a1 = (s1 + sigma) ** qmix
    a2 = (s2 + sigma) ** qmix
    a3 = (s3 + sigma) ** qmix
    asum = a1 + a2 + a3
    a1 = (a1 / asum).to(dtype)
    a2 = (a2 / asum).to(dtype)
    a3 = (a3 / asum).to(dtype)

    Z = a1 * Z1 + a2 * Z2 + a3 * Z3

    H = (Z.mH @ Z) if is_c else (Z.transpose(-2, -1) @ Z)
    I_nm1 = torch.eye(N - 1, dtype=H.dtype, device=device)
    tr = H.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1, keepdim=True).unsqueeze(-1)
    scale = (tr / float(N - 1)) + 0.0
    jitter = ((jitter + float(tiny)) * (scale + 1.0)).to(H.dtype)
    H_spd = H + jitter * I_nm1

    evals, U = torch.linalg.eigh(H_spd)
    inv_sqrt = (evals.clamp_min(float(tiny)) ** -0.5).diag_embed().to(H_spd.dtype)
    UH = U.mH if is_c else U.transpose(-2, -1)
    H_inv_sqrt = U @ inv_sqrt @ UH
    Q_perp = Z @ H_inv_sqrt

    Q_perp = P_perp @ Q_perp
    S = (Q_perp.mH @ Q_perp) if is_c else (Q_perp.transpose(-2, -1) @ Q_perp)
    evals2, U2 = torch.linalg.eigh(S)
    inv_sqrt2 = (evals2.clamp_min(float(tiny)) ** -0.5).diag_embed().to(S.dtype)
    U2H = U2.mH if is_c else U2.transpose(-2, -1)
    S_inv_sqrt = U2 @ inv_sqrt2 @ U2H
    Q_perp = Q_perp @ S_inv_sqrt

    R = torch.cat([b.unsqueeze(-1), Q_perp], dim=-1)

    return R
