import torch


def R(
    vec_d: torch.Tensor,
    # smoothing & stability
    kappa: float = 1.0e-3,  # soft de-clumping of J_E weights
    sigma: float = 1.0e-3,  # smoothing for anchor mixing
    jitter: float = 1.0e-6,  # SPD jitter in the polar normalization
    # mixing profile
    p: float = 2.0,  # exponent for J_E weights
    qmix: float = 2.0,  # "temperature" of energy-based anchor mixing
) -> torch.Tensor:
    """
    Construct a CPSF-orthonormal frame R(d) in U(N) over C^N.

    Given a complex, nonzero vector d in C^{..., N}, this routine returns a unitary
    frame
        R(d) = [ b(d) | Q_perp(d) ]  in C^{..., N, N},
    where
        b(d) = d / ||d||                  (first column; exact alignment, CPSF-R3)
        Q_perp(d) in C^{..., N, N-1}      (orthonormal basis of the orthogonal complement of b(d), CPSF-R4)
    and the whole matrix is unitary (CPSF-R1+R2). The construction is smooth in d,
    right-U(N-1) equivariant on the complement (CPSF-R6), and designed to satisfy
    the CPSF regularity criteria (local trivialization / continuity CPSF-R8 and
    bounded-derivative proxy CPSF-R9) within machine precision.

    Complex-only:
    This implementation expects d to have a complex dtype (torch.complex64 or torch.complex128).
    Real dtypes are not part of the CPSF canon and are not supported here.

    Algorithm (complex-only)
    ------------------------
    1) Normalize b = d / max(||d||, sqrt(eps)) and form the projector
    P_perp = I - b b^H.
    2) Build three N x (N-1) anchors in C^N:
    - J_E  : coordinate anchor (standard basis without e1) with smooth weights,
    - J_F1 : complex Fourier-like anchor, phases 2*pi*i*k/N,
    - J_F2 : complex Fourier-like anchor, phases 2*pi*i*(k+0.5)/N.
    3) Project anchors to b_perp: Z_i = P_perp J_i, and mix them with smooth,
    energy-based weights a_i proportional to (||Z_i||_F^2 + sigma)^{qmix}
    (sigma and qmix control switching smoothness).
    4) Polar orthonormalization in the (N-1)-subspace:
    Q_tilde = Z (Z^H Z)^(-1/2) with a scale-aware SPD jitter (jitter) to keep Z^H Z
    well-conditioned.
    5) Re-project and re-polarize once more:
    Q_perp = proj_{b_perp}(Q_tilde) * (Q_tilde^H Q_tilde)^(-1/2).
    6) Concatenate R = [ b | Q_perp ].

    Parameters
    ----------
    d : torch.Tensor, complex dtype (torch.complex64 or torch.complex128)
        Input vectors of shape [..., N].
    kappa : float, default 1e-3
        Soft de-clumping for the J_E weights to avoid dominance by large |b_j|.
    sigma : float, default 1e-3
        Smoothing for anchor mixing (prevents hard switches between anchors).
    jitter : float, default 1e-6
        Scale-aware SPD jitter added to Z^H Z (and to the second polar step) for numerical stability.
    p : float, default 2.0
        Exponent in the J_E weighting (controls emphasis of less-aligned coordinates).
    qmix : float, default 2.0
        "Temperature" of the energy-based anchor mixing; lower is sharper, higher is smoother.

    Returns
    -------
    torch.Tensor
        A unitary matrix R(d) of shape [..., N, N] with:
        - first column exactly b(d) (R3),
        - columns 2..N orthonormal and orthogonal to b(d) (R4),
        - R^H R = R R^H = I (R1-R2).
        For N == 1, returns [..., 1, 1] with the single column b(d).

    CPSF Requirements (R1-R9)
    -------------------------
    R1 (Left unitarity):    R(d)^H R(d) = I_N.
    R2 (Right unitarity):   R(d) R(d)^H = I_N.
    R3 (Alignment):         The first column equals b(d) = d / ||d|| exactly.
    R4 (Complement):        Columns 2..N form an orthonormal basis of b(d)-perp:
                            b(d)^H Q_perp(d) = 0 and Q_perp(d)^H Q_perp(d) = I_{N-1}.
    R5 (Local smoothness):  R(d + delta) depends smoothly on d; for small tangential
                            perturbations delta, ||R(d + delta) - R(d)|| = O(||delta||)
                            and the columns vary continuously without sudden flips.
    R6 (Right U(N-1) equivariance): For any U in U(N-1),
                            R(d) * diag(1, U) is a valid frame with the same first column;
                            the projector Q_perp Q_perp^H is invariant under this right action.
    R7 (Extended-frame unitarity): Any frame obtained by the right block-diagonal action
                            diag(1, U), U in U(N-1), remains unitary in chained compositions
                            used by CPSF (unitarity is preserved under the extension).
    R8 (Local trivialization along paths): Along a geodesic path between d0 and d1 in C^N,
                            there exists a continuous right alignment in U(N-1) such that
                            successive frames remain close (no discontinuous jumps).
    R9 (Bounded derivative proxy): The finite-difference gradient of R with respect to
                            tangential perturbations remains bounded as the step size goes
                            to zero (no blow-up of ||R(d + h*xi) - R(d)|| / h as h -> 0).

    Notes
    -----
    - Fully batched and differentiable (uses Hermitian eigendecompositions on (N-1) x (N-1) SPD
    matrices). Computational cost is O((N-1)^3) per batch element.
    - Small constants (sqrt(eps) from real(d.dtype), sigma, jitter) are chosen to ensure smoothness and
    numerical robustness consistent with CPSF R5/R8/R9 in complex64/complex128 precision.
    """

    if vec_d.dim() < 1:
        raise ValueError(f"R(d): expected [..., N], got {tuple(vec_d.shape)}")
    if not torch.is_complex(vec_d):
        raise TypeError(
            "R(d): CPSF canon expects a complex input (torch.complex64/torch.complex128)."
        )

    *B, N = vec_d.shape
    dtype, device = vec_d.dtype, vec_d.device

    finfo = torch.finfo(vec_d.real.dtype)
    tiny = torch.sqrt(torch.tensor(finfo.eps, dtype=vec_d.real.dtype, device=device))

    dn = torch.linalg.vector_norm(vec_d, dim=-1, keepdim=True)
    b = vec_d / torch.clamp(dn, min=tiny)

    if N == 1:
        return b.unsqueeze(-1)

    I = torch.eye(N, dtype=dtype, device=device)
    if B:
        I = I.expand(*B, N, N).clone()
    P_perp = I - b.unsqueeze(-1) @ b.conj().unsqueeze(-2)

    E = I[..., :, 1:]
    absb2 = (b.conj() * b).real
    wE_real = (
        1.0 - absb2[..., 1:] + torch.tensor(kappa, dtype=vec_d.real.dtype, device=device)
    ) ** p
    DE = torch.diag_embed(wE_real.to(dtype))
    J_E = E @ DE

    reals = vec_d.real.dtype
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


def R_ext(
    R: torch.Tensor,
) -> torch.Tensor:
    if R.dim() < 2 or R.shape[-1] != R.shape[-2]:
        raise ValueError(f"R_ext(R): expected [..., N, N], got {tuple(R.shape)}")

    *B, N, _ = R.shape
    Z = torch.zeros(*B, N, N, dtype=R.dtype, device=R.device)

    top = torch.cat([R, Z], dim=-1)
    bottom = torch.cat([Z, R], dim=-1)
    R_ext = torch.cat([top, bottom], dim=-2)

    return R_ext
