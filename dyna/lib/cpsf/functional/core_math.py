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
    frame:

        R(d) = [ b(d) | Q_perp(d) ]  in C^{..., N, N},
    where:

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
    R1 (Left unitarity) :   R(d)^H R(d) = I_N.
    R2 (Right unitarity) :  R(d) R(d)^H = I_N.
    R3 (Alignment) :        The first column equals b(d) = d / ||d|| exactly.
    R4 (Complement) :       Columns 2..N form an orthonormal basis of b(d)-perp:
                            b(d)^H Q_perp(d) = 0 and Q_perp(d)^H Q_perp(d) = I_{N-1}.
    R5 (Local smoothness) : R(d + delta) depends smoothly on d; for small tangential
                            perturbations delta, ||R(d + delta) - R(d)|| = O(||delta||)
                            and the columns vary continuously without sudden flips.
    R6 (Right U(N-1) equivariance) : For any U in U(N-1),
                            R(d) * diag(1, U) is a valid frame with the same first column;
                            the projector Q_perp Q_perp^H is invariant under this right action.
    R7 (Extended-frame unitarity) : Any frame obtained by the right block-diagonal action
                            diag(1, U), U in U(N-1), remains unitary in chained compositions
                            used by CPSF (unitarity is preserved under the extension).
    R8 (Local trivialization along paths) : Along a geodesic path between d0 and d1 in C^N,
                            there exists a continuous right alignment in U(N-1) such that
                            successive frames remain close (no discontinuous jumps).
    R9 (Bounded derivative proxy) : The finite-difference gradient of R with respect to
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
        1.0
        - absb2[..., 1:]
        + torch.tensor(kappa, dtype=vec_d.real.dtype, device=device)
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
    """
    Construct the extended CPSF frame R_ext(R) in U(2N) over C^{2N}.

    Given a unitary CPSF frame R in C^{..., N, N} (typically produced by R(d)),
    this routine returns the canonical block-diagonal extension:

        R_ext(R) = block_diag(R, R)  in C^{..., 2N, 2N}.

    It applies the same frame R to both subspaces (position and direction) with no cross-coupling.

    This implementation is allocation-friendly: it preallocates the output once and
    writes the two diagonal blocks via copy_(), avoiding temporary tensors and cat() chains.

    Properties
    ----------
    - Unitarity:           R_ext^H R_ext = R_ext R_ext^H = I_{2N}  (inherited from R).
    - Block structure:     Off-diagonal blocks are exactly zero; the two diagonal blocks are identical.
    - First-column align:  If R = R(d) with first column b(d) = d / ||d||, then the first column
                        in each diagonal block equals b(d).
    - Right U(N-1) action: For any Q in U(N-1), replacing R by R * diag(1, Q) leaves downstream
                        Sigma = R_ext D R_ext^H unchanged when D = diag(s_par, s_perp, ..., s_perp)
                        within each N-block.
    - Batched:             Leading batch dimensions are preserved.
    - Deterministic:       Pure tensor ops; no randomness. Fully differentiable w.r.t. R.

    Algorithm
    ---------
    1) Validate that the input has shape [..., N, N] with square last dimensions.
    2) Allocate the output: out = R.new_zeros(..., 2N, 2N).
    3) Write diagonal blocks:
        out[..., :N, :N].copy_(R)
        out[..., N:, N:].copy_(R)
    4) Return out.

    Parameters
    ----------
    R : torch.Tensor, complex dtype (torch.complex64 or torch.complex128)
        Input unitary frame(s) of shape [..., N, N]. Typically obtained from R(d).

    Returns
    -------
    torch.Tensor
        The extended unitary frame of shape [..., 2N, 2N], equal to block_diag(R, R).

    CPSF Notes
    ----------
    - This extension is used to build Sigma via
        Sigma(R_ext, s_par, s_perp) = R_ext D R_ext^H,
    where D selects the "parallel" index (0 and N in the 2N diagonal) and fills the others
    with the "perp" value in each block.

    Edge cases
    ----------
    - N == 1: returns a 2x2 diagonal matrix with the single 1x1 block repeated.

    Complex-only
    ------------
    R is expected to have a complex dtype. Real dtypes are not part of the CPSF canon here.
    """

    if R.dim() < 2 or R.shape[-1] != R.shape[-2]:
        raise ValueError(f"R_ext(R): expected [..., N, N], got {tuple(R.shape)}")

    *B, N, _ = R.shape
    out = R.new_zeros(*B, 2 * N, 2 * N)
    out[..., :N, :N].copy_(R)
    out[..., N:, N:].copy_(R)

    return out


def Sigma(
    R_ext: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
) -> torch.Tensor:
    """
    Construct the CPSF covariance Sigma from an extended frame R_ext and (sigma_par, sigma_perp).

    Canonical definition
    --------------------
    Given an extended CPSF frame
        R_ext = block_diag(R, R) in C^{..., 2N, 2N}
    and the diagonal weight operator
        D = diag(sigma_par, sigma_perp, ..., sigma_perp,  sigma_par, sigma_perp, ..., sigma_perp),
    the covariance is
        Sigma = R_ext * D * R_ext^H,
    where ^H denotes the conjugate transpose.

    Implementation (allocation-friendly)
    ------------------------------------
    This implementation avoids materializing D and 2Nx2N GEMMs. It extracts R = R_ext[..., :N, :N],
    lets b = R[:, 0] (the first column), forms:

        S0 = sigma_perp * I_N + (sigma_par - sigma_perp) * (b b^H),
    and assembles:

        Sigma = diag(S0, S0)
    by preallocating the 2Nx2N output and writing the two diagonal blocks via copy_().

    Properties
    ----------
    - Hermitian SPD:      Sigma == Sigma^H and Sigma is positive definite for sigma_par > 0 and sigma_perp > 0.
    - Block structure:    Sigma = diag(S0, S0) with S0 = R * diag(sigma_par, sigma_perp, ..., sigma_perp) * R^H.
    - Spectrum:           eig(Sigma) = {sigma_par (mult 2), sigma_perp (mult 2*(N-1))}.
    - Isotropy:           if sigma_par == sigma_perp == s, then Sigma = s * I(2N).
    - Inverse (closed form):
                        Sigma^{-1} = R_ext * diag(1/sigma_par, 1/sigma_perp, ..., 1/sigma_perp,
                                                1/sigma_par, 1/sigma_perp, ..., 1/sigma_perp) * R_ext^H.
    - Invariances:        right action R -> R * diag(1, Q), Q in U(N-1), and the first-column phase
                        R -> R * diag(exp(i*phi), I) leave Sigma unchanged.
    - Linearity in D:     Sigma is linear w.r.t. (sigma_par, sigma_perp) and scales as
                        Sigma(alpha*sigma_par, alpha*sigma_perp) = alpha * Sigma(sigma_par, sigma_perp).
    - Batched:            leading batch dimensions are preserved; sigma_par/sigma_perp broadcast across them.
    - Differentiable:     pure tensor ops; gradients flow through R_ext and sigma parameters.

    Algorithm
    ---------
    1) Validate input shape [..., 2N, 2N] and positivity of sigma_par, sigma_perp.
    2) Extract R = R_ext[..., :N, :N] and its first column b.
    3) Build S0 = sigma_perp*I_N + (sigma_par - sigma_perp)*(b b^H).
    4) Preallocate out[..., 2N, 2N]; write S0 into TL and BR blocks; return out.

    Parameters
    ----------
    R_ext : torch.Tensor (complex64 or complex128), shape [..., 2N, 2N]
        Extended CPSF frame, typically obtained as block_diag(R, R) with R = R(d).
    sigma_par : torch.Tensor (real), broadcastable to leading batch dims
        Positive scalar/tensor for the "parallel" direction (index 0 in each N-block).
    sigma_perp : torch.Tensor (real), broadcastable to leading batch dims
        Positive scalar/tensor for the orthogonal complement (indices 1..N-1 in each N-block).

    Returns
    -------
    torch.Tensor (complex, same dtype/device as R_ext), shape [..., 2N, 2N]
        The CPSF covariance matrix Sigma.

    CPSF notes
    ----------
    - Matrix-free application: Sigma * [u; v] = [S0*u; S0*v] with S0 as above, which can be computed
    without materializing Sigma.
    - logdet(Sigma) = 2*log(sigma_par) + 2*(N-1)*log(sigma_perp). Sigma^{+/-1/2} follow by replacing
    sigma with sigma^{+/-1/2}.

    Edge cases
    ----------
    - CPSF typically assumes N >= 2. For completeness, at N == 1 the formula degenerates to
    Sigma = diag(sigma_par, sigma_par). No randomness is used.
    """

    if R_ext.dim() < 2 or R_ext.shape[-1] != R_ext.shape[-2]:
        raise ValueError(
            f"Sigma: expected R_ext as [..., 2N, 2N], got {tuple(R_ext.shape)}"
        )

    *B, twoN, _ = R_ext.shape
    if twoN % 2 != 0:
        raise ValueError(f"Sigma: last dims must be even, got {twoN}")
    N = twoN // 2

    device = R_ext.device
    dt_cplx = R_ext.dtype
    dt_real = R_ext.real.dtype

    sp = torch.as_tensor(sigma_par, device=device, dtype=dt_real)
    sq = torch.as_tensor(sigma_perp, device=device, dtype=dt_real)
    if not (torch.all(sp > 0) and torch.all(sq > 0)):
        raise ValueError("Sigma: sigma_par and sigma_perp must be positive")

    R = R_ext[..., :N, :N]
    b = R[..., :, 0]
    delta = (sp - sq).to(dtype=dt_cplx)[..., None, None]
    S0 = (b.unsqueeze(-1) * b.conj().unsqueeze(-2)) * delta
    S0.diagonal(dim1=-2, dim2=-1).add_(sq[..., None])

    out = S0.new_zeros(*B, 2 * N, 2 * N)
    out[..., :N, :N].copy_(S0)
    out[..., N:, N:].copy_(S0)

    return out


def delta_vec_d(
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    Compute the CPSF directional tangent offset delta_vec_d(vec_d, vec_d_j; eps).

    Canonical definition
    --------------------
    Let <u,v> = sum(conj(u_k) * v_k) be the Hermitian inner product. For unit directions
    vec_d, vec_d_j in C^N, define:

        inner = <vec_d_j, vec_d>                  # complex
        c     = |inner|                           # real in [0, 1]
        theta = arccos(c)                         # geodesic angle
        t     = vec_d - inner * vec_d_j           # tangent at vec_d_j (orthogonal to vec_d_j)
    and the smoothed sine:

        N_eps(c) = sqrt( 1 - c^2 + eps * exp( -(1 - c^2)/eps ) ),  eps > 0.

    Then:

        delta_vec_d(vec_d, vec_d_j; eps) = (theta / N_eps(c)) * t.

    Properties
    ----------
    - Tangency:        <vec_d_j, delta_vec_d> == 0 (numerically ~ 0).
    - Phase equiv.:    For any real phi, delta(e^{i*phi}*vec_d, vec_d_j) = e^{i*phi} * delta(vec_d, vec_d_j).
    - Joint phase eq.: For any real psi, delta(e^{i*psi}*vec_d, e^{i*psi}*vec_d_j) = e^{i*psi} * delta(vec_d, vec_d_j).
    - Zero at collin.: If vec_d is collinear with vec_d_j (c = 1), then delta_vec_d == 0.
    - Smoothness:      N_eps prevents division by zero at c -> 1; function is differentiable in all inputs for eps > 0.
    - Norm bound:      ||delta_vec_d|| <= theta; away from the smoothing region (1 - c^2 >> eps), ||delta_vec_d|| ~ theta.
    - Batched:         Leading batch dimensions are preserved; no RNG; deterministic for fixed inputs.

    Algorithm
    ---------
    1) Validate shapes [..., N], N >= 2; same dtype/device; eps > 0.
    2) inner = sum(conj(vec_d_j) * vec_d), c = clamp(|inner|, 0, 1), theta = arccos(c).
    3) t = vec_d - inner * vec_d_j (automatically orthogonal to vec_d_j).
    4) denom = sqrt( (1 - c^2) + eps * exp(-(1 - c^2)/eps) ).
    5) Return delta = (theta / denom) * t.

    Parameters
    ----------
    vec_d : torch.Tensor (complex64 or complex128), shape [..., N]
        Query direction (assumed unit-norm by the caller).
    vec_d_j : torch.Tensor (complex, same dtype/device), shape [..., N]
        Anchor direction (assumed unit-norm by the caller).
    eps : float
        Positive smoothing parameter; must be > 0.

    Returns
    -------
    torch.Tensor (complex, same dtype/device), shape [..., N]
        Tangent displacement at vec_d_j toward vec_d.

    Notes
    -----
    - CPSF assumes N >= 2 and unit-norm inputs; this function does not renormalize them.
    - All operations are elementwise or reductions; fully batched; gradient-safe.

    Edge cases
    ----------
    - If vec_d == vec_d_j (or vec_d = e^{i*phi} * vec_d_j), then t == 0 and delta_vec_d == 0.
    - If 1 - c^2 is very small, denom ~ sqrt(eps), ensuring finite and smooth output.
    """

    if vec_d.shape != vec_d_j.shape:
        raise ValueError(
            f"delta_vec_d: expected matching shapes [..., N], got {tuple(vec_d.shape)} vs {tuple(vec_d_j.shape)}"
        )
    if vec_d.dtype != vec_d_j.dtype:
        raise ValueError("delta_vec_d: dtype mismatch between vec_d and vec_d_j")
    if vec_d.device != vec_d_j.device:
        raise ValueError("delta_vec_d: device mismatch between vec_d and vec_d_j")
    if vec_d.dim() < 1 or vec_d.shape[-1] < 2:
        raise ValueError(
            f"delta_vec_d: expected last dim N>=2, got N={vec_d.shape[-1]}"
        )
    if eps <= 0:
        raise ValueError("delta_vec_d: eps must be positive")

    inner = torch.sum(vec_d_j.conj() * vec_d, dim=-1)
    tangent = vec_d - inner.unsqueeze(-1) * vec_d_j
    real_dtype = vec_d.real.dtype
    c = inner.abs().to(real_dtype).clamp(0.0, 1.0)
    theta = torch.acos(c)
    sin2 = (1.0 - c * c).clamp_min(0.0)
    eps_t = torch.as_tensor(eps, dtype=real_dtype, device=vec_d.device)
    denom = torch.sqrt(sin2 + eps_t * torch.exp(-sin2 / eps_t))
    scale = (theta / denom).unsqueeze(-1)
    delta = tangent * scale

    return delta


def iota(
    delta_z: torch.Tensor,
    delta_vec_d: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_complex(delta_z) or not torch.is_complex(delta_vec_d):
        raise ValueError(
            "\n".join(
                [
                    f"iota: expected complex [..., N], got:",
                    f"delta_z: {delta_z.dtype} {tuple(delta_z.shape)}",
                    f"delta_vec_d: {delta_vec_d.dtype} {tuple(delta_vec_d.shape)}",
                ]
            )
        )

    try:
        return torch.cat([delta_z, delta_vec_d], dim=-1)
    except RuntimeError as e:
        raise ValueError(
            f"iota: concat failed for shapes {tuple(delta_z.shape)} and {tuple(delta_vec_d.shape)}"
        ) from e
