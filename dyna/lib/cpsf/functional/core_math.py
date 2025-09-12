import torch
import math

from typing import Optional, Union

from dyna.lib.cpsf.structures import CPSFPsiOffsetsIterator


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
    - J_F1 : complex Fourier-like anchor, phases 2 * pi * i * k/N,
    - J_F2 : complex Fourier-like anchor, phases 2 * pi * i * (k+0.5)/N.
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
    R1 (Left unitarity)             : R(d)^H R(d) = I_N.
    R2 (Right unitarity)            : R(d) R(d)^H = I_N.
    R3 (Alignment)                  : The first column equals b(d) = d / ||d|| exactly.
    R4 (Complement)                 : Columns 2..N form an orthonormal basis of b(d)-perp:
                                        b(d)^H Q_perp(d) = 0 and Q_perp(d)^H Q_perp(d) = I_{N-1}.
    R5 (Local smoothness)           : R(d + delta) depends smoothly on d; for small tangential
                                        perturbations delta, ||R(d + delta) - R(d)|| = O(||delta||)
                                        and the columns vary continuously without sudden flips.
    R6 (Right U(N-1) equivariance)  : For any U in U(N-1),
                                        R(d) * diag(1, U) is a valid frame with the same first column;
                                        the projector Q_perp Q_perp^H is invariant under this right action.
    R7 (Extended-frame unitarity)   : Any frame obtained by the right block-diagonal action
                                        diag(1, U), U in U(N-1), remains unitary in chained compositions
                                        used by CPSF (unitarity is preserved under the extension).
    R8 (Local trivialization)       : Along a geodesic path between d0 and d1 in C^N,
                                        there exists a continuous right alignment in U(N-1) such that
                                        successive frames remain close (no discontinuous jumps).
    R9 (Bounded derivative proxy)   : The finite-difference gradient of R with respect to
                                        tangential perturbations remains bounded as the step size goes
                                        to zero (no blow-up of ||R(d + h * xi) - R(d)|| / h as h -> 0).

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
    - Unitarity             : R_ext^H R_ext = R_ext R_ext^H = I_{2N}  (inherited from R).
    - Block structure       : Off-diagonal blocks are exactly zero; the two diagonal blocks are identical.
    - First-column align    : If R = R(d) with first column b(d) = d / ||d||, then the first column
                                in each diagonal block equals b(d).
    - Right U(N-1) action   : For any Q in U(N-1), replacing R by R * diag(1, Q) leaves downstream
                                Sigma = R_ext D R_ext^H unchanged when D = diag(s_par, s_perp, ..., s_perp)
                                within each N-block.
    - Batched               : Leading batch dimensions are preserved.
    - Deterministic         : Pure tensor ops; no randomness. Fully differentiable w.r.t. R.

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
    - Hermitian SPD         : Sigma == Sigma^H and Sigma is positive definite for sigma_par > 0 and sigma_perp > 0.
    - Block structure       : Sigma = diag(S0, S0) with S0 = R * diag(sigma_par, sigma_perp, ..., sigma_perp) * R^H.
    - Spectrum              : eig(Sigma) = {sigma_par (mult 2), sigma_perp (mult 2*(N-1))}.
    - Isotropy              : if sigma_par == sigma_perp == s, then Sigma = s * I(2N).
    - Inverse (closed form) : Sigma^{-1} = R_ext * diag(1/sigma_par, 1/sigma_perp, ..., 1/sigma_perp,
                                1/sigma_par, 1/sigma_perp, ..., 1/sigma_perp) * R_ext^H.
    - Invariances           : right action R -> R * diag(1, Q), Q in U(N-1), and the first-column phase
                                R -> R * diag(exp(i*phi), I) leave Sigma unchanged.
    - Linearity in D        : Sigma is linear w.r.t. (sigma_par, sigma_perp) and scales as
                                Sigma(alpha * sigma_par, alpha * sigma_perp) = alpha * Sigma(sigma_par, sigma_perp).
    - Batched               : leading batch dimensions are preserved; sigma_par/sigma_perp broadcast across them.
    - Differentiable        : pure tensor ops; gradients flow through R_ext and sigma parameters.

    Algorithm
    ---------
    1) Validate input shape [..., 2N, 2N] and positivity of sigma_par, sigma_perp.
    2) Extract R = R_ext[..., :N, :N] and its first column b.
    3) Build S0 = sigma_perp * I_N + (sigma_par - sigma_perp) * (b b^H).
    4) Preallocate out[..., 2N, 2N]; write S0 into TL and BR blocks; return out.

    Parameters
    ----------
    R_ext       : torch.Tensor (complex64 or complex128), shape [..., 2N, 2N]
        Extended CPSF frame, typically obtained as block_diag(R, R) with R = R(d).
    sigma_par   : torch.Tensor (real), broadcastable to leading batch dims
        Positive scalar/tensor for the "parallel" direction (index 0 in each N-block).
    sigma_perp  : torch.Tensor (real), broadcastable to leading batch dims
        Positive scalar/tensor for the orthogonal complement (indices 1..N-1 in each N-block).

    Returns
    -------
    torch.Tensor (complex, same dtype/device as R_ext), shape [..., 2N, 2N]
        The CPSF covariance matrix Sigma.

    CPSF notes
    ----------
    - Matrix-free application: Sigma * [u; v] = [S0 * u; S0 * v] with S0 as above, which can be computed
    without materializing Sigma.
    - logdet(Sigma) = 2 * log(sigma_par) + 2 * (N-1) * log(sigma_perp). Sigma^{+/-1/2} follow by replacing
    sigma with sigma^{+/-1/2}.

    Edge cases
    ----------
    - CPSF typically assumes N >= 2. For completeness, at N == 1 the formula degenerates to
    Sigma = diag(sigma_par, sigma_par). No randomness is used.
    """

    if torch.is_complex(sigma_par) or torch.is_complex(sigma_perp):
        raise ValueError("Sigma: sigma_par and sigma_perp must be real-valued")
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


def q(
    w: torch.Tensor,
    R_ext: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the CPSF quadratic form q = <Sigma^{-1} w, w>.

    Canon
    -----
    Sigma^{-1} = R_ext * D^{-1} * R_ext^H, with
        R_ext = block_diag(R, R), R ∈ U(N),
        D^{-1} = diag(1/sp, 1/sq, ..., 1/sq, 1/sp, 1/sq, ..., 1/sq).

    Let w = [u; v] ∈ C^{2N}.

    The implementation evaluates q without forming Sigma^{-1} explicitly via y = R^H u, z = R^H v.

    Shapes
    ------
    w           : [..., 2N] complex
    R_ext       : [..., 2N, 2N] complex
    sigma_par   : real, broadcastable to leading batch dims
    sigma_perp  : real, broadcastable to leading batch dims
    Returns     : [...,] real (non-negative)

    Args
    ----
    w               : Complex 2N-vector (concatenation of u and v).
    R_ext           : Block-diagonal extension block_diag(R, R), consistent with w.
    sigma_par (sp)  : Parallel variance (>0).
    sigma_perp (sq) : Perpendicular variance (>0).

    Raises
    ------
    ValueError if:
    - inputs are not complex or have mismatched dtype/device;
    - R_ext is not square [..., 2N, 2N] or trailing dims mismatch;
    - last dim of w is not even (2N) or N < 2;
    - sigma_par / sigma_perp are non-real or non-positive.

    Notes
    -----
    On CUDA, positivity of sigmas is enforced without host synchronization
    (via torch._assert_async when available).
    """

    if torch.is_complex(sigma_par) or torch.is_complex(sigma_perp):
        raise ValueError("q: sigma_par and sigma_perp must be real-valued")
    if w.dtype != R_ext.dtype:
        raise ValueError(
            f"q: dtype mismatch: w.dtype={w.dtype}, R_ext.dtype={R_ext.dtype}"
        )
    if w.device != R_ext.device:
        raise ValueError(
            f"q: device mismatch: w.device={w.device}, R_ext.device={R_ext.device}"
        )
    if not torch.is_complex(w) or not torch.is_complex(R_ext):
        raise ValueError(
            f"q: expected complex inputs, got w:{w.dtype}, R_ext:{R_ext.dtype}"
        )
    if R_ext.dim() < 2 or R_ext.shape[-1] != R_ext.shape[-2]:
        raise ValueError(f"q: R_ext must be [..., 2N, 2N], got {tuple(R_ext.shape)}")
    if w.shape[-1] != R_ext.shape[-1]:
        raise ValueError(
            f"q: trailing dim mismatch, w:[..., {w.shape[-1]}] vs R_ext:[..., {R_ext.shape[-1]}, {R_ext.shape[-1]}]"
        )

    twoN = w.shape[-1]
    N = twoN // 2

    if twoN % 2 != 0:
        raise ValueError("q: expected even last dim 2N")
    if N < 2:
        raise ValueError(f"q: N must be >= 2 per CPSF (got N={N})")

    WV = w.reshape(*w.shape[:-1], 2, N).transpose(-2, -1)
    R = R_ext[..., :N, :N]
    YZ = R.mH @ WV

    dt_real = w.real.dtype
    device = w.device
    sp = torch.as_tensor(sigma_par, dtype=dt_real, device=device)
    sq = torch.as_tensor(sigma_perp, dtype=dt_real, device=device)

    # --- Positivity guard for sigmas (avoid CUDA sync in the hot path) ---
    if sp.is_cuda or sq.is_cuda:
        if hasattr(torch, "_assert_async"):
            cond = (sp > 0).all() & (sq > 0).all()
            torch._assert_async(
                cond,
                "q: sigma_par and sigma_perp must be positive",
            )
        else:
            if not isinstance(
                sigma_par,
                torch.Tensor,
            ) and not isinstance(
                sigma_perp,
                torch.Tensor,
            ):
                if not (sigma_par > 0 and sigma_perp > 0):
                    raise ValueError("q: sigma_par and sigma_perp must be positive")
    else:
        if not ((sp > 0).all().item() and (sq > 0).all().item()):
            raise ValueError("q: sigma_par and sigma_perp must be positive")

    inv_par = torch.reciprocal(sp)
    inv_perp = torch.reciprocal(sq)
    sq_mag = (YZ.conj() * YZ).real.sum(dim=-1)
    q0 = sq_mag[..., 0] * inv_par
    qperp = sq_mag[..., 1:].sum(dim=-1) * inv_perp
    out = (q0 + qperp).to(dtype=dt_real)

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
    - Tangency          : <vec_d_j, delta_vec_d> == 0 (numerically ~ 0).
    - Phase equiv.      : For any real phi, delta(e^{i * phi} * vec_d, vec_d_j) = e^{i * phi} * delta(vec_d, vec_d_j).
    - Joint phase eq.   : For any real psi, delta(e^{i * psi} * vec_d, e^{i * psi} * vec_d_j) = e^{i * psi} * delta(vec_d, vec_d_j).
    - Zero at collin.   : If vec_d is collinear with vec_d_j (c = 1), then delta_vec_d == 0.
    - Smoothness        : N_eps prevents division by zero at c -> 1; function is differentiable in all inputs for eps > 0.
    - Norm bound        : ||delta_vec_d|| <= theta; away from the smoothing region (1 - c^2 >> eps), ||delta_vec_d|| ~ theta.
    - Batched           : Leading batch dimensions are preserved; no RNG; deterministic for fixed inputs.

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
    - If vec_d == vec_d_j (or vec_d = e^{i * phi} * vec_d_j), then t == 0 and delta_vec_d == 0.
    - If 1 - c^2 is very small, denom ~ sqrt(eps), ensuring finite and smooth output.
    """

    if vec_d.shape != vec_d_j.shape:
        raise ValueError(
            f"delta_vec_d: expected matching shapes [..., N], got {tuple(vec_d.shape)} vs {tuple(vec_d_j.shape)}"
        )
    if vec_d.dtype != vec_d_j.dtype:
        raise ValueError(
            f"delta_vec_d: dtype mismatch between vec_d and vec_d_j, got {vec_d.dtype} vs {vec_d_j.dtype}"
        )
    if vec_d.device != vec_d_j.device:
        raise ValueError(
            f"delta_vec_d: device mismatch between vec_d and vec_d_j, got {vec_d.device} vs {vec_d_j.device}"
        )
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
    """
    Embed a pair of complex N-vectors as a single 2N-vector via concatenation.

    Canonical map: iota(u, v) = [u; v], used by CPSF to couple spatial and directional
    components.

    Shapes
    ------
    delta_z     : [..., N] complex
    delta_vec_d : [..., N] complex
    Returns     : [..., 2N] complex

    Args
    ----
    delta_z     : Complex displacement (e.g., z - z_j + n) in C^N.
    delta_vec_d : Complex directional offset delta_vec_d(d, d_j) in C^N.

    Returns
    -------
    Concatenated tensor with the last dimension formed as [delta_z, delta_vec_d].

    Raises
    ------
    ValueError if inputs are not complex or if shapes/dtypes/devices mismatch,
    or if concatenation fails.
    """

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
    if delta_z.shape != delta_vec_d.shape:
        raise ValueError(
            f"iota: expected matching shapes [..., N], got {tuple(delta_z.shape)} vs {tuple(delta_vec_d.shape)}"
        )
    if delta_z.dtype != delta_vec_d.dtype:
        raise ValueError(
            f"iota: dtype mismatch between delta_z and delta_vec_d, got {delta_z.dtype} vs {delta_vec_d.dtype}"
        )
    if delta_z.device != delta_vec_d.device:
        raise ValueError(
            f"iota: device mismatch between delta_z and delta_vec_d, got {delta_z.device} vs {delta_vec_d.device}"
        )

    try:
        return torch.cat([delta_z, delta_vec_d], dim=-1)
    except RuntimeError as e:
        raise ValueError(
            f"iota: concat failed for shapes {tuple(delta_z.shape)} and {tuple(delta_vec_d.shape)}"
        ) from e


def rho(
    q: torch.Tensor,
    q_max: Optional[Union[int, float, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Gaussian envelope rho(q) = exp(-pi * q).

    Canon
    -----
    Used inside psi^T_j(z, d) via rho(q(w)), where q(w) = <Sigma^{-1} w, w>.
    Optional clamping of q by q_max is applied before the exponent to
    improve numerical stability for large q.

    Shapes
    ------
    q       : [...,] real
    q_max   : None or broadcastable real scalar/tensor
    returns : [...,] real (in (0, 1] when q >= 0)

    Args
    ----
    q       : Real-valued quadratic form (e.g., from q(w)).
    q_max   : Optional upper bound applied elementwise to q before exp.

    Returns
    -------
    Tensor of the same leading shape as q with values exp(-pi * clamp(q, max=q_max)).

    Raises
    ------
    ValueError if q is complex or q_max is complex.

    Notes
    -----
    - dtype/device follow q (and q_max is cast to q.dtype/q.device if provided).
    - The function does not enforce q >= 0; if q < 0, the output may exceed 1.
    - Typical integration in CPSF: q_max is provided from context (e.g., ctx.exp_clip_q_max).
    """

    if torch.is_complex(q):
        raise ValueError(f"rho: expected real q, got dtype={q.dtype}")

    if q_max is not None:
        if isinstance(q_max, torch.Tensor):
            if torch.is_complex(q_max):
                raise ValueError("rho: q_max must be real-valued")
            q_cap = q_max.to(dtype=q.dtype, device=q.device)
        else:
            q_cap = torch.as_tensor(q_max, dtype=q.dtype, device=q.device)
        q_ = torch.clamp(q, max=q_cap)
    else:
        q_ = q

    return torch.exp(-torch.pi * q_)


def lift(
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Lift map: return lifted coordinates of z.

    Canon
    -----
    CPSF uses a "lifted" spatial coordinate tilde{z} when forming iota(...).
    In the canonical parameterization currently adopted, the lift is identity:
    tilde{z} == z. This function exists to keep the pipeline explicit and
    centralized, so all callers uniformly apply lift(z) even when it is a no-op.

    Behavior
    --------
    - Validates that z is complex and shaped as [..., N].
    - Returns z unchanged (no reparameterization, no scaling).
    - Preserves shape, dtype, and device; does not allocate a new tensor.

    Shapes
    ------
    z       : [..., N] complex
    returns : [..., N] complex (same object)

    Args
    ----
    z: Complex spatial coordinate.

    Returns
    -------
    The same tensor z (identity lift).

    Raises
    ------
    ValueError if z is not complex, if it has no trailing dimension, or if N < 2.

    Notes
    -----
    Keeping lift as a dedicated function improves readability (matching the CPSF
    notation) and future-proofs the code should a non-trivial lifting be introduced
    (e.g., alternative coordinate embeddings or normalizations).
    """

    if not torch.is_complex(z):
        raise ValueError(
            f"lift: expected complex [..., N], got {z.dtype} {tuple(z.shape)}"
        )
    if z.dim() == 0:
        raise ValueError("lift: expected [..., N], got a 0-dim scalar tensor")
    if z.shape[-1] < 2:
        raise ValueError(f"lift: N must be >= 2 per CPSF (got N={z.shape[-1]})")

    return z


def hermitianize(
    A: torch.Tensor,
) -> torch.Tensor:
    """
    Hermitian (symmetric) part: (A + A^H) / 2.
    """

    if torch.is_complex(A):
        return 0.5 * (A + A.mH)
    else:
        return 0.5 * (A + A.transpose(-2, -1))


def cholesky_spd(
    A: torch.Tensor,
    eps: Optional[Union[float, torch.Tensor]] = None,
    use_jitter: bool = False,
) -> torch.Tensor:
    """
    Cholesky factor L for Hermitian/Symmetric positive-definite A.
    Hermitianizes input and, if requested, retries with a small real diagonal jitter.

    A       : [..., n, n] real/complex
    returns : [..., n, n] real/complex (lower-triangular)

    Notes:
    ------
    On CUDA, finiteness and eps positivity are asserted without host sync via
    torch._assert_async when available.
    """

    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"cholesky_spd: expected [..., n, n], got {tuple(A.shape)}")
    if not (torch.is_complex(A) or torch.is_floating_point(A)):
        raise ValueError(
            f"cholesky_spd: expected real/complex floating dtype, got {A.dtype}"
        )

    A_h = hermitianize(A=A)

    # --- Finiteness guard for A_h (avoid CUDA sync in the hot path) ---
    e_str = "cholesky_spd: non-finite entries in input"
    if A_h.is_cuda:
        if hasattr(torch, "_assert_async"):
            torch._assert_async(torch.isfinite(A_h).all(), e_str)
    else:
        if not torch.isfinite(A_h).all().item():
            raise ValueError(e_str)

    try:
        return torch.linalg.cholesky(A_h)
    except RuntimeError as e:
        if not use_jitter:
            raise e

        dt_real = A_h.real.dtype if torch.is_complex(A_h) else A_h.dtype
        device = A_h.device

        # --- Positivity guard for eps (avoid CUDA sync) ---
        e_str = "cholesky_spd: eps must be positive when provided"
        if eps is not None:
            if isinstance(eps, torch.Tensor):
                if eps.is_cuda:
                    if hasattr(torch, "_assert_async"):
                        torch._assert_async((eps > 0).all(), e_str)
                    # else: skip strict CUDA check to avoid sync; rely on jitter path to fail if wrong
                else:
                    if not (eps > 0).all().item():
                        raise ValueError(e_str)
            else:
                if float(eps) <= 0.0:
                    raise ValueError(e_str)

        if eps is None:
            eps_t = torch.tensor(
                math.sqrt(torch.finfo(dt_real).eps), dtype=dt_real, device=device
            )
        else:
            eps_t = (
                eps.to(dtype=dt_real, device=device)
                if isinstance(eps, torch.Tensor)
                else torch.tensor(float(eps), dtype=dt_real, device=device)
            )

        diag_abs_mean = (
            A_h.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True).to(dt_real)
        ).clamp(min=1.0)
        n = A_h.shape[-1]
        I = torch.eye(n, device=device, dtype=dt_real)
        jitter = (eps_t * diag_abs_mean)[..., None] * I
        A_h_jittered = A_h + (
            jitter if not torch.is_complex(A_h) else jitter.type_as(A_h)
        )

        try:
            return torch.linalg.cholesky(A_h_jittered)
        except RuntimeError as e2:
            raise ValueError(f"cholesky_spd: failed even with jitter: {e2}") from e2


def Tau_nearest(
    z: torch.Tensor,
    z_j: torch.Tensor,
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    T_hat_j: torch.Tensor,
    alpha_j: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
    R_j: Optional[torch.Tensor] = None,
    q_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Tau backend: nearest-image approximation.

    What it does
    ------------
    - Computes CPSF contribution using only the nearest torus image.
    - Wraps Re(dz) into [-0.5, 0.5) while leaving Im(dz) unchanged.
    - Evaluates eta = rho(q(iota(dz_wrapped, dd))) and returns
    T(z, vec_d) = sum_j (alpha_j * eta_j) * T_hat_j.

    Where it is used
    ----------------
    - Narrow kernels (small sigma_*), far from fundamental-domain boundaries.
    - Large N where enumerating many lattice images is infeasible.
    - Router should select this when predicted tail beyond the nearest image
    is negligible.

    Why no runtime checks here
    --------------------------
    - All validations (shapes, dtypes, parameter sanity) are performed by the router.
    - This function assumes canonical inputs.

    Strengths
    ---------
    - O(M) cost; minimal memory; excellent throughput.
    - Scales well to high N; zero tuning (no window or frequency set).

    Weaknesses
    ----------
    - Approximate: ignores all non-nearest images; near boundaries or with
    fat kernels this can bias the result.
    - No intrinsic tail control (decision belongs to the router).
    """

    r_dtype = z.real.dtype
    dz = lift(z) - lift(z_j)
    dz_wrapped_re = torch.remainder(dz.real + 0.5, 1.0) - 0.5
    dz_T = torch.complex(dz_wrapped_re.to(r_dtype), dz.imag.to(r_dtype))
    dd = delta_vec_d(vec_d, vec_d_j)
    Rmat = R(vec_d_j) if R_j is None else R_j
    Rext = R_ext(Rmat)
    w = iota(dz_T, dd)
    qv = q(w, Rext, sigma_par, sigma_perp)
    eta = rho(qv, q_max=q_max)
    weight = (alpha_j * eta.to(alpha_j.dtype)).unsqueeze(-1)
    T = (weight * T_hat_j).sum(dim=-2)

    return T


def Tau_dual(
    z: torch.Tensor,
    z_j: torch.Tensor,
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    T_hat_j: torch.Tensor,
    alpha_j: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
    k: torch.Tensor,
    R_j: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Tau backend: dual (Poisson) summation over integer frequencies.

    What it does
    ------------
    - Evaluates CPSF contribution in the reciprocal lattice.
    - Uses frequencies k in Z^N (router supplies symmetric set, e.g. {0} U {+-k}).
    - Computes angular factor exp(-pi * delta^H S0^{-1} delta), spectral weights
    exp(-pi * k^H S0 k), and complex phase from dz.
    - Takes real part of the spectral sum to obtain a real envelope eta.
    - Returns T(z, vec_d) = sum_j (alpha_j * eta_j) * T_hat_j with real weight.

    Where it is used
    ----------------
    - Fat kernels (large sigma_*): only a small number of harmonics matter.
    - High-dimensional regimes: K stays small (often K ~ 1 or 1+2N for R=1).

    Why no runtime checks here
    --------------------------
    - The router enforces k-shape [K, N], symmetry, truncation policy, and dtypes.
    - This function focuses on fast vectorized evaluation.

    Strengths
    ---------
    - Good scaling in N when sigma_* is large; compute and memory ~ O(M * K).
    - No spatial combinatorial explosion; numerically stable.

    Weaknesses
    ----------
    - Requires a frequency-set builder and truncation tolerance.
    - For very narrow kernels K grows and this backend becomes inefficient
    compared to nearest/classic.
    - Non-symmetric k may produce tiny imaginary residue (we drop via .real).
    """

    device = z.device
    c_dtype = z.dtype
    r_dtype = z.real.dtype
    N = z.shape[-1]
    dz = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rmat = R(vec_d_j) if R_j is None else R_j
    b = Rmat[..., :, 0]
    sp = torch.as_tensor(sigma_par, device=device, dtype=r_dtype)
    sq = torch.as_tensor(sigma_perp, device=device, dtype=r_dtype)
    a = 1.0 / sq
    c = (sp - sq) / (sp * sq + 0.0)
    dd_norm2 = (dd.abs() ** 2).sum(dim=-1)
    bh_dd = (torch.conj(b) * dd).sum(dim=-1)
    q_ang = a * dd_norm2 - c * (bh_dd.abs() ** 2)
    ang = torch.exp(-math.pi * q_ang)
    k_r = k.to(device=device, dtype=r_dtype)
    k_norm2 = (k_r**2).sum(dim=-1)
    k_c = torch.complex(k_r, torch.zeros_like(k_r))
    bh_k = (torch.conj(b)[..., None, :] * k_c.unsqueeze(0)).sum(dim=-1)
    q_k_a = sq[..., None] * k_norm2[None, :]
    q_k_b = (sp - sq)[..., None] * (bh_k.abs() ** 2)
    q_k = q_k_a + q_k_b
    w_k = torch.exp(-math.pi * q_k)
    phase_arg = torch.einsum("...jn,kn->...jk", dz.real, k_r)
    phase = torch.exp((2j * math.pi) * phase_arg.to(c_dtype))
    dual_sum = (w_k.to(c_dtype) * phase).sum(dim=-1).real
    det_sqrt = torch.sqrt((sq ** (N - 1)) * sp)
    eta = ang * det_sqrt * dual_sum
    weight = (alpha_j.to(r_dtype) * eta).unsqueeze(-1)
    T = (weight.to(c_dtype) * T_hat_j).sum(dim=-2)

    return T


def psi_over_offsets(
    z: torch.Tensor,
    z_j: torch.Tensor,
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
    offsets: torch.Tensor,
    R_j: Optional[torch.Tensor] = None,
    q_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Classic CPSF kernel over explicit lattice offsets.

    What it does
    ------------
    - Computes the torus-periodized envelope per contributor j by summing
    eta = rho(q(iota(dz + n, dd))) over the provided integer offsets n.
    - Offsets are applied to the REAL part of dz only; Im(dz) is left unchanged.
    - Broadcasts over contributors and offsets, returns the sum over offsets:
    eta_sum_j with shape [..., M].

    Where it is used
    ----------------
    - Core building block for the two classic backends:
    T_classic_window (one-shot window) and T_classic_full (streaming shells).
    - Neutral to the periodization policy: accepts any [O, N] batch of offsets.

    Why no runtime checks here
    --------------------------
    - All shape/dtype/iterator validations are delegated to the router.
    - This function assumes canonical inputs and focuses on fast vectorized math.

    Inputs (shapes)
    ---------------
    - z:        [..., N] complex
    - z_j:      [..., M, N] complex (or broadcastable to it)
    - vec_d:    [..., N] complex
    - vec_d_j:  [..., M, N] complex (or broadcastable)
    - sigma_par, sigma_perp: real, broadcastable
    - offsets:  [O, N] long (on the target device)
    - R_j:      optional precomputed geometry (broadcastable to [..., M, N, 2])
    - q_max:    optional scalar clamp for q before rho

    Output
    ------
    - eta_sum_j: [..., M] (typically real; depends on rho implementation)

    Strengths
    ---------
    - Fully vectorized over contributors and offsets (high throughput).
    - Simple contract; re-usable across periodization strategies.

    Weaknesses
    ----------
    - Temporarily materializes eta[..., M, O] before reduction over O.
    Memory scales as O(M * O); prefer calling it from a streaming loop with
    small shells/batches of offsets when O is large.
    """

    device = z.device
    r_dtype = z.real.dtype
    dz = lift(z) - lift(z_j)
    dd = delta_vec_d(vec_d, vec_d_j)
    Rmat = R(vec_d_j) if R_j is None else R_j
    Rext = R_ext(Rmat)
    off_r = offsets.to(device=device, dtype=r_dtype)
    dz_re = dz.real.unsqueeze(-2) + off_r.unsqueeze(-3)
    dz_im = dz.imag.unsqueeze(-2)
    dzb = torch.complex(dz_re, dz_im)
    dd_b = dd.unsqueeze(-2)
    w = iota(dzb, dd_b)
    qv = q(w, Rext.unsqueeze(-3), sigma_par, sigma_perp)
    eta = rho(qv, q_max=q_max)
    eta_sum_j = eta.sum(dim=-1)

    return eta_sum_j


def T_classic_window(
    z: torch.Tensor,
    z_j: torch.Tensor,
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    T_hat_j: torch.Tensor,
    alpha_j: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
    offsets_iterator: CPSFPsiOffsetsIterator,
    R_j: Optional[torch.Tensor] = None,
    q_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Classic CPSF field via WINDOW periodization (single-batch offsets).

    What it does
    ------------
    - Obtains exactly one batch of integer offsets [O, N] from offsets_iterator
    (e.g. produced by CPSFPeriodization(kind=WINDOW)).
    - Calls psi_over_offsets to sum eta over the window and aggregates
    T(z, vec_d) = sum_j (alpha_j * eta_j) * T_hat_j.

    Where it is used
    ----------------
    - Small-to-moderate N and window radius W so the full cube |n|_inf <= W
    fits into one vectorized pass.

    Why no runtime checks here
    --------------------------
    - Iterator discipline (exactly one batch) and input validations are handled
    by the router; this function assumes canonical inputs.

    Strengths
    ---------
    - One-shot execution; maximum vectorization; minimal Python overhead.
    - Accurate for the provided window.

    Weaknesses
    ----------
    - Peak memory ~ O(M * O) with O = (2W+1)^N due to temporary eta[..., M, O].
    - Accuracy depends on W: tails outside the window are truncated.
    Prefer FULL when you need robust tail control.
    """

    N = z.shape[-1]
    device = z.device
    eta_sum_j = psi_over_offsets(
        z=z,
        z_j=z_j,
        vec_d=vec_d,
        vec_d_j=vec_d_j,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
        offsets=next(offsets_iterator(N=N, device=device)),
        R_j=R_j,
        q_max=q_max,
    )
    c_dtype = z.dtype
    weight = (alpha_j.to(eta_sum_j.real.dtype) * eta_sum_j.real).to(c_dtype)
    weight = weight.unsqueeze(-1)
    T = (weight * T_hat_j.to(c_dtype)).sum(dim=-2)

    return T


def T_classic_full(
    z: torch.Tensor,
    z_j: torch.Tensor,
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    T_hat_j: torch.Tensor,
    alpha_j: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
    offsets_iterator: CPSFPsiOffsetsIterator,
    R_j: Optional[torch.Tensor] = None,
    q_max: Optional[float] = None,
    tol_abs: Optional[float] = None,
    tol_rel: Optional[float] = None,
    consecutive_below: int = 1,
) -> torch.Tensor:
    """
    Classic CPSF field via FULL periodization (streaming over shells).

    What it does
    ------------
    - Streams successive infinity-norm shells ||n||_inf = W from offsets_iterator
    (e.g. CPSFPeriodization(kind=FULL)).
    - For each shell, computes and immediately reduces psi_over_offsets.
    - Optionally applies early stopping via tol_abs / tol_rel when the shell
    contribution becomes negligible.
    - Returns accumulated T(z, vec_d).

    Where it is used
    ----------------
    - Default robust classic backend when window size is unknown or large,
    memory is tight, or evaluation is near torus boundaries.

    Why no runtime checks here
    --------------------------
    - The router validates iterator semantics and parameters; this function
    focuses on lean streaming accumulation.

    Strengths
    ---------
    - No thick materialization: peak memory ~ O(M * O_W) for the current shell.
    - Natural place for tolerance-based early stop; robust near boundaries.
    - Works uniformly across configurations (safe fallback).

    Weaknesses
    ----------
    - Runtime grows with the number of shells; in high dimensions shell size
    grows roughly as (2W+1)^N - (2W-1)^N ~ O(W^(N-1)).
    - Needs a meaningful tol_* or a finite max_radius to avoid oversumming.
    """

    N = z.shape[-1]
    device = z.device
    c_dtype = z.dtype
    T_acc = torch.zeros_like(T_hat_j[..., 0, :].to(c_dtype))
    below_count = 0
    accum_norm = torch.tensor(0.0, device=device, dtype=z.real.dtype)

    for offsets in offsets_iterator(N=N, device=device):
        eta_sum_j = psi_over_offsets(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            offsets=offsets,
            R_j=R_j,
            q_max=q_max,
        )

        weight_shell = (alpha_j.to(eta_sum_j.real.dtype) * eta_sum_j.real).to(c_dtype)
        T_shell = (weight_shell.unsqueeze(-1) * T_hat_j.to(c_dtype)).sum(dim=-2)
        T_acc = T_acc + T_shell

        if tol_abs is None and tol_rel is None:
            continue

        shell_norm = T_shell.abs().max()
        accum_norm = torch.maximum(accum_norm, T_acc.abs().max())
        cond_abs = (tol_abs is not None) and (
            shell_norm
            <= torch.as_tensor(tol_abs, device=device, dtype=shell_norm.dtype)
        )
        cond_rel = (tol_rel is not None) and (
            shell_norm
            <= (
                torch.as_tensor(tol_rel, device=device, dtype=shell_norm.dtype)
                * (accum_norm + 1e-30)
            )
        )

        if cond_abs or cond_rel:
            below_count += 1
        else:
            below_count = 0
        if below_count >= consecutive_below:
            break

    return T_acc
