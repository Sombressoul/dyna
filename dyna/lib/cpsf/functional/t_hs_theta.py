import math
import torch

@torch.no_grad()
def T_HS_Theta(
    z,
    z_j,
    vec_d,
    vec_d_j,
    T_hat_j,
    alpha_j,
    sigma_par,
    sigma_perp,
    *,
    quad_nodes: int = 12,
    theta_mode: str = "poisson",   # игнорируется
    eps_total: float = 1.0e-3,
    a_threshold: float = 1.0,      # игнорируется
    n_chunk: int = 64,
    m_chunk: int = 65536,
    dtype_override: torch.dtype | None = None,
):
    device = z.device
    c_dtype = z.dtype if dtype_override is None else dtype_override
    r_dtype = torch.float32 if c_dtype == torch.complex64 else torch.float64
    PI = math.pi
    TWO_PI = 2.0 * math.pi
    tiny = torch.as_tensor(torch.finfo(r_dtype).tiny, device=device, dtype=r_dtype)

    def _wrap01(x):
        return torch.remainder(x + 0.5, 1.0) - 0.5

    def _norm(x, dim=-1, keepdim=False):
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)

    # ---------- GH cache (Q×Q -> flatten) ----------
    if not hasattr(T_HS_Theta, "_gh_cache"):
        T_HS_Theta._gh_cache = {}
    gh_key = (device.type, getattr(device, "index", -1), str(r_dtype), int(quad_nodes))
    cached = T_HS_Theta._gh_cache.get(gh_key)
    if cached is None:
        Q = quad_nodes
        if Q < 1:
            raise ValueError("quad_nodes must be >= 1")
        k = torch.arange(1, Q, device=device, dtype=r_dtype)
        off = torch.sqrt(k / 2.0)
        J = torch.zeros((Q, Q), device=device, dtype=r_dtype)
        J.diagonal(1).copy_(off)
        J.diagonal(-1).copy_(off)
        tau, V = torch.linalg.eigh(J)                            # (Q,)
        w = (math.sqrt(math.pi) * (V[0, :] ** 2)).to(r_dtype)    # (Q,)
        t1_flat = tau.view(-1, 1).expand(-1, Q).reshape(-1)      # (Q^2,)
        t2_flat = tau.view(1, -1).expand(Q, -1).reshape(-1)
        logw2_flat = (torch.log(torch.clamp(w, min=tiny)).view(-1, 1) +
                      torch.log(torch.clamp(w, min=tiny)).view(1, -1)).reshape(-1)
        T_HS_Theta._gh_cache[gh_key] = (t1_flat, t2_flat, logw2_flat)
        t1_flat, t2_flat, logw2_flat = T_HS_Theta._gh_cache[gh_key]
    else:
        t1_flat, t2_flat, logw2_flat = cached
        Q = quad_nodes
    qc_full = t1_flat.numel()  # всегда берём весь Q^2 за раз

    # ---------- shapes & casts ----------
    B, N = z.shape[-2], z.shape[-1]
    M = z_j.shape[-2]
    S_dim = T_hat_j.shape[-1]

    z         = z.to(c_dtype)
    z_j       = z_j.to(c_dtype)
    vec_d     = vec_d.to(c_dtype)
    vec_d_j   = vec_d_j.to(c_dtype)
    T_hat_j   = T_hat_j.to(c_dtype)
    alpha_j   = alpha_j.to(r_dtype)
    sigma_par = sigma_par.to(r_dtype)
    sigma_perp= sigma_perp.to(r_dtype)

    # ---------- anisotropy & θ (Poisson only) ----------
    a_j = 1.0 / torch.clamp(sigma_perp, min=tiny)                           # (M,)
    gamma_j = torch.clamp(sigma_perp / torch.clamp(sigma_par, min=tiny), min=tiny, max=1.0)
    kappa_j = torch.sqrt(torch.clamp((sigma_par - sigma_perp) * torch.clamp(sigma_perp, min=tiny)
                                     / (PI * torch.clamp(sigma_par, min=tiny)), min=0.0))
    inv_sqrt_gamma = 1.0 / torch.sqrt(torch.clamp(gamma_j, min=tiny))
    kappa_eff = kappa_j * inv_sqrt_gamma                                   # (M,)
    norm_fac = (1.0 / PI) * inv_sqrt_gamma                                 # (M,)

    # целевая точность → усечение Kp_j
    eps_theta = torch.as_tensor(eps_total, dtype=r_dtype, device=device) / (2.0 * Q * Q * max(N, 1))
    Kp_j = torch.ceil(torch.sqrt(torch.clamp((a_j / PI) * (-torch.log(torch.clamp(eps_theta, min=1e-30))), min=0.0))).to(torch.int64)  # (M,)

    # ---------- angular factor (Tau_dual-consistent) ----------
    one_over_sq = 1.0 / torch.clamp(sigma_perp, min=tiny)
    c_ang = (sigma_par - sigma_perp) / (torch.clamp(sigma_par * sigma_perp, min=tiny))
    try:
        from dyna.lib.cpsf.functional.core_math import delta_vec_d as _delta_vec_d  # type: ignore
        use_delta = True
    except Exception:
        try:
            from .core_math import delta_vec_d as _delta_vec_d  # type: ignore
            use_delta = True
        except Exception:
            use_delta = False

    # ---------- memory targeting for ns_sub ----------
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        target_bytes = int(max(256 * 1024 * 1024, min(1_000 * 1024 * 1024, int(free_bytes * 0.40))))
    else:
        target_bytes = 256 * 1024 * 1024
    bytes_per_elem = 4 if r_dtype == torch.float32 else 8

    # ---------- output (2×SGEMM; TF32 OFF локально) ----------
    T_out = torch.zeros((B, S_dim), device=device, dtype=c_dtype)

    # ---------- main loop over m-chunks ----------
    for m0 in range(0, M, m_chunk):
        m1 = min(m0 + m_chunk, M)
        mc = m1 - m0

        # slice per-chunk
        z_j_c     = z_j[m0:m1]
        a_c       = a_j[m0:m1]                     # (mc,)
        alpha_c   = alpha_j[m0:m1]
        T_hat_c   = T_hat_j[m0:m1].contiguous()
        Kp_c      = Kp_j[m0:m1]                     # (mc,)
        kappa_eff_c = kappa_eff[m0:m1]              # (mc,)
        norm_fac_c  = norm_fac[m0:m1]               # (mc,)

        # directions (unit) and embed κ
        den = _norm(vec_d_j[m0:m1], dim=-1, keepdim=True).to(r_dtype).clamp_min(torch.as_tensor(1e-20, device=device, dtype=r_dtype))
        b = vec_d_j[m0:m1] / den.to(c_dtype)
        bR = b.real.to(r_dtype).contiguous()        # (mc,N)
        bI = b.imag.to(r_dtype).contiguous()
        bR_eff = (kappa_eff_c.view(-1, 1) * bR).contiguous()     # (mc,N)
        bI_eff = (kappa_eff_c.view(-1, 1) * bI).contiguous()

        # wrapped displacements
        dz = _wrap01((z.unsqueeze(1) - z_j_c.unsqueeze(0)).real.to(r_dtype)).contiguous()  # (B,mc,N)

        # angular factor (B,mc)
        if use_delta:
            vdb = vec_d.unsqueeze(1).expand(B, mc, N)
            vjj = vec_d_j[m0:m1].unsqueeze(0).expand(B, mc, N)
            dd  = _delta_vec_d(vdb, vjj)
        else:
            dd  = (vec_d.unsqueeze(1) - vec_d_j[m0:m1].unsqueeze(0))
        dd_norm2 = (dd.abs() ** 2).sum(dim=-1).to(r_dtype)
        bh_dd = (torch.conj(b).unsqueeze(0) * dd).sum(dim=-1)
        q_ang   = one_over_sq[m0:m1].unsqueeze(0) * dd_norm2 - c_ang[m0:m1].unsqueeze(0) * (bh_dd.abs() ** 2)
        ang_fac = torch.exp(-PI * q_ang).to(r_dtype)        # (B,mc)

        # ---------- sort/bucket by Kp to avoid masks in Clenshaw ----------
        perm = torch.argsort(Kp_c)                 # (mc,)
        invp = torch.empty_like(perm)
        invp[perm] = torch.arange(mc, device=device)
        # permute all per-j arrays along mc
        a_c        = a_c[perm]
        alpha_c    = alpha_c[perm]
        T_hat_c    = T_hat_c[perm, :]
        Kp_c       = Kp_c[perm]
        kappa_eff_c= kappa_eff_c[perm]
        norm_fac_c = norm_fac_c[perm]
        bR_eff     = bR_eff[perm, :]
        bI_eff     = bI_eff[perm, :]
        dz         = dz[:, perm, :]
        ang_fac    = ang_fac[:, perm]

        # groups
        if mc > 0:
            Kvals, counts = torch.unique_consecutive(Kp_c, return_counts=True)
            starts = torch.cumsum(torch.cat([torch.zeros(1, device=device, dtype=counts.dtype), counts[:-1]]), dim=0)
            num_groups = Kvals.numel()
        else:
            num_groups = 0

        # LSE accumulators over q (закрываем на каждом ns-сабтайле)
        A_lse = torch.full((B, mc), -float("inf"), device=device, dtype=r_dtype)
        S_lse = torch.zeros((B, mc), device=device, dtype=r_dtype)

        # one-shot q vectors
        t1c = t1_flat.to(r_dtype)          # (qc,)
        t2c = t2_flat.to(r_dtype)
        lwc = logw2_flat.to(r_dtype)       # (qc,)

        # choose ns_sub to keep 2 buffers (B,mc,ns,qc)
        per_elem_bufs = 2
        ns_cap_den = max(1, B * mc * qc_full)
        ns_cap = max(1, int(target_bytes // (bytes_per_elem * per_elem_bufs * ns_cap_den)))
        step_ns = min(n_chunk, max(1, ns_cap))

        # preallocate Clenshaw buffers for max group width (reuse)
        b1_buf = torch.zeros((B, mc, step_ns, qc_full), device=device, dtype=r_dtype)
        b2_buf = torch.zeros_like(b1_buf)

        # log factor outside Σ_ns
        log_inv_sqrt_a = -0.5 * torch.log(torch.clamp(a_c, min=tiny))  # (mc,)

        for n0 in range(0, N, step_ns):
            n1 = min(n0 + step_ns, N)
            ns = n1 - n0

            # shift = t1*bR_eff + t2*bI_eff  (broadcast)
            sR = bR_eff[:, n0:n1].unsqueeze(-1) * t1c.view(1, 1, -1)   # (mc,ns,qc)
            sI = bI_eff[:, n0:n1].unsqueeze(-1) * t2c.view(1, 1, -1)
            shift = (sR + sI).contiguous()                              # (mc,ns,qc)

            # u and cosx
            u = _wrap01(dz[:, :, n0:n1].unsqueeze(-1) - shift.unsqueeze(0))   # (B,mc,ns,qc)
            cosx = torch.cos(TWO_PI * u).contiguous()

            # process Kp-buckets without masks inside k-loop
            for g in range(num_groups):
                Kval = int(Kvals[g].item())
                s = int(starts[g].item())
                e = s + int(counts[g].item())
                if Kval == 0:
                    # sum_ns logθ = ns*(-½ log a)
                    part = (ns * log_inv_sqrt_a[s:e].view(1, e - s, 1)).expand(B, e - s, qc_full)
                else:
                    # Clenshaw on slice (no masks): prepare backward ladder c_k and r_k-1
                    inv_a = 1.0 / torch.clamp(a_c[s:e], min=tiny)             # (m_g,)
                    Kf = float(Kval)
                    cK  = torch.exp((-PI * (Kf * Kf)) * inv_a)                # c_K
                    rKm1= torch.exp((-(2.0 * Kf - 1.0) * PI) * inv_a)         # r_{K-1}
                    rho = torch.exp(( -2.0 * PI) * inv_a)                     # rho

                    # Clenshaw buffers (views) and zero_
                    b1 = b1_buf[:, s:e, :ns, :]; b1.zero_()
                    b2 = b2_buf[:, s:e, :ns, :]; b2.zero_()
                    cosx_g = cosx[:, s:e, :ns, :]

                    c = cK.view(1, e - s, 1, 1).to(r_dtype)
                    r = rKm1.view(1, e - s, 1, 1).to(r_dtype)
                    rho_v = rho.view(1, e - s, 1, 1).to(r_dtype)

                    for _ in range(Kval, 0, -1):
                        # b0 = 2*cosx*b1 - b2 + c
                        b0 = torch.addcmul(-b2, cosx_g, b1, value=2.0)
                        b0.add_(c)
                        b2, b1 = b1, b0
                        # c_{k-1} = c_k / r_{k-1};  r_{k-2} = r_{k-1} / rho
                        c = c / r
                        r = r / rho_v

                    S = b1 * cosx_g - b2
                    sum_expr = S.mul_(2.0).add_(1.0)   # 1 + 2S
                    sum_expr.clamp_(min=tiny).log_()   # log(1+2S)
                    part = sum_expr.sum(dim=2).add_(ns * log_inv_sqrt_a[s:e].view(1, e - s, 1))

                # LSE update for this bucket
                L = part.add_(lwc.view(1, 1, -1))                 # (B,m_g,qc)
                Lmax = L.max(dim=-1).values                       # (B,m_g)
                A_blk = A_lse[:, s:e]; S_blk = S_lse[:, s:e]
                Mx = torch.maximum(A_blk, Lmax)
                S_blk = S_blk.mul_(torch.exp(A_blk - Mx)).add_(torch.exp(L - Mx.unsqueeze(-1)).sum(dim=-1))
                A_lse[:, s:e] = Mx
                S_lse[:, s:e] = S_blk

        # η и финальный комбинированный вес; 2×SGEMM (TF32 off локально)
        eta = (norm_fac_c.view(1, mc) * torch.exp(A_lse) * S_lse).reshape(B, mc)
        weight = (alpha_c.reshape(1, mc) * ang_fac.reshape(B, mc) * eta).to(torch.float32)

        R = T_hat_c.real.contiguous().to(torch.float32)  # (mc,S)
        I = T_hat_c.imag.contiguous().to(torch.float32)

        if device.type == "cuda":
            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                T_out.real.addmm_(weight, R)
                T_out.imag.addmm_(weight, I)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        else:
            T_out.real.addmm_(weight, R)
            T_out.imag.addmm_(weight, I)

    return T_out
