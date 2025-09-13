import math
import torch


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
    eps_total: float = 1.0e-3,
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

    def _frac01(x):
        # одинарная нормализация только по Δz (A); далее cos() без повторного wrap
        return torch.frac(x + 0.5) - 0.5

    def _norm(x, dim=-1, keepdim=False):
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)

    # ---- 2D GH cache (плоский список узлов и лог-весов) ----
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
        logw2_flat = (
            torch.log(torch.clamp(w, min=tiny)).view(-1, 1)
            + torch.log(torch.clamp(w, min=tiny)).view(1, -1)
        ).reshape(-1)
        T_HS_Theta._gh_cache[gh_key] = (t1_flat, t2_flat, logw2_flat)
        t1_flat, t2_flat, logw2_flat = T_HS_Theta._gh_cache[gh_key]
    else:
        t1_flat, t2_flat, logw2_flat = cached
        Q = quad_nodes
    qc_full = t1_flat.numel()

    # ---- shapes / casts ----
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

    # ---- anisotropy (Poisson only) ----
    a_j = 1.0 / torch.clamp(sigma_perp, min=tiny)                           # (M,)
    gamma_j = torch.clamp(sigma_perp / torch.clamp(sigma_par, min=tiny), min=tiny, max=1.0)
    kappa_j = torch.sqrt(torch.clamp((sigma_par - sigma_perp) * torch.clamp(sigma_perp, min=tiny)
                                     / (PI * torch.clamp(sigma_par, min=tiny)), min=0.0))
    inv_sqrt_gamma = 1.0 / torch.sqrt(torch.clamp(gamma_j, min=tiny))
    kappa_eff = kappa_j * inv_sqrt_gamma                                   # (M,)
    norm_fac = (1.0 / PI) * inv_sqrt_gamma                                 # (M,)

    # ---- K_j из строгой мажоранты (E), считаем от detachd a ----
    eps_theta = torch.as_tensor(eps_total, dtype=r_dtype, device=device) / (2.0 * max(N, 1) * max(qc_full, 1))
    a_det = a_j.detach()
    # решаем грубой инверсией: ceil( sqrt( (a/π)*(-log ε_θ) ) )
    Kp_j = torch.ceil(torch.sqrt(torch.clamp((a_det / PI) * (-torch.log(torch.clamp(eps_theta, min=1e-30))), min=0.0))).to(torch.int64)

    # ---- angular factor (Tau_dual-consistent) ----
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

    # ---- memory targeting for ns_sub ----
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        target_bytes = int(max(256 * 1024 * 1024, min(1_000 * 1024 * 1024, int(free_bytes * 0.40))))
    else:
        target_bytes = 256 * 1024 * 1024
    bytes_per_elem = 4 if r_dtype == torch.float32 else 8

    # ---- output ----
    T_out = torch.zeros((B, S_dim), device=device, dtype=c_dtype)

    # ======================= main m-chunks =======================
    for m0 in range(0, M, m_chunk):
        m1 = min(m0 + m_chunk, M)
        mc = m1 - m0

        z_j_c     = z_j[m0:m1]
        a_c       = a_j[m0:m1]                     # (mc,)
        alpha_c   = alpha_j[m0:m1]
        T_hat_c   = T_hat_j[m0:m1]
        Kp_c_det  = Kp_j[m0:m1]
        kappa_eff_c = kappa_eff[m0:m1]
        norm_fac_c  = norm_fac[m0:m1]

        # unit directions and κ embedding
        den = _norm(vec_d_j[m0:m1], dim=-1, keepdim=True).to(r_dtype).clamp_min(torch.as_tensor(1e-20, device=device, dtype=r_dtype))
        b = vec_d_j[m0:m1] / den.to(c_dtype)
        bR = b.real.to(r_dtype)
        bI = b.imag.to(r_dtype)
        bR_eff = (kappa_eff_c.view(-1, 1) * bR).contiguous()     # (mc,N)
        bI_eff = (kappa_eff_c.view(-1, 1) * bI).contiguous()

        # Δz: единственное frac (A), далее без wrap
        dz = _frac01((z.unsqueeze(1) - z_j_c.unsqueeze(0)).real.to(r_dtype))  # (B,mc,N)

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

        # ---- bucket by K (детачен) ----
        perm = torch.argsort(Kp_c_det)
        # permute per-j arrays
        a_c        = a_c[perm]
        alpha_c    = alpha_c[perm]
        T_hat_c    = T_hat_c[perm, :]
        Kp_c       = Kp_c_det[perm]
        kappa_eff_c= kappa_eff_c[perm]
        norm_fac_c = norm_fac_c[perm]
        bR_eff     = bR_eff[perm, :]
        bI_eff     = bI_eff[perm, :]
        dz         = dz[:, perm, :]
        ang_fac    = ang_fac[:, perm]

        if mc > 0:
            Kvals, counts = torch.unique_consecutive(Kp_c, return_counts=True)
            starts = torch.cumsum(torch.cat([torch.zeros(1, device=device, dtype=counts.dtype), counts[:-1]]), dim=0)
            num_groups = Kvals.numel()
        else:
            num_groups = 0

        # ---- LSE over q (accumulators) ----
        A_lse = torch.full((B, mc), -float("inf"), device=device, dtype=r_dtype)
        S_lse = torch.zeros((B, mc), device=device, dtype=r_dtype)

        # one-shot q arrays
        t1c = t1_flat.to(r_dtype)          # (qc,)
        t2c = t2_flat.to(r_dtype)
        lwc = logw2_flat.to(r_dtype)       # (qc,)

        # choose ns_sub (2 buffers B,mc,ns,qc)
        per_elem_bufs = 2
        ns_cap_den = max(1, B * mc * qc_full)
        ns_cap = max(1, int(target_bytes // (bytes_per_elem * per_elem_bufs * ns_cap_den)))
        step_ns = min(n_chunk, max(1, ns_cap))

        # precompute log a^{-1/2} outside Σ_ns
        log_inv_sqrt_a = -0.5 * torch.log(torch.clamp(a_c, min=tiny))  # (mc,)

        # ======================= n-tiles =======================
        for n0 in range(0, N, step_ns):
            n1 = min(n0 + step_ns, N)
            ns = n1 - n0

            # shift = t1*bR_eff + t2*bI_eff (без повторного wrap)
            sR = bR_eff[:, n0:n1].unsqueeze(-1) * t1c.view(1, 1, -1)   # (mc,ns,qc)
            sI = bI_eff[:, n0:n1].unsqueeze(-1) * t2c.view(1, 1, -1)
            shift = (sR + sI)                                          # (mc,ns,qc)

            # theta = 2π(Δz - shift), x = cos(theta)
            theta = TWO_PI * (dz[:, :, n0:n1].unsqueeze(-1) - shift.unsqueeze(0))  # (B,mc,ns,qc)
            x = torch.cos(theta).to(r_dtype)                                          # (B,mc,ns,qc)

            # ========= process K-buckets (Chebyshev for K<=4, else Clenshaw) =========
            for g in range(num_groups):
                Kval = int(Kvals[g].item())
                s = int(starts[g].item())
                e = s + int(counts[g].item())
                xg = x[:, s:e, :ns, :]                              # (B,m_g,ns,qc)

                if Kval == 0:
                    part = (ns * log_inv_sqrt_a[s:e].view(1, e - s, 1)).expand(B, e - s, qc_full)

                elif Kval <= 4:
                    # Chebyshev closed-form (D)
                    inv_a = 1.0 / torch.clamp(a_c[s:e], min=tiny)   # (m_g,)
                    c1  = torch.exp((-PI * 1.0) * inv_a).view(1, e - s, 1, 1)
                    if Kval >= 2:
                        r1  = torch.exp((-3.0 * PI) * inv_a).view(1, e - s, 1, 1)
                        c2  = c1 * r1
                    if Kval >= 3:
                        rho = torch.exp((-2.0 * PI) * inv_a).view(1, e - s, 1, 1)
                        r2  = r1 * rho
                        c3  = c2 * r2
                    if Kval == 4:
                        r3  = r2 * rho
                        c4  = c3 * r3

                    T1 = xg
                    x2 = xg * xg
                    T2 = 2.0 * x2 - 1.0
                    T3 = 4.0 * x2 * xg - 3.0 * xg
                    T4 = 8.0 * x2 * x2 - 8.0 * x2 + 1.0

                    S = c1 * T1
                    if Kval >= 2: S = S + c2 * T2
                    if Kval >= 3: S = S + c3 * T3
                    if Kval == 4: S = S + c4 * T4

                    sum_expr = torch.log1p(torch.clamp(2.0 * S, min=-1.0 + tiny))
                    part = sum_expr.sum(dim=2) + (ns * log_inv_sqrt_a[s:e].view(1, e - s, 1))

                else:
                    # Clenshaw with strict tail bound (E) and “ladder” (C), backward from k=K..1
                    inv_a = 1.0 / torch.clamp(a_c[s:e], min=tiny)             # (m_g,)
                    Kf = float(Kval)
                    cK  = torch.exp((-PI * (Kf * Kf)) * inv_a).view(1, e - s, 1, 1)
                    rKm1= torch.exp((-(2.0 * Kf - 1.0) * PI) * inv_a).view(1, e - s, 1, 1)
                    rho = torch.exp(( -2.0 * PI) * inv_a).view(1, e - s, 1, 1)

                    b1 = torch.zeros_like(xg, dtype=r_dtype)
                    b2 = torch.zeros_like(xg, dtype=r_dtype)
                    c = cK.clone()
                    r = rKm1.clone()

                    for _ in range(Kval, 0, -1):
                        b0 = 2.0 * xg * b1 - b2 + c
                        b2, b1 = b1, b0
                        # “ladder” шаг назад:
                        c = c / r
                        r = r / rho
                        # строгая мажоранта хвоста: tail ≈ c * r / (1 - r)
                        tail = (c * r) / torch.clamp(1.0 - r, min=1e-12)
                        if float(tail.max().detach()) < float(eps_theta * 0.125):
                            break

                    S = b1 * xg - b2
                    sum_expr = torch.log1p(torch.clamp(2.0 * S, min=-1.0 + tiny))
                    part = sum_expr.sum(dim=2) + (ns * log_inv_sqrt_a[s:e].view(1, e - s, 1))

                # LSE update on q for this bucket
                L = part + lwc.view(1, 1, -1)                     # (B,m_g,qc)
                Lmax = L.max(dim=-1).values                       # (B,m_g)
                A_blk = A_lse[:, s:e]; S_blk = S_lse[:, s:e]
                Mx = torch.maximum(A_blk, Lmax)
                S_new = S_blk * torch.exp(A_blk - Mx) + torch.exp(L - Mx.unsqueeze(-1)).sum(dim=-1)
                if s == 0 and e == mc:
                    A_lse, S_lse = Mx, S_new
                else:
                    A_lse = torch.cat([A_lse[:, :s], Mx, A_lse[:, e:]], dim=1)
                    S_lse = torch.cat([S_lse[:, :s], S_new, S_lse[:, e:]], dim=1)

        # η и финальный вес; 2×SGEMM (TF32 off локально)
        eta = (norm_fac_c.view(1, mc) * torch.exp(A_lse) * S_lse).reshape(B, mc)
        weight = (alpha_c.reshape(1, mc) * ang_fac.reshape(B, mc) * eta).to(torch.float32)

        R = T_hat_c.real.to(torch.float32)  # (mc,S)
        I = T_hat_c.imag.to(torch.float32)

        if device.type == "cuda":
            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                T_out.real = T_out.real + weight @ R
                T_out.imag = T_out.imag + weight @ I
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        else:
            T_out.real = T_out.real + weight @ R
            T_out.imag = T_out.imag + weight @ I

    return T_out
