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
        return torch.frac(x + 0.5) - 0.5

    def _norm(x, dim=-1, keepdim=False):
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)

    def _int_pow(base: torch.Tensor, n: int) -> torch.Tensor:
        # Возвращает base**n для ЦЕЛОГО n ≥ 0 через двоичное возведение в степень.
        # base: тензор любой формы; n: скаляр int. Без exp/log, только умножения.
        out = torch.ones_like(base)
        b = base
        k = int(n)
        while k > 0:
            if (k & 1) != 0:
                out = out * b
            b = b * b
            k >>= 1
        return out

    # ---- 1D GH cache (nodes + log-weights) ----
    if not hasattr(T_HS_Theta, "_gh1d_cache"):
        T_HS_Theta._gh1d_cache = {}
    gh_key = (device.type, getattr(device, "index", -1), str(r_dtype), int(quad_nodes))
    cached = T_HS_Theta._gh1d_cache.get(gh_key)
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
        logw_1d = torch.log(torch.clamp(w, min=tiny))            # (Q,)
        T_HS_Theta._gh1d_cache[gh_key] = (tau, logw_1d)
        tau_1d, logw_1d = tau, logw_1d
    else:
        tau_1d, logw_1d = cached
        Q = quad_nodes
    qc_full = Q * Q

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

    # ---- K_j via strict tail bound (detached) ----
    eps_theta = torch.as_tensor(eps_total, dtype=r_dtype, device=device) / (2.0 * max(N, 1) * max(qc_full, 1))
    a_det = a_j.detach()
    Kp_j = torch.ceil(torch.sqrt(torch.clamp((a_det / PI) * (-torch.log(torch.clamp(eps_theta, min=1e-30))), min=0.0))).to(torch.int64)

    # ---- angular factor ----
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

    # ---- mem targeting ----
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
        a_c       = a_j[m0:m1]
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

        # Δz: one frac
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

        # ---- bucket by K (detached) ----
        perm = torch.argsort(Kp_c_det)
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

        # ---- LSE accumulators (over q) ----
        A_lse = torch.full((B, mc), -float("inf"), device=device, dtype=r_dtype)
        S_lse = torch.zeros((B, mc), device=device, dtype=r_dtype)

        # choose ns_sub budget
        per_elem_bufs = 2
        ns_cap_den = max(1, B * mc * qc_full)
        ns_cap = max(1, int(target_bytes // (bytes_per_elem * per_elem_bufs * ns_cap_den)))
        step_ns = min(n_chunk, max(1, ns_cap))

        log_inv_sqrt_a = -0.5 * torch.log(torch.clamp(a_c, min=tiny))  # (mc,)

        buf_inv_a = 1.0 / torch.clamp(a_c, min=tiny)
        buf_ex01 = torch.exp(-PI * buf_inv_a)
        buf_ex02 = buf_ex01 * buf_ex01
        buf_ex04 = buf_ex02 * buf_ex02
        buf_ex08 = buf_ex04 * buf_ex04
        buf_ex16 = buf_ex08 * buf_ex08

        # ======================= n-tiles =======================
        for n0 in range(0, N, step_ns):
            n1 = min(n0 + step_ns, N)
            ns = n1 - n0

            # A = 2π Δz : (B,mc,ns)
            Aphase = TWO_PI * dz[:, :, n0:n1]
            cA = torch.cos(Aphase)
            sA = torch.sin(Aphase)

            # φ = 2π * t1 * bR_eff[:,ns], ψ = 2π * t2 * bI_eff[:,ns]
            phi = TWO_PI * (bR_eff[:, n0:n1].unsqueeze(-1) * tau_1d.view(1, 1, -1))  # (mc,ns,Q)
            psi = TWO_PI * (bI_eff[:, n0:n1].unsqueeze(-1) * tau_1d.view(1, 1, -1))  # (mc,ns,Q)
            cphi = torch.cos(phi)
            sphi = torch.sin(phi)
            cpsi = torch.cos(psi)
            spsi = torch.sin(psi)

            # q-tiling in 2D (q1 × q2)
            denom = max(1, B * mc * ns)
            q_cap = max(1, int(target_bytes // (bytes_per_elem * per_elem_bufs * denom)))
            base = max(1, int(math.sqrt(q_cap)))
            q1_step = max(1, min(Q, base))
            q2_step = max(1, min(Q, max(1, q_cap // q1_step)))

            # ======================= (q1,q2)-tiles =======================
            for i0 in range(0, Q, q1_step):
                i1 = min(i0 + q1_step, Q)
                q1 = i1 - i0

                cphi_i = cphi[:, :, i0:i1]     # (mc,ns,q1)
                sphi_i = sphi[:, :, i0:i1]
                logw_i = logw_1d[i0:i1]        # (q1,)

                for j0 in range(0, Q, q2_step):
                    j1 = min(j0 + q2_step, Q)
                    q2 = j1 - j0

                    cpsi_j = cpsi[:, :, j0:j1] # (mc,ns,q2)
                    spsi_j = spsi[:, :, j0:j1]
                    logw_j = logw_1d[j0:j1]    # (q2,)

                    # FIX: use unsqueeze(-2) so shapes broadcast as (mc,ns,q1,1)*(mc,ns,1,q2)
                    cB = cphi_i.unsqueeze(-1) * cpsi_j.unsqueeze(-2) - sphi_i.unsqueeze(-1) * spsi_j.unsqueeze(-2)
                    sB = sphi_i.unsqueeze(-1) * cpsi_j.unsqueeze(-2) + cphi_i.unsqueeze(-1) * spsi_j.unsqueeze(-2)

                    # x = cos(A - B) = cosA*cosB + sinA*sinB  → (B,mc,ns,q1,q2)
                    x_block = (cA.unsqueeze(-1).unsqueeze(-1) * cB.unsqueeze(0) + sA.unsqueeze(-1).unsqueeze(-1) * sB.unsqueeze(0)).to(r_dtype)

                    # flatten q1×q2 → ql
                    x_flat = x_block.reshape(B, mc, ns, q1 * q2)  # (B,mc,ns,ql)
                    lw_blk = (logw_i.view(1, 1, q1, 1) + logw_j.view(1, 1, 1, q2)).reshape(1, 1, q1 * q2)

                    # ========= process K-buckets (Chebyshev for K<=4, else Clenshaw) =========
                    for g in range(num_groups):
                        Kval = int(Kvals[g].item())
                        s = int(starts[g].item())
                        e = s + int(counts[g].item())
                        xg = x_flat[:, s:e, :ns, :]

                        if Kval == 0:
                            part = (ns * log_inv_sqrt_a[s:e].view(1, e - s, 1)).expand(B, e - s, q1 * q2)

                        elif Kval <= 4:
                            x2 = xg * xg
                            S = buf_ex01[s:e].view(1, e - s, 1, 1) * xg
                            if Kval >= 2:
                                S = S + buf_ex04[s:e].view(1, e - s, 1, 1) * (2.0 * x2 - 1.0)
                            if Kval >= 3:
                                S = S + (buf_ex08[s:e].view(1, e - s, 1, 1) * buf_ex01[s:e].view(1, e - s, 1, 1)) * (4.0 * x2 * xg - 3.0 * xg)
                            if Kval == 4:
                                S = S + buf_ex16[s:e].view(1, e - s, 1, 1) * (8.0 * x2 * x2 - 8.0 * x2 + 1.0)

                            sum_expr = torch.log1p(torch.clamp(2.0 * S, min=-1.0 + tiny))
                            part = sum_expr.sum(dim=2) + (ns * log_inv_sqrt_a[s:e].view(1, e - s, 1))

                        else:
                            m_g = e - s
                            e1   = buf_ex01[s:e].view(1, m_g, 1, 1)
                            rho  = buf_ex02[s:e].view(1, m_g, 1, 1)
                            rKm1 = e1 * _int_pow(rho, Kval - 1)

                            if (Kval & 1) == 0:
                                cK = _int_pow(rho, (Kval * Kval) // 2)
                            else:
                                cK = e1 * _int_pow(rho, (Kval * Kval - 1) // 2)

                            b1 = torch.zeros_like(xg, dtype=r_dtype)
                            b2 = torch.zeros_like(xg, dtype=r_dtype)
                            c  = cK.clone()
                            r  = rKm1.clone()

                            for _ in range(Kval, 0, -1):
                                b0 = 2.0 * xg * b1 - b2 + c
                                b2, b1 = b1, b0
                                c = c / r
                                r = r / rho
                                tail = (c * r) / torch.clamp(1.0 - r, min=1e-12)
                                if float(tail.max().detach()) < float(eps_theta * 0.125):
                                    break

                            S = b1 * xg - b2
                            sum_expr = torch.log1p(torch.clamp(2.0 * S, min=-1.0 + tiny))
                            part = sum_expr.sum(dim=2) + (ns * log_inv_sqrt_a[s:e].view(1, m_g, 1))

                        L = part + lw_blk
                        Lmax = L.max(dim=-1).values
                        Mx = torch.maximum(A_lse[:, s:e], Lmax)
                        S_lse[:, s:e] = S_lse[:, s:e] * torch.exp(A_lse[:, s:e] - Mx) + torch.exp(L - Mx.unsqueeze(-1)).sum(dim=-1)
                        A_lse[:, s:e] = Mx

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
