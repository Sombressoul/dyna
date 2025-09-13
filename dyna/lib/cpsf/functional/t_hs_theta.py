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
    theta_mode: str = "auto",
    eps_total: float = 1.0e-3,
    a_threshold: float = 1.0,
    n_chunk: int = 64,
    m_chunk: int = 65536,
    dtype_override: torch.dtype | None = None,
):
    """
    HS-Theta (оптимизировано без изменения семантики):
      - GH-узлы/веса 2D кэшируются по (device,dtype,Q)
      - Единый расчёт u с wrap в [-1/2,1/2)
      - Poisson θ: векторизованный Clenshaw для Σ c_k cos(k·(2πu)), один проход по k
      - Direct θ: cosh-вариант в лог-домене (устойчивый), ns-сабчанк
      - Online-LSE по q, финальный GEMM (cuBLAS)
    """
    device = z.device
    c_dtype = z.dtype if dtype_override is None else dtype_override
    r_dtype = torch.float32 if c_dtype == torch.complex64 else torch.float64
    tiny = torch.as_tensor(torch.finfo(r_dtype).tiny, device=device, dtype=r_dtype)
    PI = math.pi

    def _wrap01(x):  # map to [-1/2, 1/2)
        return torch.remainder(x + 0.5, 1.0) - 0.5

    def _norm(x, dim=-1, keepdim=False):
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)

    # ---------- GH cache ----------
    if not hasattr(T_HS_Theta, "_gh_cache"):
        T_HS_Theta._gh_cache = {}
    gh_key = (device.type, getattr(device, "index", -1), str(r_dtype), int(quad_nodes))
    cached = T_HS_Theta._gh_cache.get(gh_key)
    if cached is None:
        K = quad_nodes
        if K < 1:
            raise ValueError("quad_nodes must be >= 1")
        k = torch.arange(1, K, device=device, dtype=r_dtype)
        off = torch.sqrt(k / 2.0)
        J = torch.zeros((K, K), device=device, dtype=r_dtype)
        J.diagonal(1).copy_(off)
        J.diagonal(-1).copy_(off)
        tau, V = torch.linalg.eigh(J)  # ascending
        w = (math.sqrt(math.pi) * (V[0, :] ** 2)).to(r_dtype)
        t1 = tau.view(-1, 1).expand(-1, K)
        t2 = tau.view(1, -1).expand(K, -1)
        t1_flat = t1.reshape(-1)
        t2_flat = t2.reshape(-1)
        logw2_flat = (torch.log(torch.clamp(w, min=tiny)).view(-1, 1) +
                      torch.log(torch.clamp(w, min=tiny)).view(1, -1)).reshape(-1)
        T_HS_Theta._gh_cache[gh_key] = (t1_flat, t2_flat, logw2_flat)
        t1_flat, t2_flat, logw2_flat = T_HS_Theta._gh_cache[gh_key]
    else:
        t1_flat, t2_flat, logw2_flat = cached

    # ---------- casts & shapes ----------
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

    # ---------- anisotropy & θ-params ----------
    a_j = 1.0 / torch.clamp(sigma_perp, min=tiny)  # a = 1/σ⊥
    gamma_j = torch.clamp(sigma_perp / torch.clamp(sigma_par, min=tiny), min=tiny, max=1.0)
    kappa_j = torch.sqrt(torch.clamp((sigma_par - sigma_perp) * torch.clamp(sigma_perp, min=tiny) / (PI * torch.clamp(sigma_par, min=tiny)), min=0.0))

    def _theta_params(a, Nf, Q, eps_tot, mode, a_thr):
        eps_theta = eps_tot / (2.0 * Q * Q * torch.clamp(Nf, min=1.0))
        if mode == "direct":
            use_direct = torch.ones_like(a, dtype=torch.bool)
            W  = torch.ceil(torch.sqrt(torch.clamp(((-torch.log(torch.clamp(eps_theta, min=1e-30))) / (PI * torch.clamp(a, min=1e-30))), min=0.0))).to(torch.int64) + 1
            Kp = torch.zeros_like(W)
        elif mode == "poisson":
            Kp = torch.ceil(torch.sqrt(torch.clamp((a / PI) * (-torch.log(torch.clamp(eps_theta, min=1e-30))), min=0.0))).to(torch.int64)
            W = torch.zeros_like(Kp)
            use_direct = torch.zeros_like(Kp, dtype=torch.bool)
        else:
            use_direct = a >= a_thr
            W  = torch.ceil(torch.sqrt(torch.clamp(((-torch.log(torch.clamp(eps_theta, min=1e-30))) / (PI * torch.clamp(a, min=1e-30))), min=0.0))).to(torch.int64) + 1
            Kp = torch.ceil(torch.sqrt(torch.clamp((a / PI) * (-torch.log(torch.clamp(eps_theta, min=1e-30))), min=0.0))).to(torch.int64)
        return use_direct, W, Kp

    use_direct_j, W_j, Kp_j = _theta_params(
        a_j, torch.as_tensor(N, device=device, dtype=r_dtype),
        quad_nodes, torch.as_tensor(eps_total, device=device, dtype=r_dtype),
        theta_mode, torch.as_tensor(a_threshold, device=device, dtype=r_dtype)
    )

    # ---------- unit complex directions b_j ----------
    den = _norm(vec_d_j, dim=-1, keepdim=True).to(r_dtype).clamp_min(torch.as_tensor(1e-20, device=device, dtype=r_dtype))
    b = vec_d_j / den.to(c_dtype)
    bR = b.real.to(r_dtype).contiguous()
    bI = b.imag.to(r_dtype).contiguous()

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

    # ---------- mem targeting ----------
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        target_bytes = int(max(200 * 1024 * 1024, min(1_200 * 1024 * 1024, free_bytes * 0.35)))
    else:
        target_bytes = 256 * 1024 * 1024
    bytes_per_elem = 4 if r_dtype == torch.float32 else 8

    T_out = torch.zeros((B, S_dim), device=device, dtype=c_dtype)

    inv_sqrt_gamma = 1.0 / torch.sqrt(torch.clamp(gamma_j, min=tiny))
    kappa_eff = kappa_j * inv_sqrt_gamma
    norm_fac = (1.0 / PI) * inv_sqrt_gamma  # (M,)

    # ---------- main loops ----------
    for m0 in range(0, M, m_chunk):
        m1 = min(m0 + m_chunk, M)
        mc = m1 - m0

        z_j_c     = z_j[m0:m1]
        a_c       = a_j[m0:m1]
        alpha_c   = alpha_j[m0:m1]
        T_hat_c   = T_hat_j[m0:m1].contiguous()
        use_direct_c = use_direct_j[m0:m1]
        W_c       = W_j[m0:m1]
        Kp_c      = Kp_j[m0:m1]
        bR_c      = bR[m0:m1]
        bI_c      = bI[m0:m1]
        kappa_eff_c = kappa_eff[m0:m1]
        norm_fac_c  = norm_fac[m0:m1]

        # wrapped real displacements per coordinate
        dz = _wrap01((z.unsqueeze(1) - z_j_c.unsqueeze(0)).real.to(r_dtype)).contiguous()  # (B,mc,N)

        # angular factor (B,mc)
        if use_delta:
            vdb = vec_d.unsqueeze(1).expand(B, mc, N)
            vjj = vec_d_j[m0:m1].unsqueeze(0).expand(B, mc, N)
            dd  = _delta_vec_d(vdb, vjj)
        else:
            dd  = (vec_d.unsqueeze(1) - vec_d_j[m0:m1].unsqueeze(0))
        dd_norm2 = (dd.abs() ** 2).sum(dim=-1).to(r_dtype)
        bh_dd = (torch.conj(b[m0:m1]).unsqueeze(0) * dd).sum(dim=-1)
        q_ang   = one_over_sq[m0:m1].unsqueeze(0) * dd_norm2 - c_ang[m0:m1].unsqueeze(0) * (bh_dd.abs() ** 2)
        ang_fac = torch.exp(-PI * q_ang).to(r_dtype)  # (B,mc)

        idx_dir = torch.nonzero(use_direct_c, as_tuple=False).flatten()
        idx_poi = torch.nonzero(~use_direct_c, as_tuple=False).flatten()

        # Online-LSE accumulators over q
        A_lse = torch.full((B, mc), -float("inf"), device=device, dtype=r_dtype)
        S_lse = torch.zeros((B, mc), device=device, dtype=r_dtype)

        Q2 = t1_flat.numel()
        denom_base = max(1, B * mc * max(1, n_chunk))
        qc_max = max(1, min(Q2, int(target_bytes // (bytes_per_elem * denom_base))))

        # Precompute Poisson coefficients base (mc, Kp_max)
        Kp_max = int(Kp_c.max().item()) if idx_poi.numel() > 0 else 0
        if Kp_max > 0:
            k_idx = torch.arange(1, Kp_max + 1, device=device, dtype=r_dtype)
            base_exp_p = torch.exp(-PI * (k_idx.view(1, -1) ** 2) / torch.clamp(a_c.view(-1, 1), min=tiny))  # (mc,Kp_max)

        # Precompute Direct coefficients (mc, W_max)
        W_max = int(W_c.max().item()) if idx_dir.numel() > 0 else 0
        if W_max > 0:
            n_idx = torch.arange(1, W_max + 1, device=device, dtype=r_dtype)
            cjn_all = torch.exp(-PI * torch.clamp(a_c.view(-1,1), min=tiny) * (n_idx.view(1, -1) ** 2))  # (mc,W_max)

        for q0 in range(0, Q2, qc_max):
            q1 = min(q0 + qc_max, Q2)
            qc = q1 - q0
            t1c = t1_flat[q0:q1]
            t2c = t2_flat[q0:q1]
            lwc = logw2_flat[q0:q1]

            log_acc_dir = None  # (B,mc_dir,qc)
            log_acc_poi = None  # (B,mc_poi,qc)

            # ------- inner loop over N with ns-subchunk to cap memory -------
            for n0 in range(0, N, n_chunk):
                n1 = min(n0 + n_chunk, N)
                Nc = n1 - n0

                # estimate ns_sub so that ~5 buffers of (B,mc,ns,qc) fit target_bytes
                per_elem_bufs = 5
                ns_cap = max(1, int(target_bytes // (bytes_per_elem * per_elem_bufs * max(1, B * mc * qc))))
                ns_sub = int(max(1, min(Nc, ns_cap)))

                for nn in range(n0, n1, ns_sub):
                    nn1 = min(n1, nn + ns_sub)
                    ns = nn1 - nn

                    # unified shift & u for subrange nn:nn1
                    shift = (t1c.view(1, 1, -1) * bR_c[:, nn:nn1].unsqueeze(-1) +
                             t2c.view(1, 1, -1) * bI_c[:, nn:nn1].unsqueeze(-1)).contiguous()     # (mc,ns,qc)
                    u_sub = (dz[:, :, nn:nn1].unsqueeze(-1) -
                             (kappa_eff_c.view(1, -1, 1, 1) * shift.unsqueeze(0))).contiguous()   # (B,mc,ns,qc)
                    u_sub = _wrap01(u_sub)

                    # ---------- DIRECT (log-domain cosh) ----------
                    if idx_dir.numel() > 0:
                        mc_d = idx_dir.numel()
                        a_d = a_c[idx_dir].view(1, mc_d, 1, 1)
                        u_d = u_sub[:, idx_dir, :, :]  # (B,mc_d,ns,qc)

                        log_base = (-PI * a_d * (u_d * u_d))
                        log_sum = torch.zeros_like(u_d, dtype=r_dtype)  # log(1)=0

                        if W_max > 0:
                            Wj_d = W_c[idx_dir]
                            n_vals = torch.arange(1, W_max + 1, device=device, dtype=r_dtype)
                            for s in range(0, W_max, 32):  # микрочанк по "offs"
                                e = min(W_max, s + 32)
                                wsub = e - s
                                n_sub = n_vals[s:e]
                                n_mask = (n_sub.view(1, wsub) <= Wj_d.view(-1, 1))  # (mc_d,wsub)

                                beta = (2.0 * PI) * a_d.view(mc_d, 1) * n_sub.view(1, wsub)               # (mc_d,wsub)
                                log_cjn = (-PI * a_d.view(mc_d, 1) * (n_sub.view(1, wsub) ** 2))         # (mc_d,wsub)

                                x = (beta.view(1, mc_d, 1, 1, wsub) * u_d.unsqueeze(-1))                 # (B,mc_d,ns,qc,wsub)
                                ax = torch.abs(x)
                                log_cosh = ax + torch.log1p(torch.exp(-2.0 * ax)) - math.log(2.0)

                                log_term = (math.log(2.0) + log_cjn.view(1, mc_d, 1, 1, wsub) + log_cosh)
                                if not torch.all(n_mask):
                                    nm = n_mask.view(1, mc_d, 1, 1, wsub)
                                    log_term = torch.where(nm, log_term, torch.full_like(log_term, -float("inf")))
                                log_sum = torch.logaddexp(log_sum, torch.logsumexp(log_term, dim=-1))

                        log_theta = log_base + log_sum                     # (B,mc_d,ns,qc)
                        part_d = log_theta.sum(dim=2)                      # (B,mc_d,qc)
                        log_acc_dir = part_d if log_acc_dir is None else (log_acc_dir + part_d)

                    # ---------- POISSON (vectorized Clenshaw) ----------
                    if idx_poi.numel() > 0:
                        mc_p = idx_poi.numel()
                        a_p = a_c[idx_poi].view(1, mc_p, 1, 1)
                        u_p = u_sub[:, idx_poi, :, :]  # (B,mc_p,ns,qc)

                        cosx = torch.cos((2.0 * PI) * u_p)                 # (B,mc_p,ns,qc)
                        if Kp_max == 0:
                            # θ = 1/sqrt(a)  => logθ = -0.5*log(a)
                            log_theta = (-0.5 * torch.log(torch.clamp(a_p, min=tiny))).expand_as(u_p)
                        else:
                            # coeffs (mc_p, Kp_max), masked per j
                            coeff_all = base_exp_p[idx_poi, :Kp_max]       # (mc_p, Kp_max)
                            Kp_p = Kp_c[idx_poi]                           # (mc_p,)

                            b1 = torch.zeros_like(u_p, dtype=r_dtype)
                            b2 = torch.zeros_like(u_p, dtype=r_dtype)
                            # k from Kp_max .. 1
                            for k in range(Kp_max, 0, -1):
                                ck = coeff_all[:, k-1]                     # (mc_p,)
                                # mask k > Kp_j -> ck=0
                                ck = ck * (k <= Kp_p).to(r_dtype)
                                ck = ck.view(1, mc_p, 1, 1)                # (1,mc_p,1,1)
                                b0 = 2.0 * cosx * b1 - b2 + ck             # (B,mc_p,ns,qc)
                                b2 = b1
                                b1 = b0
                            S = b1 * cosx - b2                              # Σ_{k=1..Kp_j} c_k cos(kx)
                            sum_expr = 1.0 + 2.0 * S
                            log_theta = (-0.5 * torch.log(torch.clamp(a_p, min=tiny))) + torch.log(torch.clamp(sum_expr, min=tiny))

                        part_p = log_theta.sum(dim=2)                      # (B,mc_p,qc)
                        log_acc_poi = part_p if log_acc_poi is None else (log_acc_poi + part_p)

            # ---------- close LSE over q-chunk ----------
            if idx_dir.numel() > 0:
                L_sub = log_acc_dir + lwc.view(1, 1, -1)
                Ad = A_lse[:, idx_dir]; Sd = S_lse[:, idx_dir]
                Lmax = L_sub.max(dim=-1).values
                Mx = torch.maximum(Ad, Lmax)
                Sd = Sd * torch.exp(Ad - Mx) + torch.exp(L_sub - Mx.unsqueeze(-1)).sum(dim=-1)
                A_lse[:, idx_dir] = Mx; S_lse[:, idx_dir] = Sd

            if idx_poi.numel() > 0:
                L_sub = log_acc_poi + lwc.view(1, 1, -1)
                Ap = A_lse[:, idx_poi]; Sp = S_lse[:, idx_poi]
                Lmax = L_sub.max(dim=-1).values
                Mx = torch.maximum(Ap, Lmax)
                Sp = Sp * torch.exp(Ap - Mx) + torch.exp(L_sub - Mx.unsqueeze(-1)).sum(dim=-1)
                A_lse[:, idx_poi] = Mx; S_lse[:, idx_poi] = Sp

        # ---------- η and final combine ----------
        eta = (norm_fac_c.view(1, mc) * torch.exp(A_lse) * S_lse).reshape(B, mc)   # (B,mc)
        weight = (alpha_c.reshape(1, mc) * ang_fac.reshape(B, mc) * eta).to(c_dtype)  # (B,mc)
        T_out = T_out + weight @ T_hat_c.reshape(mc, S_dim)  # GEMM

    return T_out
