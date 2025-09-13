import math
import torch

@torch.no_grad()
def T_HS_theta(
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
    device = z.device
    c_dtype = z.dtype if dtype_override is None else dtype_override
    r_dtype = torch.float32 if c_dtype == torch.complex64 else torch.float64
    tiny = torch.as_tensor(torch.finfo(r_dtype).tiny, device=device, dtype=r_dtype)

    # ---------- helpers ----------
    def _wrap01(x):  # [-1/2, 1/2)
        return torch.remainder(x + 0.5, 1.0) - 0.5

    def _norm(x, dim=-1, keepdim=False):
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)

    def _hermite_gauss(K: int, device, dtype):
        if K < 1:
            raise ValueError("quad_nodes must be >=1")
        k = torch.arange(1, K, device=device, dtype=dtype)
        off = torch.sqrt(k / 2.0)
        J = torch.zeros((K, K), device=device, dtype=dtype)
        J.diagonal(1).copy_(off)
        J.diagonal(-1).copy_(off)
        evals, evecs = torch.linalg.eigh(J)
        x = evals
        w = (evecs[0, :].abs() ** 2) * math.sqrt(math.pi)  # ∫ e^{-τ^2} f(τ) dτ
        return x, w

    def _theta_params(a, N_tot, Q, eps_tot, mode, a_thr):
        # бюджет на θ: eps_theta = eps_tot / (2 * N * Q^2)
        eps_theta = torch.as_tensor(eps_tot, dtype=r_dtype, device=device) / (2.0 * N_tot * (Q * Q))
        if mode == "direct":
            W = torch.ceil(torch.sqrt(torch.clamp(((-torch.log(torch.clamp(eps_theta, min=1e-30))) /
                                                  (math.pi * torch.clamp(a, min=1e-30))), min=0.0))).to(torch.int64) + 1
            Kp = torch.zeros_like(W)
            use_direct = torch.ones_like(W, dtype=torch.bool)
        elif mode == "poisson":
            Kp = torch.ceil(torch.sqrt(torch.clamp((a / math.pi) * (-torch.log(torch.clamp(eps_theta, min=1e-30))),
                                                   min=0.0))).to(torch.int64)
            W = torch.zeros_like(Kp)
            use_direct = torch.zeros_like(Kp, dtype=torch.bool)
        else:
            use_direct = a >= a_thr
            W  = torch.ceil(torch.sqrt(torch.clamp(((-torch.log(torch.clamp(eps_theta, min=1e-30))) /
                                                    (math.pi * torch.clamp(a, min=1e-30))), min=0.0))).to(torch.int64) + 1
            Kp = torch.ceil(torch.sqrt(torch.clamp((a / math.pi) * (-torch.log(torch.clamp(eps_theta, min=1e-30))),
                                                   min=0.0))).to(torch.int64)
        return use_direct, W, Kp

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

    # ---------- anisotropy ----------
    a_j = 1.0 / torch.clamp(sigma_perp, min=tiny)  # a = 1/σ⊥
    gamma_j = torch.clamp(sigma_perp / torch.clamp(sigma_par, min=tiny), min=tiny, max=1.0)
    # κ^2 = ((σ∥−σ⊥) σ⊥)/(π σ∥)
    kappa_j = torch.sqrt(torch.clamp((sigma_par - sigma_perp) * torch.clamp(sigma_perp, min=tiny)
                                     / (math.pi * torch.clamp(sigma_par, min=tiny)), min=0.0))

    use_direct_j, W_j, Kp_j = _theta_params(
        a_j, torch.as_tensor(N, device=device, dtype=r_dtype),
        quad_nodes, torch.as_tensor(eps_total, device=device, dtype=r_dtype),
        theta_mode, torch.as_tensor(a_threshold, device=device, dtype=r_dtype)
    )

    # unit complex directions b_j
    den = _norm(vec_d_j, dim=-1, keepdim=True).to(r_dtype).clamp_min(torch.as_tensor(1e-20, device=device, dtype=r_dtype))
    b = vec_d_j / den.to(c_dtype)
    bR = b.real.to(r_dtype)
    bI = b.imag.to(r_dtype)

    # angular factor ingredients (как в Tau_dual)
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

    # GH nodes flattened (Q2)
    tau, w = _hermite_gauss(quad_nodes, device, r_dtype)
    t1 = tau.view(-1, 1).expand(-1, quad_nodes)
    t2 = tau.view(1, -1).expand(quad_nodes, -1)
    t1_flat = t1.reshape(-1)  # (Q2,)
    t2_flat = t2.reshape(-1)  # (Q2,)
    logw2_flat = (torch.log(torch.clamp(w, min=tiny)).view(-1, 1) +
                  torch.log(torch.clamp(w, min=tiny)).view(1, -1)).reshape(-1)  # (Q2,)
    Q2 = t1_flat.numel()

    # adaptive chunk sizes by free memory
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        target_bytes = int(max(200 * 1024 * 1024, min(1_200 * 1024 * 1024, free_bytes * 0.35)))
    else:
        target_bytes = 200 * 1024 * 1024
    bytes_per_elem = 4 if r_dtype == torch.float32 else 8

    T_out = torch.zeros((B, S_dim), device=device, dtype=c_dtype)

    for m0 in range(0, M, m_chunk):
        m1 = min(m0 + m_chunk, M)
        mc = m1 - m0

        z_j_c     = z_j[m0:m1]
        vec_d_j_c = vec_d_j[m0:m1]
        a_c       = a_j[m0:m1]
        gamma_c   = gamma_j[m0:m1]
        kappa_c   = kappa_j[m0:m1]
        alpha_c   = alpha_j[m0:m1]
        T_hat_c   = T_hat_j[m0:m1]
        use_direct_c = use_direct_j[m0:m1]
        W_c       = W_j[m0:m1]
        Kp_c      = Kp_j[m0:m1]
        bR_c      = bR[m0:m1].contiguous()
        bI_c      = bI[m0:m1].contiguous()

        dz = _wrap01((z.unsqueeze(1) - z_j_c.unsqueeze(0)).real.to(r_dtype)).contiguous()  # (B,mc,N)

        # ang_fac (B,mc)
        if use_delta:
            vdb = vec_d.unsqueeze(1).expand(B, mc, N)
            vjj = vec_d_j_c.unsqueeze(0).expand(B, mc, N)
            dd  = _delta_vec_d(vdb, vjj)
        else:
            dd  = (vec_d.unsqueeze(1) - vec_d_j_c.unsqueeze(0))
        dd_norm2 = (dd.abs() ** 2).sum(dim=-1).to(r_dtype)
        bh_dd = (torch.conj(b[m0:m1]).unsqueeze(0) * dd).sum(dim=-1)
        q_ang   = one_over_sq[m0:m1].unsqueeze(0) * dd_norm2 - c_ang[m0:m1].unsqueeze(0) * (bh_dd.abs() ** 2)
        ang_fac = torch.exp(-math.pi * q_ang).to(r_dtype)  # (B,mc)

        inv_sqrt_gamma = 1.0 / torch.sqrt(torch.clamp(gamma_c, min=tiny))
        kappa_eff = kappa_c * inv_sqrt_gamma
        norm_fac = (1.0 / math.pi) * inv_sqrt_gamma.view(1, mc)  # (1,mc)

        # индексы вкладов по режимам θ
        idx_dir = torch.nonzero(use_direct_c, as_tuple=False).flatten()
        idx_poi = torch.nonzero(~use_direct_c, as_tuple=False).flatten()

        # online-LSE по Q²: sum_q e^{L_q} = e^{A_lse} * S_lse
        A_lse = torch.full((B, mc), -float("inf"), device=device, dtype=r_dtype)
        S_lse = torch.zeros((B, mc), device=device, dtype=r_dtype)

        # выбрать q_chunk по памяти: держим u:(B,mc,Nc,qc)
        denom = max(1, B * mc * max(1, n_chunk))
        qc_max = max(1, min(Q2, int(target_bytes // (bytes_per_elem * denom))))

        for q0 in range(0, Q2, qc_max):
            q1 = min(q0 + qc_max, Q2)
            t1c = t1_flat[q0:q1]  # (qc,)
            t2c = t2_flat[q0:q1]  # (qc,)
            lwc = logw2_flat[q0:q1]  # (qc,)

            # аккумулируем лог-произведение θ по координатам для каждого узла qc
            # формы: (B, mc_sub, qc)
            log_acc_dir = None
            log_acc_poi = None

            for n0 in range(0, N, n_chunk):
                n1 = min(n0 + n_chunk, N)
                Nc = n1 - n0

                # ------ DIRECT (use_direct=True) ------
                if idx_dir.numel() > 0:
                    mc_d = idx_dir.numel()
                    bR_d = bR_c[idx_dir, n0:n1]  # (mc_d,Nc)
                    bI_d = bI_c[idx_dir, n0:n1]
                    shift_d = (t1c.view(1, 1, -1) * bR_d.unsqueeze(-1) +
                               t2c.view(1, 1, -1) * bI_d.unsqueeze(-1))           # (mc_d,Nc,qc)
                    u_d = dz[:, idx_dir, n0:n1].unsqueeze(-1) - kappa_eff[idx_dir].view(1, mc_d, 1, 1) * shift_d.unsqueeze(0)  # (B,mc_d,Nc,qc)
                    a_d = a_c[idx_dir].view(1, mc_d, 1, 1)

                    Wmax = int(W_c[idx_dir].max().item())
                    if Wmax == 0:
                        theta_d = torch.exp(-math.pi * a_d * (u_d * u_d))  # (B,mc_d,Nc,qc)
                    else:
                        theta_d = torch.zeros((B, mc_d, Nc, t1c.numel()), device=device, dtype=r_dtype)  # (B,mc_d,Nc,qc)
                        off_denom = max(1, B * mc_d * Nc * max(1, (q1 - q0)))
                        off_bytes = max(32 * 1024 * 1024, target_bytes // 4)
                        off_chunk = max(1, int(off_bytes // (bytes_per_elem * off_denom)))
                        off_chunk = min(off_chunk, 2 * Wmax + 1)

                        start = -Wmax
                        while start <= Wmax:
                            end = min(Wmax, start + off_chunk - 1)
                            o = end - start + 1
                            offs = torch.arange(start, end + 1, device=device, dtype=r_dtype).view(1, 1, 1, 1, o)  # (1,1,1,1,o)
                            y = u_d.unsqueeze(-1) - offs                                              # (B,mc_d,Nc,qc,o)
                            val = torch.exp(-math.pi * a_d[..., None] * (y * y))                      # (B,mc_d,Nc,qc,o)
                            idxo = torch.arange(start, end + 1, device=device, dtype=torch.int64).abs().view(1, 1, 1, 1, o)
                            mask = (idxo <= W_c[idx_dir].view(1, -1, 1, 1, 1))
                            val = torch.where(mask, val, torch.zeros_like(val))
                            theta_d = theta_d + val.sum(dim=-1)                                      # (B,mc_d,Nc,qc)
                            start = end + 1

                    part_d = torch.clamp(theta_d, min=tiny).log().sum(dim=2)  # (B,mc_d,qc)
                    log_acc_dir = part_d if log_acc_dir is None else (log_acc_dir + part_d)

                # ------ POISSON (use_direct=False) ------
                if idx_poi.numel() > 0:
                    mc_p = idx_poi.numel()
                    bR_p = bR_c[idx_poi, n0:n1]
                    bI_p = bI_c[idx_poi, n0:n1]
                    shift_p = (t1c.view(1, 1, -1) * bR_p.unsqueeze(-1) +
                               t2c.view(1, 1, -1) * bI_p.unsqueeze(-1))           # (mc_p,Nc,qc)
                    u_p = dz[:, idx_poi, n0:n1].unsqueeze(-1) - kappa_eff[idx_poi].view(1, mc_p, 1, 1) * shift_p.unsqueeze(0)  # (B,mc_p,Nc,qc)
                    a_p = a_c[idx_poi].view(1, mc_p, 1, 1)

                    inv_sqrt_a = torch.rsqrt(a_p)
                    theta_p = inv_sqrt_a.expand(B, mc_p, Nc, t1c.numel())         # (B,mc_p,Nc,qc)  (k=0)

                    Kpmax = int(Kp_c[idx_poi].max().item())
                    if Kpmax > 0:
                        k_denom = max(1, B * mc_p * Nc * max(1, (q1 - q0)))
                        k_bytes = max(32 * 1024 * 1024, target_bytes // 4)
                        k_chunk = max(1, int(k_bytes // (bytes_per_elem * k_denom)))
                        k_chunk = min(k_chunk, Kpmax)

                        k_start = 1
                        while k_start <= Kpmax:
                            k_end = min(Kpmax, k_start + k_chunk - 1)
                            ksz = k_end - k_start + 1
                            kval = torch.arange(k_start, k_end + 1, device=device, dtype=r_dtype).view(1, 1, 1, 1, ksz)
                            coeff = torch.exp(-math.pi * (kval * kval) / a_p[..., None])               # (1,mc_p,1,1,k)
                            phi   = 2.0 * math.pi * u_p.unsqueeze(-1) * kval                           # (B,mc_p,Nc,qc,k)
                            term  = 2.0 * inv_sqrt_a[..., None] * coeff * torch.cos(phi)               # (B,mc_p,Nc,qc,k)
                            idxk = torch.arange(k_start, k_end + 1, device=device, dtype=torch.int64).view(1, 1, 1, 1, ksz)
                            maskk = (idxk <= Kp_c[idx_poi].view(1, -1, 1, 1, 1))
                            term = torch.where(maskk, term, torch.zeros_like(term))
                            theta_p = theta_p + term.sum(dim=-1)                                       # (B,mc_p,Nc,qc)
                            k_start = k_end + 1

                    part_p = torch.clamp(theta_p, min=tiny).log().sum(dim=2)  # (B,mc_p,qc)
                    log_acc_poi = part_p if log_acc_poi is None else (log_acc_poi + part_p)

            # LSE-слияние по узлам qc в аккумуляторы A_lse,S_lse
            if idx_dir.numel() > 0:
                L_sub = log_acc_dir + lwc.view(1, 1, -1)                 # (B,mc_d,qc)
                Ad = A_lse[:, idx_dir]; Sd = S_lse[:, idx_dir]
                Lmax = L_sub.max(dim=-1).values                          # (B,mc_d)
                Mx = torch.maximum(Ad, Lmax)
                Sd = Sd * torch.exp(Ad - Mx) + torch.exp(L_sub - Mx.unsqueeze(-1)).sum(dim=-1)
                A_lse[:, idx_dir] = Mx; S_lse[:, idx_dir] = Sd

            if idx_poi.numel() > 0:
                L_sub = log_acc_poi + lwc.view(1, 1, -1)                 # (B,mc_p,qc)
                Ap = A_lse[:, idx_poi]; Sp = S_lse[:, idx_poi]
                Lmax = L_sub.max(dim=-1).values
                Mx = torch.maximum(Ap, Lmax)
                Sp = Sp * torch.exp(Ap - Mx) + torch.exp(L_sub - Mx.unsqueeze(-1)).sum(dim=-1)
                A_lse[:, idx_poi] = Mx; S_lse[:, idx_poi] = Sp

        # === η и итог (shape-safe) ===
        Bc, mc_check = A_lse.shape
        assert mc_check == mc and S_lse.shape == (Bc, mc), f"bad A/S shapes: {A_lse.shape}, {S_lse.shape}, mc={mc}"
        norm_fac = norm_fac.reshape(1, mc).contiguous()                 # (1,mc)
        eta = (norm_fac * torch.exp(A_lse) * S_lse).reshape(Bc, mc)     # (B,mc)
        assert eta.shape == (Bc, mc)

        alpha_row = alpha_c.reshape(1, mc).contiguous()                 # (1,mc)
        ang_fac = ang_fac.reshape(Bc, mc).contiguous()                  # (B,mc)
        weight = (alpha_row * ang_fac * eta).to(c_dtype)                # (B,mc)
        assert weight.shape == (Bc, mc)

        T_hat_c = T_hat_c.reshape(mc, S_dim).contiguous()               # (mc,S_dim)
        T_out = T_out + (weight.unsqueeze(-1) * T_hat_c.view(1, mc, S_dim)).sum(dim=1)

    return T_out
