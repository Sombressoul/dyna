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
    theta_mode: str = "poisson",   # игнорируется, всегда Poisson
    eps_total: float = 1.0e-3,
    a_threshold: float = 1.0,      # игнорируется
    n_chunk: int = 64,
    m_chunk: int = 65536,
    dtype_override: torch.dtype | None = None,
):
    """
    HS-Theta (Poisson-only):
      - 2D Gauss–Hermite квадратура (Q×Q), кэш узлов/весов
      - Θ(u; a) ≈ a^{-1/2} * (1 + 2 Σ_{k=1..Kp} exp(-π k^2 / a) cos(2π k u))
        вычисляется векторизованным Clenshaw по k (без материализации оси k)
      - Online log-sum-exp по q (узлам GH)
      - Финал: 2×SGEMM (real/imag) вместо CGEMM; TF32 на этих GEMM выключен для точности
    """
    device = z.device
    c_dtype = z.dtype if dtype_override is None else dtype_override
    r_dtype = torch.float32 if c_dtype == torch.complex64 else torch.float64
    PI = math.pi
    TWO_PI = 2.0 * math.pi
    tiny = torch.as_tensor(torch.finfo(r_dtype).tiny, device=device, dtype=r_dtype)

    def _wrap01(x):
        # map to [-1/2, 1/2)
        return torch.remainder(x + 0.5, 1.0) - 0.5

    def _norm(x, dim=-1, keepdim=False):
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)

    # ---------- Cached GH (2D grid flattened) ----------
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
        t1_flat = t1.reshape(-1)                        # (Q^2,)
        t2_flat = t2.reshape(-1)
        logw2_flat = (torch.log(torch.clamp(w, min=tiny)).view(-1, 1) +
                      torch.log(torch.clamp(w, min=tiny)).view(1, -1)).reshape(-1)
        T_HS_Theta._gh_cache[gh_key] = (t1_flat, t2_flat, logw2_flat)
        t1_flat, t2_flat, logw2_flat = T_HS_Theta._gh_cache[gh_key]
    else:
        t1_flat, t2_flat, logw2_flat = cached

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

    # ---------- anisotropy & θ params (Poisson only) ----------
    a_j = 1.0 / torch.clamp(sigma_perp, min=tiny)                           # (M,)
    gamma_j = torch.clamp(sigma_perp / torch.clamp(sigma_par, min=tiny), min=tiny, max=1.0)
    kappa_j = torch.sqrt(torch.clamp((sigma_par - sigma_perp) * torch.clamp(sigma_perp, min=tiny)
                                     / (PI * torch.clamp(sigma_par, min=tiny)), min=0.0))
    inv_sqrt_gamma = 1.0 / torch.sqrt(torch.clamp(gamma_j, min=tiny))
    kappa_eff = kappa_j * inv_sqrt_gamma                                   # (M,)
    norm_fac = (1.0 / PI) * inv_sqrt_gamma                                 # (M,)

    # требуемая точность на θ → усечение Kp_j
    Q = quad_nodes
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

    # ---------- mem targeting ----------
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        # Poisson-only: держим 0.35 от свободной и <1GB, чтобы увеличить ns_sub
        target_bytes = int(max(256 * 1024 * 1024, min(1_000 * 1024 * 1024, int(free_bytes * 0.35))))
    else:
        target_bytes = 256 * 1024 * 1024
    bytes_per_elem = 4 if r_dtype == torch.float32 else 8

    # ---------- output (2×SGEMM; TF32 OFF локально) ----------
    T_out = torch.zeros((B, S_dim), device=device, dtype=c_dtype)

    # ---------- main loops ----------
    for m0 in range(0, M, m_chunk):
        m1 = min(m0 + m_chunk, M)
        mc = m1 - m0

        z_j_c     = z_j[m0:m1]
        a_c       = a_j[m0:m1]                              # (mc,)
        alpha_c   = alpha_j[m0:m1]
        T_hat_c   = T_hat_j[m0:m1].contiguous()
        Kp_c      = Kp_j[m0:m1]                              # (mc,)
        kappa_eff_c = kappa_eff[m0:m1]                       # (mc,)
        norm_fac_c  = norm_fac[m0:m1]                        # (mc,)

        # направления
        den = _norm(vec_d_j[m0:m1], dim=-1, keepdim=True).to(r_dtype).clamp_min(torch.as_tensor(1e-20, device=device, dtype=r_dtype))
        b = vec_d_j[m0:m1] / den.to(c_dtype)                # (mc,Ncplx)
        bR_c = b.real.to(r_dtype).contiguous()              # (mc,N)
        bI_c = b.imag.to(r_dtype).contiguous()

        # κ вшиваем в направления (меньше FLOPs дальше)
        bR_eff_c = (kappa_eff_c.view(-1, 1) * bR_c)         # (mc,N)
        bI_eff_c = (kappa_eff_c.view(-1, 1) * bI_c)

        # displacements per coord (wrap)
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

        # LSE по q
        A_lse = torch.full((B, mc), -float("inf"), device=device, dtype=r_dtype)
        S_lse = torch.zeros((B, mc), device=device, dtype=r_dtype)

        # подготовка коэффициентов Пуассона
        Kp_max = int(Kp_c.max().item())
        if Kp_max > 0:
            k_idx = torch.arange(1, Kp_max + 1, device=device, dtype=r_dtype)                        # (Kp_max,)
            base_exp_p = torch.exp(-PI * (k_idx.view(1, -1) ** 2) / torch.clamp(a_c.view(-1, 1), min=tiny))  # (mc,Kp_max)
            log_inv_sqrt_a = -0.5 * torch.log(torch.clamp(a_c, min=tiny))                            # (mc,)
        else:
            log_inv_sqrt_a = -0.5 * torch.log(torch.clamp(a_c, min=tiny))                            # (mc,)

        # выберем q-чанк так, чтобы чаще брать весь Q^2
        Q2 = t1_flat.numel()
        denom_base = max(1, B * mc * max(1, n_chunk))
        qc_max = min(Q2, max(1, int(target_bytes // (bytes_per_elem * denom_base))))
        if qc_max < 1:
            qc_max = 1

        for q0 in range(0, Q2, qc_max):
            q1 = min(q0 + qc_max, Q2)
            qc = q1 - q0
            t1c = t1_flat[q0:q1]
            t2c = t2_flat[q0:q1]
            lwc = logw2_flat[q0:q1]

            log_acc = None  # (B,mc,qc), аккум лог-вкладов по координате

            for n0 in range(0, N, n_chunk):
                n1 = min(n0 + n_chunk, N)
                Nc = n1 - n0

                # держим ~2 буфера (B,mc,ns,qc)
                per_elem_bufs = 2
                ns_cap = max(1, int(target_bytes // (bytes_per_elem * per_elem_bufs * max(1, B * mc * qc))))
                ns_sub = int(max(1, min(Nc, ns_cap)))

                for nn in range(n0, n1, ns_sub):
                    nn1 = min(n1, nn + ns_sub)
                    ns = nn1 - nn

                    # u = wrap(dz - κ * (t1*bR + t2*bI))
                    shift = (t1c.view(1, 1, -1) * bR_eff_c[:, nn:nn1].unsqueeze(-1) +
                             t2c.view(1, 1, -1) * bI_eff_c[:, nn:nn1].unsqueeze(-1)).contiguous()     # (mc,ns,qc)
                    u_sub = _wrap01(dz[:, :, nn:nn1].unsqueeze(-1) - shift.unsqueeze(0))             # (B,mc,ns,qc)

                    # Poisson θ via Clenshaw (векторизовано по B,mc,ns,qc)
                    if Kp_max == 0:
                        # sum_{ns} logθ = ns * (-½ log a)   (нет cos-серии)
                        part = (ns * log_inv_sqrt_a.view(1, mc, 1)).expand(B, mc, qc)                # (B,mc,qc)
                    else:
                        cosx = torch.cos(TWO_PI * u_sub).contiguous()                                # (B,mc,ns,qc)
                        coeff_all = base_exp_p[:, :Kp_max]                                          # (mc,Kp_max)

                        b1 = torch.zeros_like(u_sub, dtype=r_dtype)
                        b2 = torch.zeros_like(b1)
                        # глобальный проход по k=Kp_max..1 с маской (k<=Kp_j)
                        for k in range(Kp_max, 0, -1):
                            ck = (coeff_all[:, k-1] * (k <= Kp_c).to(r_dtype)).view(1, mc, 1, 1)     # (1,mc,1,1)
                            # b0 = 2*cosx*b1 - b2 + ck   (минимум времянок)
                            b0 = torch.addcmul(-b2, cosx, b1, value=2.0)
                            b0 += ck
                            b2, b1 = b1, b0
                        S = b1 * cosx - b2                                                          # (B,mc,ns,qc)
                        # sum_expr = 1 + 2S; делаем log in-place по ns
                        sum_expr = S.mul_(2.0).add_(1.0)                                            # in-place: 2S+1
                        sum_expr.clamp_(min=tiny).log_()                                            # log(1+2S)
                        # sum_{ns} logθ = ns*(-½log a) + Σ_ns log(1+2S)
                        part = sum_expr.sum(dim=2) + (ns * log_inv_sqrt_a.view(1, mc, 1))           # (B,mc,qc)

                    log_acc = part if log_acc is None else (log_acc + part)

            # закрываем LSE по q-чанку
            L_sub = log_acc + lwc.view(1, 1, -1)          # (B,mc,qc)
            A_cur = L_sub.max(dim=-1).values              # (B,mc)
            Mx = torch.maximum(A_lse, A_cur)
            S_lse = S_lse * torch.exp(A_lse - Mx) + torch.exp(L_sub - Mx.unsqueeze(-1)).sum(dim=-1)
            A_lse = Mx

        # η и финальный комбайн (2×SGEMM; TF32 OFF локально)
        eta = (norm_fac_c.view(1, mc) * torch.exp(A_lse) * S_lse).reshape(B, mc)     # (B,mc)
        weight_f32 = (alpha_c.reshape(1, mc) * ang_fac.reshape(B, mc) * eta).to(torch.float32)

        R = T_hat_c.real.contiguous().to(torch.float32)                               # (mc,S)
        I = T_hat_c.imag.contiguous().to(torch.float32)

        if device.type == "cuda":
            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                T_out.real.addmm_(weight_f32, R)
                T_out.imag.addmm_(weight_f32, I)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        else:
            T_out.real.addmm_(weight_f32, R)
            T_out.imag.addmm_(weight_f32, I)

    return T_out
