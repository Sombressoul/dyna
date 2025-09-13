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
    def _wrap01(x: torch.Tensor) -> torch.Tensor:
        # полуинтервал [-1/2, 1/2)
        return torch.remainder(x + 0.5, 1.0) - 0.5

    def _norm(x, dim=-1, keepdim=False):
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)

    def _hermite_gauss(K: int, device, dtype):
        if K < 1:
            raise ValueError("quad_nodes must be >= 1")
        k = torch.arange(1, K, device=device, dtype=dtype)
        off = torch.sqrt(k / 2.0)
        J = torch.zeros((K, K), device=device, dtype=dtype)
        J.diagonal(1).copy_(off)
        J.diagonal(-1).copy_(off)
        evals, evecs = torch.linalg.eigh(J)
        x = evals                              # узлы τ_k
        w = (evecs[0, :].abs() ** 2) * math.sqrt(math.pi)  # веса для ∫ e^{-τ^2} f(τ)dτ
        return x, w

    # θ_a(u): две формы, выбор/параметры по бюджету ошибки
    def _theta_direct(u: torch.Tensor, a: torch.Tensor, W: int) -> torch.Tensor:
        # u: (..., Nchunk), a: (1, mc, 1)
        out = torch.zeros_like(u, dtype=u.dtype)
        # суммируем симметрично для стабильности
        out = out + torch.exp(-math.pi * a * (u * u))
        for n in range(1, W + 1):
            y1 = u + float(n)
            y2 = u - float(n)
            out = out + torch.exp(-math.pi * a * (y1 * y1))
            out = out + torch.exp(-math.pi * a * (y2 * y2))
        return out

    def _theta_poisson(u: torch.Tensor, a: torch.Tensor, Kp: int) -> torch.Tensor:
        # θ_a(u) = 1/√a + 2/√a Σ_{k=1..} e^{-π k^2 / a} cos(2π k u)
        inv_sqrt_a = torch.rsqrt(a)
        base = inv_sqrt_a.clone()  # k=0
        two = torch.as_tensor(2.0, device=u.device, dtype=u.dtype)
        pi = math.pi
        for k in range(1, Kp + 1):
            coeff = torch.exp(-pi * (k * k) / a)
            base = base + two * inv_sqrt_a * coeff * torch.cos(two * pi * k * u)
        return base

    def _theta_params(a, N_tot, Kquad, eps_tot, mode, a_thr):
        # распределяем бюджет ошибки: eps_theta = eps_tot / (2 * N * Kquad^2)
        # (2 — запас на две оси τ; Kquad^2 — число узлов на 2D GH)
        eps_theta = torch.as_tensor(eps_tot, dtype=r_dtype, device=device) / (
            2.0 * N_tot * (Kquad * Kquad)
        )
        # простые безопасные оценки W/Kp
        if mode == "direct":
            W = torch.ceil(
                torch.sqrt(
                    torch.clamp(
                        (-torch.log(torch.clamp(eps_theta, min=1e-30)))
                        / (math.pi * torch.clamp(a, min=1e-30)),
                        min=0.0,
                    )
                )
            ).to(torch.int64) + 1
            Kp = torch.zeros_like(W)
            use_direct = torch.ones_like(W, dtype=torch.bool)
        elif mode == "poisson":
            Kp = torch.ceil(
                torch.sqrt(
                    torch.clamp(
                        (a / math.pi) * (-torch.log(torch.clamp(eps_theta, min=1e-30))),
                        min=0.0,
                    )
                )
            ).to(torch.int64)
            W = torch.zeros_like(Kp)
            use_direct = torch.zeros_like(Kp, dtype=torch.bool)
        else:
            # auto: direct при "толстых" θ (a ≥ a_thr), иначе Poisson
            use_direct = a >= a_thr
            W = torch.ceil(
                torch.sqrt(
                    torch.clamp(
                        (-torch.log(torch.clamp(eps_theta, min=1e-30)))
                        / (math.pi * torch.clamp(a, min=1e-30)),
                        min=0.0,
                    )
                )
            ).to(torch.int64) + 1
            Kp = torch.ceil(
                torch.sqrt(
                    torch.clamp(
                        (a / math.pi) * (-torch.log(torch.clamp(eps_theta, min=1e-30))),
                        min=0.0,
                    )
                )
            ).to(torch.int64)
        return use_direct, W, Kp

    # ---------- shapes & casts ----------
    B, N = z.shape[-2], z.shape[-1]
    M = z_j.shape[-2]
    S = T_hat_j.shape[-1]

    z         = z.to(c_dtype)
    z_j       = z_j.to(c_dtype)
    vec_d     = vec_d.to(c_dtype)
    vec_d_j   = vec_d_j.to(c_dtype)
    T_hat_j   = T_hat_j.to(c_dtype)
    alpha_j   = alpha_j.to(r_dtype)
    sigma_par = sigma_par.to(r_dtype)
    sigma_perp= sigma_perp.to(r_dtype)

    # ---------- параметры анизотропии ----------
    # a_j = 1/σ⊥, γ = σ⊥/σ∥ ∈ (0,1], κ^2 = ((σ∥−σ⊥) σ⊥)/(π σ∥)
    a_j = 1.0 / torch.clamp(sigma_perp, min=tiny)
    gamma_j = torch.clamp(sigma_perp / torch.clamp(sigma_par, min=tiny), min=tiny, max=1.0)
    kappa_j = torch.sqrt(
        torch.clamp(
            (sigma_par - sigma_perp)
            * torch.clamp(sigma_perp, min=tiny)
            / (math.pi * torch.clamp(sigma_par, min=tiny)),
            min=0.0,
        )
    )
    # выбор форм θ и усечений
    use_direct_j, W_j, Kp_j = _theta_params(
        a_j,
        torch.as_tensor(N, device=device, dtype=r_dtype),
        torch.as_tensor(quad_nodes, device=device, dtype=r_dtype),
        torch.as_tensor(eps_total, device=device, dtype=r_dtype),
        theta_mode,
        torch.as_tensor(a_threshold, device=device, dtype=r_dtype),
    )

    # единичные комплексные направления b_j
    den = _norm(vec_d_j, dim=-1, keepdim=True).to(r_dtype).clamp_min(torch.as_tensor(1e-20, device=device, dtype=r_dtype))
    b = vec_d_j / den.to(c_dtype)          # (M,N) complex
    bR = b.real.to(r_dtype)                # (M,N)
    bI = b.imag.to(r_dtype)                # (M,N)

    # угловая часть (ровно как в Tau_dual)
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

    tau, w = _hermite_gauss(quad_nodes, device, r_dtype)
    logw = torch.log(torch.clamp(w, min=tiny))

    T_out = torch.zeros((B, S), device=device, dtype=c_dtype)

    # ---------- главный цикл по M (чанкуем вклады) ----------
    for m0 in range(0, M, m_chunk):
        m1 = min(m0 + m_chunk, M)
        mc = m1 - m0

        z_j_c     = z_j[m0:m1]
        vec_d_j_c = vec_d_j[m0:m1]
        a_c       = a_j[m0:m1]           # (mc,)
        gamma_c   = gamma_j[m0:m1]       # (mc,)
        kappa_c   = kappa_j[m0:m1]       # (mc,)
        alpha_c   = alpha_j[m0:m1]       # (mc,)
        sp_c      = sigma_par[m0:m1]
        sq_c      = sigma_perp[m0:m1]
        T_hat_c   = T_hat_j[m0:m1]
        use_direct_c = use_direct_j[m0:m1]
        W_c       = W_j[m0:m1]
        Kp_c      = Kp_j[m0:m1]
        bR_c      = bR[m0:m1]            # (mc,N)
        bI_c      = bI[m0:m1]            # (mc,N)

        # Δz на торе
        dz = _wrap01((z.unsqueeze(1) - z_j_c.unsqueeze(0)).real.to(r_dtype))  # (B,mc,N)

        # ang (совпадает с Tau_dual)
        if use_delta:
            vdb = vec_d.unsqueeze(1).expand(B, mc, N)
            vjj = vec_d_j_c.unsqueeze(0).expand(B, mc, N)
            dd = _delta_vec_d(vdb, vjj)
        else:
            dd = (vec_d.unsqueeze(1) - vec_d_j_c.unsqueeze(0))
        dd_norm2 = (dd.abs() ** 2).sum(dim=-1).to(r_dtype)          # (B,mc)
        bh_dd = (torch.conj(b[m0:m1]).unsqueeze(0) * dd).sum(dim=-1)  # (B,mc)
        q_ang = one_over_sq[m0:m1].unsqueeze(0) * dd_norm2 - c_ang[m0:m1].unsqueeze(0) * (bh_dd.abs() ** 2)
        ang   = torch.exp(-math.pi * q_ang).to(r_dtype)              # (B,mc)

        # нормировка HS-интеграла: переход t = τ / √γ  => множитель 1/√γ и сдвиг κ/√γ
        inv_sqrt_gamma = 1.0 / torch.sqrt(torch.clamp(gamma_c, min=tiny))  # (mc,)
        kappa_eff = kappa_c * inv_sqrt_gamma                               # (mc,)
        norm_fac = (1.0 / math.pi) * inv_sqrt_gamma.view(1, mc)            # (1,mc)

        # лог-суммирование по узлам (p,q)
        Amax = None
        Sexp = None

        # двойной цикл по узлам GH
        for p in range(quad_nodes):
            t1 = tau[p]
            for q in range(quad_nodes):
                t2 = tau[q]

                # аккумулируем сумму по координатам в лог-домене
                log_prod = torch.zeros((B, mc), device=device, dtype=r_dtype)

                for n0 in range(0, N, n_chunk):
                    n1 = min(n0 + n_chunk, N)

                    # u = Δz - (κ/√γ) * ( t1 * bR + t2 * bI )
                    shift = (
                        t1 * bR_c[:, n0:n1] + t2 * bI_c[:, n0:n1]
                    )                               # (mc, Nchunk)
                    u = dz[:, :, n0:n1] - kappa_eff.view(1, mc, 1) * shift.view(1, mc, -1)  # (B,mc,Nchunk)

                    # θ по выбранной форме
                    if theta_mode == "direct":
                        Wmax = int(W_c.max().item())
                        theta_val = _theta_direct(u, a_c.view(1, mc, 1), Wmax)
                    elif theta_mode == "poisson":
                        Kpmax = int(Kp_c.max().item())
                        theta_val = _theta_poisson(u, a_c.view(1, mc, 1), Kpmax)
                    else:
                        Wmax = int(W_c.max().item())
                        Kpmax = int(Kp_c.max().item())
                        theta_dir = _theta_direct(u, a_c.view(1, mc, 1), Wmax)
                        theta_sp  = _theta_poisson(u, a_c.view(1, mc, 1), Kpmax)
                        mask = use_direct_c.view(1, mc, 1)
                        theta_val = torch.where(mask, theta_dir, theta_sp)

                    theta_val = torch.clamp(theta_val, min=tiny)
                    log_prod = log_prod + theta_val.log().sum(dim=-1)  # суммируем log θ по координатам

                # добавляем веса узлов GH в лог-сумме
                L = log_prod + (logw[p] + logw[q])  # (B,mc)
                if Amax is None:
                    Amax = L
                    Sexp = torch.ones_like(L)
                else:
                    Mx = torch.maximum(Amax, L)
                    Sexp = Sexp * torch.exp(Amax - Mx) + torch.exp(L - Mx)
                    Amax = Mx

        # η = (1/π)(1/√γ) * Σ_{p,q} w_p w_q ∏_i θ_a(·)
        eta = norm_fac * torch.exp(Amax) * Sexp         # (B,mc)

        # итоговый вес по вкладам
        weight = (alpha_c.view(1, mc) * ang * eta).to(c_dtype)  # (B,mc)
        # свернуть по j
        T_out = T_out + (weight.unsqueeze(-1) * T_hat_c.view(1, mc, S)).sum(dim=1)

    return T_out
