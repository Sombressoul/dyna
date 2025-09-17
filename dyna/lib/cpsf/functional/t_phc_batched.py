import math
import torch


from dyna.lib.cpsf.functional.core_math import delta_vec_d


@torch.jit.script
def _int_pow(
    base: torch.Tensor,
    n: int,
) -> torch.Tensor:
    out = torch.ones_like(base)
    b = base
    k = int(n)
    while k > 0:
        if (k & 1) != 0:
            out = out * b
        b = b * b
        k >>= 1
    return out


@torch.jit.script
def fused_sincos(
    Aphase: torch.Tensor,
    phi: torch.Tensor,
    psi: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    cA, sA = torch.cos(Aphase), torch.sin(Aphase)
    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cpsi, spsi = torch.cos(psi), torch.sin(psi)
    return cA, sA, cphi, sphi, cpsi, spsi


def T_PHC_Batched(
    z: torch.Tensor,
    vec_d: torch.Tensor,
    z_j: torch.Tensor,
    vec_d_j: torch.Tensor,
    T_hat_j: torch.Tensor,
    alpha_j: torch.Tensor,
    sigma_par_j: torch.Tensor,
    sigma_perp_j: torch.Tensor,
    *,
    quad_nodes: int = 7,
    eps_total: float = 1.0e-3,
    n_chunk: int = 256,
    m_chunk: int = 256,
    dtype_override: torch.dtype | None = None,
):
    """
    T_PHC_Batched: Poisson-Hermite-Clenshaw evaluator for the torus-periodic field T.

    Computes the complex field T for a batch of query points against a set of per-query
    contributions using the PHC algorithm.

    Parameters
    ----------
    z           : Complex tensor of shape [B, N]
        Query points on the torus. Only Re(z) is wrapped to the lattice; Im(z) is free.
    vec_d       : Complex tensor of shape [B, N]
        Unit direction vectors for each query (unit norm along the last dimension is
        expected).
    z_j         : Complex tensor of shape [B, M, N]
        Per-query contribution centers. Only Re(z_j) participates in wrap with Re(z);
        Im(z_j) is free.
    vec_d_j     : Complex tensor of shape [B, M, N]
        Per-query unit direction vectors for each contribution (unit norm along the
        last dimension is expected).
    T_hat_j     : Complex tensor of shape [B, M, S]
        Per-contribution spectral profiles (columns index S output channels or features).
    alpha_j     : Real tensor of shape [B, M] or broadcastable to [B, M]
        Per-contribution scalar weights. Sign is preserved.
    sigma_par_j : Real positive tensor of shape [B, M] or broadcastable to [B, M]
        Parallel variance parameter (along vec_d_j). Must satisfy sigma_par_j > 0 and
        typically sigma_par_j >= sigma_perp_j.
    sigma_perp_j  : Real positive tensor of shape [B, M] or broadcastable to [B, M]
        Perpendicular variance parameter. Must satisfy sigma_perp_j > 0.
    quad_nodes  : Integer
        Number of Gauss-Hermite nodes per axis (default 12). Larger values increase
        accuracy and cost.
    eps_total   : Float
        Total error budget for the PHC evaluation (default 1e-3). Internally split
        between quadrature and theta tail to guarantee a relative L2 error target
        on active tiles.
    n_chunk     : Int (optional)
        Tile size over the Hermite integration dimension. Use None to auto-select
        from available memory.
    m_chunk     : Int (optional)
        Tile size over contributions M. Use None to auto-select from available memory.
    """

    device = z.device
    c_dtype = z.dtype if dtype_override is None else dtype_override
    r_dtype = torch.float32 if c_dtype == torch.complex64 else torch.float64
    PI = math.pi
    TWO_PI = 2.0 * math.pi
    tiny = torch.as_tensor(torch.finfo(r_dtype).tiny, device=device, dtype=r_dtype)

    if not hasattr(T_PHC_Batched, "_gh1d_cache"):
        T_PHC_Batched._gh1d_cache = {}

    gh_key = (device.type, getattr(device, "index", -1), str(r_dtype), int(quad_nodes))
    cached = T_PHC_Batched._gh1d_cache.get(gh_key)
    if cached is None:
        Q = quad_nodes
        if Q < 1:
            raise ValueError("quad_nodes must be >= 1")
        k = torch.arange(1, Q, device=device, dtype=r_dtype)
        off = torch.sqrt(k / 2.0)
        J = torch.zeros((Q, Q), device=device, dtype=r_dtype)
        J.diagonal(1).copy_(off)
        J.diagonal(-1).copy_(off)
        tau, V = torch.linalg.eigh(J)
        w = (math.sqrt(math.pi) * (V[0, :] ** 2)).to(r_dtype)
        logw_1d = torch.log(torch.clamp(w, min=tiny))
        T_PHC_Batched._gh1d_cache[gh_key] = (tau, logw_1d)
        tau_1d, logw_1d = tau, logw_1d
    else:
        tau_1d, logw_1d = cached
        Q = quad_nodes

    qc_full = Q * Q

    B, N = z.shape[-2], z.shape[-1]
    M = z_j.shape[-2]
    S_dim = T_hat_j.shape[-1]

    z = z.to(c_dtype)
    z_j = z_j.to(c_dtype)
    vec_d = vec_d.to(c_dtype)
    vec_d_j = vec_d_j.to(c_dtype)
    T_hat_j = T_hat_j.to(c_dtype)
    alpha_j = alpha_j.to(r_dtype)
    sigma_par_j = sigma_par_j.to(r_dtype)
    sigma_perp_j = sigma_perp_j.to(r_dtype)

    a_j = 1.0 / torch.clamp(sigma_perp_j, min=tiny)
    gamma_j = torch.clamp(
        sigma_perp_j / torch.clamp(sigma_par_j, min=tiny), min=tiny, max=1.0
    )
    kappa_j = torch.sqrt(
        torch.clamp(
            (sigma_par_j - sigma_perp_j)
            * torch.clamp(sigma_perp_j, min=tiny)
            / (PI * torch.clamp(sigma_par_j, min=tiny)),
            min=0.0,
        )
    )
    inv_sqrt_gamma = 1.0 / torch.sqrt(torch.clamp(gamma_j, min=tiny))
    kappa_eff = kappa_j * inv_sqrt_gamma
    norm_fac = (1.0 / PI) * inv_sqrt_gamma
    eps_theta = torch.as_tensor(eps_total, dtype=r_dtype, device=device) / (
        2.0 * max(N, 1) * max(qc_full, 1)
    )
    a_det = a_j.detach()
    Kp_j = torch.ceil(
        torch.sqrt(
            torch.clamp(
                (a_det / PI) * (-torch.log(torch.clamp(eps_theta, min=tiny))), min=0.0
            )
        )
    ).to(torch.int64)
    one_over_sq = 1.0 / torch.clamp(sigma_perp_j, min=tiny)
    c_ang = (sigma_par_j - sigma_perp_j) / (
        torch.clamp(sigma_par_j * sigma_perp_j, min=tiny)
    )

    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        target_bytes = int(
            max(256 * 1024 * 1024, min(1_000 * 1024 * 1024, int(free_bytes * 0.40)))
        )
    else:
        target_bytes = 256 * 1024 * 1024

    bytes_per_elem = 4 if r_dtype == torch.float32 else 8
    T_out = torch.zeros((B, S_dim), device=device, dtype=c_dtype)

    # ======================= Main m-chunks =======================
    for m0 in range(0, M, m_chunk):
        m1 = min(m0 + m_chunk, M)
        mc = m1 - m0
        BM = B * mc

        z_j_c = z_j[:, m0:m1, :]
        a_c = a_j[:, m0:m1]
        alpha_c = alpha_j[:, m0:m1]
        T_hat_c = T_hat_j[:, m0:m1, :]
        Kp_c_det = Kp_j[:, m0:m1]
        kappa_eff_c = kappa_eff[:, m0:m1]
        norm_fac_c = norm_fac[:, m0:m1]

        inv_den = 1.0 / torch.linalg.vector_norm(
            vec_d_j[:, m0:m1, :], dim=-1, keepdim=True
        ).to(r_dtype).clamp_min(tiny)
        b = vec_d_j[:, m0:m1, :] * inv_den.to(c_dtype)
        bR = b.real.to(r_dtype)
        bI = b.imag.to(r_dtype)
        bR_eff = (kappa_eff_c.unsqueeze(-1) * bR).contiguous()
        bI_eff = (kappa_eff_c.unsqueeze(-1) * bI).contiguous()
        dZ = z.unsqueeze(1) - z_j_c
        dzR = torch.remainder(dZ.real.to(r_dtype) + 0.5, 1.0) - 0.5
        dzI = dZ.imag.to(r_dtype)

        dd = delta_vec_d(
            vec_d=vec_d.unsqueeze(1).expand(B, mc, N),
            vec_d_j=vec_d_j[:, m0:m1, :],
            eps=tiny,
        )
        dd_norm2 = (dd.abs() ** 2).sum(dim=-1).to(r_dtype)
        bh_dd = (torch.conj(b) * dd).sum(dim=-1)
        q_ang = one_over_sq[:, m0:m1] * dd_norm2 - c_ang[:, m0:m1] * (bh_dd.abs() ** 2)
        log_ang = (-PI) * q_ang

        # ======================= K-bucket (detached) across flattened (B*mc) =======================
        K_flat = Kp_c_det.reshape(BM)
        a_flat = a_c.reshape(BM)
        log_inv_sqrt_a_f = -0.5 * torch.log(torch.clamp(a_flat, min=tiny))
        buf_inv_a_f = 1.0 / torch.clamp(a_flat, min=tiny)
        buf_ex01_f = torch.exp(-PI * buf_inv_a_f)
        buf_ex02_f = buf_ex01_f * buf_ex01_f
        buf_ex04_f = buf_ex02_f * buf_ex02_f
        buf_ex08_f = buf_ex04_f * buf_ex04_f
        buf_ex16_f = buf_ex08_f * buf_ex08_f
        per_elem_bufs = 2
        ns_cap_den = max(1, B * mc * qc_full)
        ns_cap = max(
            1, int(target_bytes // (bytes_per_elem * per_elem_bufs * ns_cap_den))
        )
        step_ns = min(n_chunk, max(1, ns_cap))

        ns_ref = min(step_ns, N)
        denom_ref = max(1, B * mc * ns_ref)
        q_cap_ref = max(
            1, int(target_bytes // (bytes_per_elem * per_elem_bufs * denom_ref))
        )
        base_ref = max(1, int(math.sqrt(q_cap_ref)))
        q1_step = max(1, min(Q, base_ref))
        q2_step = max(1, min(Q, max(1, q_cap_ref // q1_step)))
        num_q1 = (Q + q1_step - 1) // q1_step
        num_q2 = (Q + q2_step - 1) // q2_step

        perm = torch.argsort(K_flat)
        K_sorted = K_flat[perm]
        if BM > 0:
            Kvals, counts = torch.unique_consecutive(K_sorted, return_counts=True)
            starts = torch.cumsum(
                torch.cat(
                    [torch.zeros(1, device=device, dtype=counts.dtype), counts[:-1]]
                ),
                dim=0,
            )
            num_groups = Kvals.numel()
        else:
            num_groups = 0

        q_accum = [
            [[None for _ in range(num_q2)] for _ in range(num_q1)]
            for _ in range(num_groups)
        ]
        q_lwblk = [[None for _ in range(num_q2)] for _ in range(num_q1)]
        L_accum_flat = torch.full((BM, 1), -float("inf"), device=device, dtype=r_dtype)

        # ======================= n-tiles =======================
        for n0 in range(0, N, step_ns):
            n1 = min(n0 + step_ns, N)
            ns = n1 - n0

            Aphase = TWO_PI * dzR[:, :, n0:n1]
            phi = TWO_PI * (
                bR_eff[:, :, n0:n1].unsqueeze(-1) * tau_1d.view(1, 1, 1, -1)
            )
            psi = TWO_PI * (
                bI_eff[:, :, n0:n1].unsqueeze(-1) * tau_1d.view(1, 1, 1, -1)
            )
            cA, sA, cphi, sphi, cpsi, spsi = fused_sincos(Aphase, phi, psi)

            # ======================= (q1,q2)-tiles =======================
            for i0 in range(0, Q, q1_step):
                i1 = min(i0 + q1_step, Q)
                q1 = i1 - i0

                cphi_i = cphi[:, :, :, i0:i1]
                sphi_i = sphi[:, :, :, i0:i1]

                for j0 in range(0, Q, q2_step):
                    j1 = min(j0 + q2_step, Q)
                    q2 = j1 - j0

                    cpsi_j = cpsi[:, :, :, j0:j1]
                    spsi_j = spsi[:, :, :, j0:j1]

                    cB = cphi_i.unsqueeze(-1) * cpsi_j.unsqueeze(-2) - sphi_i.unsqueeze(
                        -1
                    ) * spsi_j.unsqueeze(-2)
                    sB = sphi_i.unsqueeze(-1) * cpsi_j.unsqueeze(-2) + cphi_i.unsqueeze(
                        -1
                    ) * spsi_j.unsqueeze(-2)

                    x_block = (
                        cA.unsqueeze(-1).unsqueeze(-1) * cB
                        + sA.unsqueeze(-1).unsqueeze(-1) * sB
                    ).to(r_dtype)

                    q1q2 = q1 * q2
                    if q_lwblk[i0 // q1_step][j0 // q2_step] is None:
                        q_lwblk[i0 // q1_step][j0 // q2_step] = (
                            logw_1d[i0:i1].view(1, 1, q1, 1)
                            + logw_1d[j0:j1].view(1, 1, 1, q2)
                        ).view(1, 1, q1q2)

                    x_flat = x_block.view(BM, ns, q1q2)

                    # ======================= K-buckets (Chebyshev for K<=4, else Clenshaw) =======================
                    for g in range(num_groups):
                        Kval = int(Kvals[g].item())
                        s = int(starts[g].item())
                        e = s + int(counts[g].item())
                        idx = perm[s:e]
                        m_g = e - s
                        if m_g == 0:
                            continue

                        xg = x_flat.index_select(0, idx)

                        if Kval == 0:
                            part = (
                                ns * log_inv_sqrt_a_f.index_select(0, idx).view(m_g, 1)
                            ).expand(m_g, q1q2)

                        elif Kval <= 4:
                            ex01 = buf_ex01_f.index_select(0, idx).view(m_g, 1, 1)
                            ex04 = buf_ex04_f.index_select(0, idx).view(m_g, 1, 1)
                            ex08 = buf_ex08_f.index_select(0, idx).view(m_g, 1, 1)
                            ex16 = buf_ex16_f.index_select(0, idx).view(m_g, 1, 1)

                            x2 = xg * xg
                            S = ex01 * xg
                            if Kval >= 2:
                                S = S + ex04 * (2.0 * x2 - 1.0)
                            if Kval >= 3:
                                S = S + (ex08 * ex01) * (4.0 * x2 * xg - 3.0 * xg)
                            if Kval == 4:
                                S = S + ex16 * (8.0 * x2 * x2 - 8.0 * x2 + 1.0)

                            sum_expr = torch.log1p(
                                torch.clamp(2.0 * S, min=-1.0 + tiny)
                            )
                            part = sum_expr.sum(dim=1) + (
                                ns * log_inv_sqrt_a_f.index_select(0, idx).view(m_g, 1)
                            )

                        else:
                            ex01 = buf_ex01_f.index_select(0, idx).view(m_g, 1, 1)
                            ex02 = buf_ex02_f.index_select(0, idx).view(m_g, 1, 1)
                            rKm1 = ex01 * _int_pow(ex02, Kval - 1)

                            if (Kval & 1) == 0:
                                cK = _int_pow(ex02, (Kval * Kval) // 2)
                            else:
                                cK = ex01 * _int_pow(ex02, (Kval * Kval - 1) // 2)

                            inv_r_init = 1.0 / rKm1
                            inv_ex02 = 1.0 / ex02

                            b1 = torch.zeros_like(xg, dtype=r_dtype)
                            b2 = torch.zeros_like(xg, dtype=r_dtype)
                            c = cK.clone()
                            r = rKm1.clone()
                            inv_r = inv_r_init.clone()

                            for _ in range(Kval, 0, -1):
                                b0 = 2.0 * xg * b1 - b2 + c
                                b2, b1 = b1, b0
                                c = c * inv_r
                                r = r * inv_ex02
                                tail = (c * r) / torch.clamp(1.0 - r, min=tiny)
                                if float(tail.max().detach()) <= float(eps_theta):
                                    break

                            S = b1 * xg - b2
                            sum_expr = torch.log1p(
                                torch.clamp(2.0 * S, min=-1.0 + tiny)
                            )
                            part = sum_expr.sum(dim=1) + (
                                ns * log_inv_sqrt_a_f.index_select(0, idx).view(m_g, 1)
                            )

                        i_idx = i0 // q1_step
                        j_idx = j0 // q2_step
                        acc = q_accum[g][i_idx][j_idx]
                        if acc is None:
                            q_accum[g][i_idx][j_idx] = part
                        else:
                            q_accum[g][i_idx][j_idx] = acc + part

        for i_idx in range(num_q1):
            for j_idx in range(num_q2):
                lw_blk = q_lwblk[i_idx][j_idx]
                if lw_blk is None:
                    continue
                lw_blk_flat = lw_blk.view(1, -1)
                for g in range(num_groups):
                    s = int(starts[g].item())
                    e = s + int(counts[g].item())
                    idx = perm[s:e]
                    m_g = e - s
                    if m_g == 0:
                        continue
                    acc = q_accum[g][i_idx][j_idx]
                    if acc is None:
                        continue
                    L = acc + lw_blk_flat
                    group_lse = torch.logsumexp(L, dim=-1, keepdim=True)
                    cur = L_accum_flat.index_select(0, idx)
                    combined = torch.logsumexp(
                        torch.cat([cur, group_lse], dim=-1), dim=-1, keepdim=True
                    )
                    L_accum_flat.index_copy_(0, idx, combined)

        L_accum = L_accum_flat.view(B, mc, 1)
        u = dzI
        u2_sum = (u * u).sum(dim=2)
        proj_r = (bR * u).sum(dim=2)
        proj_i = (bI * u).sum(dim=2)
        proj_abs2 = proj_r * proj_r + proj_i * proj_i
        log_im = (-PI) * (a_c * u2_sum) + (PI) * (c_ang[:, m0:m1] * proj_abs2)
        log_eta = (
            torch.log(torch.clamp(norm_fac_c, min=tiny)) + L_accum.squeeze(-1) + log_im
        )
        alpha_abs = alpha_c.abs()
        log_alpha_c = torch.where(
            alpha_abs == 0,
            torch.full_like(alpha_abs, -float("inf")),
            torch.log(alpha_abs),
        )
        sign_full = torch.sign(alpha_c).to(torch.float32)
        log_w_full = log_eta + log_ang + log_alpha_c
        row_max = torch.amax(log_w_full, dim=1, keepdim=True)
        w32 = torch.exp(log_w_full - row_max).to(torch.float32) * sign_full

        R = T_hat_c.real.to(torch.float32)
        I = T_hat_c.imag.to(torch.float32)

        if device.type == "cuda":
            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                part_R = torch.bmm(w32.unsqueeze(1), R).squeeze(1)
                part_I = torch.bmm(w32.unsqueeze(1), I).squeeze(1)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        else:
            part_R = torch.bmm(w32.unsqueeze(1), R).squeeze(1)
            part_I = torch.bmm(w32.unsqueeze(1), I).squeeze(1)

        scale = torch.exp(row_max.squeeze(1)).to(part_R.dtype)
        T_out.real = T_out.real + (scale.unsqueeze(1) * part_R).to(T_out.real.dtype)
        T_out.imag = T_out.imag + (scale.unsqueeze(1) * part_I).to(T_out.imag.dtype)

    return T_out
