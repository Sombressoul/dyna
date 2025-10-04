import torch
import math

from dyna.lib.cpsf.functional.core_math import delta_vec_d, R, R_ext


def T_New(
    *,
    z: torch.Tensor,  # [B, N] (complex)
    z_j: torch.Tensor,  # [B, M, N] (complex)
    vec_d: torch.Tensor,  # [B, N] (complex)
    vec_d_j: torch.Tensor,  # [B, M, N] (complex)
    T_hat_j: torch.Tensor,  # [B, M, S] (complex)
    alpha_j: torch.Tensor,  # [B, M] (real)
    sigma_par: torch.Tensor,  # [B, M] (real > 0)
    sigma_perp: torch.Tensor,  # [B, M] (real > 0)
    error_budget: float = 1.0e-5,
) -> torch.Tensor:
    def _frac_unit(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + 0.5, 1.0) - 0.5

    def _choose_hermite_order(eps_quad: float) -> int:
        return 7 if eps_quad < 3e-6 else 5

    def _gh_nodes_weights(n: int, device, dtype):
        if n == 5:
            nodes = torch.tensor(
                [
                    -2.0201828704560856,
                    -0.9585724646138185,
                    0.0,
                    0.9585724646138185,
                    2.0201828704560856,
                ],
                dtype=dtype,
                device=device,
            )
            weights = torch.tensor(
                [
                    0.01995324205904591,
                    0.39361932315224116,
                    0.9453087204829419,
                    0.39361932315224116,
                    0.01995324205904591,
                ],
                dtype=dtype,
                device=device,
            )
        elif n == 7:
            nodes = torch.tensor(
                [
                    -2.651961356835233,
                    -1.673551628767471,
                    -0.816287882858965,
                    0.0,
                    0.816287882858965,
                    1.673551628767471,
                    2.651961356835233,
                ],
                dtype=dtype,
                device=device,
            )
            weights = torch.tensor(
                [
                    0.000971781245099519,
                    0.054515582819127,
                    0.425607252610127,
                    0.810264617556807,
                    0.425607252610127,
                    0.054515582819127,
                    0.000971781245099519,
                ],
                dtype=dtype,
                device=device,
            )
        else:
            raise ValueError("Hermite order must be 5 or 7.")
        return nodes, weights

    def _theta1d_poisson_shifted(a, beta, lam, K, PI):
        device = a.device
        dr = a.dtype
        n = torch.arange(-K, K + 1, dtype=dr, device=device)
        phase = torch.exp(
            (2.0 * PI).to(dr)[...]
            * 1j
            * (n * a[..., None]).to(
                torch.complex64 if dr == torch.float32 else torch.complex128
            )
        )
        expo = torch.exp(-PI.to(dr) * ((n - beta[..., None]) ** 2) / lam[..., None])
        s = (expo.to(phase.dtype) * phase).sum(dim=-1) / torch.sqrt(lam.to(dr))
        return s

    def _theta1d_poisson_no_shift(beta, lam, K, PI):
        device = beta.device
        dr = beta.dtype
        n = torch.arange(-K, K + 1, dtype=dr, device=device)
        expo = torch.exp(-PI.to(dr) * ((n - beta[..., None]) ** 2) / lam[..., None])
        s = expo.sum(dim=-1) / torch.sqrt(lam.to(dr))
        return s.to(torch.complex64 if dr == torch.float32 else torch.complex128)

    def _choose_K_for_theta(
        lam_max: float, eps_theta: float, dims: int, q2: int
    ) -> int:
        per = max(eps_theta / max(1, 2 * dims * q2), 1e-16)
        val = math.sqrt(max(lam_max, 1e-12) * math.log(1.0 / per) / math.pi)
        K = int(math.ceil(val + 0.5))
        return max(K, 1)

    device = z.device
    rdt = z.real.dtype
    cdt = z.dtype
    B, M, N = vec_d_j.shape

    PI = torch.tensor(math.pi, dtype=rdt, device=device)

    z_b = z.unsqueeze(1).expand(B, M, N)
    d_b = vec_d.unsqueeze(1).expand(B, M, N)

    dz = z_b - z_j
    dz_re = _frac_unit(dz.real)
    dz_im = _frac_unit(dz.imag)

    delta_d = delta_vec_d(d_b, vec_d_j)
    dd2 = (delta_d.real**2 + delta_d.imag**2).sum(dim=-1)
    C_dir = torch.exp(-PI * dd2 / sigma_perp).to(cdt)

    denom = torch.sqrt(
        (vec_d_j.real**2 + vec_d_j.imag**2).sum(dim=-1, keepdim=True).clamp_min(1e-30)
    )
    b = (vec_d_j / denom).to(cdt)
    b_conj = b.conj()

    eps_total = float(error_budget)
    eps_quad = 0.4 * eps_total
    eps_theta = 0.6 * eps_total

    q_order = _choose_hermite_order(eps_quad)
    gh_nodes, gh_w = _gh_nodes_weights(q_order, device=device, dtype=rdt)
    Q = gh_nodes.numel()
    Xr = gh_nodes.view(Q, 1).expand(Q, Q).reshape(-1)
    Xi = gh_nodes.view(1, Q).expand(Q, Q).reshape(-1)
    Wr2d = (gh_w.view(Q, 1) * gh_w.view(1, Q)).reshape(-1)
    QQ = Xr.numel()

    lam_real_full = 1.0 / sigma_perp
    mu_real_full = (1.0 / sigma_par) - (1.0 / sigma_perp)
    use_real_full = mu_real_full >= 0.0

    def flat3(x):
        return x.reshape(B * M, x.shape[-1])

    def flat2(x):
        return x.reshape(B * M)

    dz_re_f = flat3(dz_re)
    dz_im_f = flat3(dz_im)
    b_conj_f = flat3(b_conj)
    lam_real_f = flat2(lam_real_full)
    mu_real_f = flat2(mu_real_full)
    sigma_par_f = flat2(sigma_par)
    sigma_perp_f = flat2(sigma_perp)

    mask_real = use_real_full.reshape(-1)
    mask_dual = (~use_real_full).reshape(-1)

    Theta_pos_flat = torch.zeros(B * M, dtype=cdt, device=device)

    Xr_b = Xr.view(1, QQ, 1).to(rdt)
    Xi_b = Xi.view(1, QQ, 1).to(rdt)
    Wr_b = Wr2d.view(1, QQ).to(rdt)

    if mask_real.any():
        lam_sel = lam_real_f[mask_real]
        mu_sel = mu_real_f[mask_real]
        dz_re_sel = dz_re_f[mask_real, :]
        dz_im_sel = dz_im_f[mask_real, :]
        bconj_sel = b_conj_f[mask_real, :]

        Kterms = max(
            _choose_K_for_theta(
                float(lam_sel.max().item()), eps_theta, dims=2 * N, q2=QQ
            )
            + 1,
            2,
        )

        s = torch.sqrt(torch.clamp(mu_sel, min=0.0) / PI)
        s_b = s.view(-1, 1, 1)

        xi = s_b.to(cdt) * (Xr_b + 1j * Xi_b).to(cdt)
        gamma = xi * bconj_sel.to(cdt).unsqueeze(1)
        beta_R = gamma.real
        beta_I = -gamma.imag

        lam_b = lam_sel.view(-1, 1, 1)
        aR, aI = dz_re_sel.unsqueeze(1), dz_im_sel.unsqueeze(1)

        theta_R = _theta1d_poisson_shifted(aR, beta_R, lam_b, Kterms, PI)
        theta_I = _theta1d_poisson_shifted(aI, beta_I, lam_b, Kterms, PI)

        emin = torch.finfo(rdt).tiny
        mag_log = torch.log(theta_R.abs().clamp_min(emin)).sum(dim=-1) + torch.log(
            theta_I.abs().clamp_min(emin)
        ).sum(dim=-1)
        ang_sum = torch.angle(theta_R).sum(dim=-1) + torch.angle(theta_I).sum(dim=-1)
        f_l = torch.exp(mag_log).to(cdt) * torch.exp(1j * ang_sum)

        acc = (f_l * Wr_b).sum(dim=-1) * (1.0 / math.pi)
        Theta_pos_flat[mask_real] = acc.to(cdt)

    if mask_dual.any():
        sp_sel = sigma_par_f[mask_dual]
        ss_sel = sigma_perp_f[mask_dual]
        dz_re_sel = dz_re_f[mask_dual, :]
        dz_im_sel = dz_im_f[mask_dual, :]
        bconj_sel = b_conj_f[mask_dual, :]

        lam_dual = ss_sel
        mu_dual = sp_sel - ss_sel

        Kterms = max(
            _choose_K_for_theta(
                float(lam_dual.max().item()), eps_theta, dims=2 * N, q2=QQ
            )
            + 1,
            2,
        )

        s = torch.sqrt(torch.clamp(mu_dual, min=0.0) / PI)
        s_b = s.view(-1, 1, 1)

        xi = s_b.to(cdt) * (Xr_b + 1j * Xi_b).to(cdt)
        gamma = xi * bconj_sel.to(cdt).unsqueeze(1)

        beta_R_tot = dz_re_sel.unsqueeze(1) + gamma.real
        beta_I_tot = dz_im_sel.unsqueeze(1) - gamma.imag

        lam_b = lam_dual.view(-1, 1, 1)
        phi_R = _theta1d_poisson_no_shift(beta_R_tot, lam_b, Kterms, PI)
        phi_I = _theta1d_poisson_no_shift(beta_I_tot, lam_b, Kterms, PI)

        emin = torch.finfo(rdt).tiny
        mag_log = torch.log(phi_R.abs().clamp_min(emin)).sum(dim=-1) + torch.log(
            phi_I.abs().clamp_min(emin)
        ).sum(dim=-1)
        ang_sum = torch.angle(phi_R).sum(dim=-1) + torch.angle(phi_I).sum(dim=-1)
        f_l = torch.exp(mag_log).to(cdt) * torch.exp(1j * ang_sum)

        acc = (f_l * Wr_b).sum(dim=-1) * (1.0 / math.pi)
        pref = (sp_sel * (ss_sel ** (N - 1))).to(rdt)
        Theta_pos_flat[mask_dual] = pref.to(cdt) * acc.to(cdt)

    Theta_pos = Theta_pos_flat.view(B, M)
    eta_j = C_dir * Theta_pos
    w = (alpha_j * eta_j.real).unsqueeze(-1).to(T_hat_j.dtype)
    T = (w * T_hat_j).sum(dim=1)

    return T
