import torch
import math

from dyna.lib.cpsf.functional.core_math import delta_vec_d, R, R_ext


def T_New(
    *,
    z: torch.Tensor,
    z_j: torch.Tensor,
    vec_d: torch.Tensor,
    vec_d_j: torch.Tensor,
    T_hat_j: torch.Tensor,
    alpha_j: torch.Tensor,
    sigma_par: torch.Tensor,
    sigma_perp: torch.Tensor,
    error_budget: float = 1.0e-5,
) -> torch.Tensor:
    def _frac_unit(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + 0.5, 1.0) - 0.5

    def _choose_gh_order(eps_quad: float) -> int:
        if eps_quad < 3e-7:
            return 9
        elif eps_quad < 3e-6:
            return 7
        else:
            return 5

    def _gh_nodes_weights(n: int, device, dtype):
        if n == 5:
            nodes = torch.tensor(
                [-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856],
                dtype=dtype, device=device
            )
            weights = torch.tensor(
                [0.01995324205904591, 0.39361932315224116, 0.9453087204829419, 0.39361932315224116, 0.01995324205904591],
                dtype=dtype, device=device
            )
        elif n == 7:
            nodes = torch.tensor(
                [-2.651961356835233, -1.673551628767471, -0.816287882858965, 0.0,
                 0.816287882858965, 1.673551628767471, 2.651961356835233],
                dtype=dtype, device=device
            )
            weights = torch.tensor(
                [0.000971781245099519, 0.054515582819127, 0.425607252610127, 0.810264617556807,
                 0.425607252610127, 0.054515582819127, 0.000971781245099519],
                dtype=dtype, device=device
            )
        elif n == 9:
            nodes = torch.tensor(
                [-2.9592107790638380, -2.0232301911005157, -1.2247448713915890,
                 -0.5240335474869576, 0.0,
                  0.5240335474869576,  1.2247448713915890,  2.0232301911005157,  2.9592107790638380],
                dtype=dtype, device=device
            )
            weights = torch.tensor(
                [0.000199604072211367, 0.017077983007413, 0.207802325814892,
                 0.661147012558241,   0.981560634246719,
                 0.661147012558241,   0.207802325814892, 0.017077983007413, 0.000199604072211367],
                dtype=dtype, device=device
            )
        else:
            raise ValueError("Hermite order must be one of {5,7,9}.")
        return nodes, weights

    def _theta1d_phase(a: torch.Tensor, lam: torch.Tensor, K: int, PI: torch.Tensor) -> torch.Tensor:
        L, QQ, N = a.shape
        device = a.device
        dr = a.dtype
        n = torch.arange(-K, K + 1, dtype=dr, device=device).view(1, 1, 1, -1)
        a_b = a.unsqueeze(-1)
        lam_b = lam.view(-1, 1, 1, 1)
        phase = torch.exp(
            (2.0 * PI).to(dr)
            * 1j
            * (n * a_b).to(torch.complex64 if dr == torch.float32 else torch.complex128)
        )
        expo = torch.exp(-PI.to(dr) * (n ** 2) / lam_b)
        return (expo.to(phase.dtype) * phase).sum(dim=-1)

    def _theta1d_shifted(a: torch.Tensor, beta: torch.Tensor, lam: torch.Tensor, K: int, PI: torch.Tensor) -> torch.Tensor:
        L, QQ, N = a.shape
        device = a.device
        dr = a.dtype
        n = torch.arange(-K, K + 1, dtype=dr, device=device).view(1, 1, 1, -1)
        a_b = a.unsqueeze(-1)
        beta_b = beta.unsqueeze(-1)
        lam_b = lam.view(-1, 1, 1, 1)
        phase = torch.exp(
            (2.0 * PI).to(dr)
            * 1j
            * (n * a_b).to(torch.complex64 if dr == torch.float32 else torch.complex128)
        )
        expo = torch.exp(-PI.to(dr) * ((n - beta_b) ** 2) / lam_b)
        return (expo.to(phase.dtype) * phase).sum(dim=-1)

    def _choose_K_for_theta(lam_max: float, eps_theta: float, dims: int, q2: int) -> int:
        per = max(eps_theta / max(1, dims * q2), 1e-16)
        val = math.sqrt(max(lam_max, 1.0e-12) * math.log(1.0 / per) / math.pi)
        K = int(math.ceil(val + 0.5))
        return max(K, 1)

    device = z.device
    rdt = z.real.dtype
    cdt = z.dtype
    B, M, N = vec_d_j.shape

    PI = torch.tensor(math.pi, dtype=rdt, device=device)
    tiny = torch.tensor(torch.finfo(rdt).tiny, dtype=rdt, device=device)

    z_b = z.unsqueeze(1).expand(B, M, N)
    d_b = vec_d.unsqueeze(1).expand(B, M, N)
    dz = z_b - z_j
    dz_re = _frac_unit(dz.real)
    dz_im = _frac_unit(dz.imag)

    delta_d = delta_vec_d(d_b, vec_d_j)
    dd2 = (delta_d.real**2 + delta_d.imag**2).sum(dim=-1)
    C_dir = torch.exp(-PI * dd2 / sigma_perp).to(cdt)

    denom = torch.sqrt((vec_d_j.real**2 + vec_d_j.imag**2).sum(dim=-1).clamp_min(tiny))
    b = (vec_d_j / denom.unsqueeze(-1)).to(cdt)
    b_conj = b.conj()

    eps_total = float(error_budget)
    eps_quad = 0.4 * eps_total
    eps_theta = 0.6 * eps_total

    q_order = _choose_gh_order(eps_quad)
    gh_nodes, gh_w = _gh_nodes_weights(q_order, device=device, dtype=rdt)
    Q = gh_nodes.numel()
    Xr = gh_nodes.view(Q, 1).expand(Q, Q).reshape(-1)
    Xi = gh_nodes.view(1, Q).expand(Q, Q).reshape(-1)
    Wr2d = (gh_w.view(Q, 1) * gh_w.view(1, Q)).reshape(-1)
    QQ = Xr.numel()

    lam_base = 1.0 / sigma_perp
    lam_for_K = float(lam_base.max().item())
    Kterms = _choose_K_for_theta(lam_for_K, eps_theta, dims=2 * N, q2=QQ)

    def flat3(x): return x.reshape(B * M, x.shape[-1])
    def flat2(x): return x.reshape(B * M)

    dz_re_f = flat3(dz_re)
    dz_im_f = flat3(dz_im)
    b_conj_f = flat3(b_conj)

    sigma_par_f = flat2(sigma_par)
    sigma_perp_f = flat2(sigma_perp)
    lam_base_f = flat2(lam_base)

    mu = sigma_par_f - sigma_perp_f
    mask_pos = (mu >= 0.0)
    mask_neg = ~mask_pos

    Theta_pos_flat = torch.zeros(B * M, dtype=cdt, device=device)

    Xr_b = Xr.view(1, QQ, 1).to(rdt)
    Xi_b = Xi.view(1, QQ, 1).to(rdt)
    Wr_b = Wr2d.view(1, QQ).to(rdt)

    pref_global = (sigma_par_f * (sigma_perp_f ** (N - 1))).to(rdt).to(cdt)

    if mask_pos.any():
        idx = mask_pos

        lam_sel = lam_base_f[idx]
        mu_sel = mu[idx].clamp_min(0.0)
        dz_re_sel = dz_re_f[idx, :]
        dz_im_sel = dz_im_f[idx, :]
        bconj_sel = b_conj_f[idx, :]

        s = torch.sqrt(PI * mu_sel)
        xi = s.view(-1, 1, 1).to(cdt) * (Xr_b + 1j * Xi_b).to(cdt)
        gamma = xi * bconj_sel.to(cdt).unsqueeze(1)

        aR_eff = dz_re_sel.unsqueeze(1).expand(-1, QQ, -1) + (gamma.real / PI)
        aI_eff = dz_im_sel.unsqueeze(1).expand(-1, QQ, -1) - (gamma.imag / PI)

        theta_R = _theta1d_phase(aR_eff, lam_sel, Kterms, PI)
        theta_I = _theta1d_phase(aI_eff, lam_sel, Kterms, PI)

        mag_log = torch.log(theta_R.abs().clamp_min(tiny)).sum(dim=-1) + torch.log(theta_I.abs().clamp_min(tiny)).sum(dim=-1)
        ang_sum = torch.angle(theta_R).sum(dim=-1) + torch.angle(theta_I).sum(dim=-1)
        f_l = torch.exp(mag_log).to(cdt) * torch.exp(1j * ang_sum)

        acc = (f_l * Wr_b.to(cdt)).sum(dim=-1) * (1.0 / math.pi)
        Theta_pos_flat[idx] = pref_global[idx] * acc

    if mask_neg.any():
        idx = mask_neg

        lam_sel = lam_base_f[idx]
        mu_abs = (sigma_perp_f[idx] - sigma_par_f[idx]).clamp_min(0.0)
        dz_re_sel = dz_re_f[idx, :]
        dz_im_sel = dz_im_f[idx, :]
        bconj_sel = b_conj_f[idx, :]

        s = torch.sqrt(PI * mu_abs)
        xi = s.view(-1, 1, 1).to(cdt) * (Xr_b + 1j * Xi_b).to(cdt)
        gamma = xi * bconj_sel.to(cdt).unsqueeze(1)

        cR = (gamma.real / PI)
        cI = (-gamma.imag / PI)

        beta_R = lam_sel.view(-1, 1, 1) * cR
        beta_I = lam_sel.view(-1, 1, 1) * cI

        aR = dz_re_sel.unsqueeze(1).expand(-1, QQ, -1)
        aI = dz_im_sel.unsqueeze(1).expand(-1, QQ, -1)

        theta_R = _theta1d_shifted(aR, beta_R, lam_sel, Kterms, PI)
        theta_I = _theta1d_shifted(aI, beta_I, lam_sel, Kterms, PI)

        gain_log = (PI * lam_sel.view(-1, 1, 1)) * (cR.pow(2) + cI.pow(2))
        gain_log = gain_log.sum(dim=-1)

        mag_log = torch.log(theta_R.abs().clamp_min(tiny)).sum(dim=-1) + torch.log(theta_I.abs().clamp_min(tiny)).sum(dim=-1)
        mag_log = mag_log + gain_log
        ang_sum = torch.angle(theta_R).sum(dim=-1) + torch.angle(theta_I).sum(dim=-1)
        f_l = torch.exp(mag_log).to(cdt) * torch.exp(1j * ang_sum)

        acc = (f_l * Wr_b.to(cdt)).sum(dim=-1) * (1.0 / math.pi)
        Theta_pos_flat[idx] = pref_global[idx] * acc

    Theta_pos = Theta_pos_flat.view(B, M)
    eta_j = C_dir * Theta_pos
    w = (alpha_j * eta_j.real).unsqueeze(-1).to(T_hat_j.dtype)
    T = (w * T_hat_j).sum(dim=1)
    return T
