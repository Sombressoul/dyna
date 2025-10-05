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
        k = torch.arange(1, n, device=device, dtype=dtype)
        off = torch.sqrt(k * 0.5)
        J = torch.zeros((n, n), device=device, dtype=dtype)
        J = J + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
        evals, evecs = torch.linalg.eigh(J)
        w0 = evecs[0, :] ** 2
        PI = torch.tensor(math.pi, dtype=dtype, device=device)
        weights = torch.sqrt(PI) * w0
        return evals, weights

    def _theta1d_phase(a: torch.Tensor, lam: torch.Tensor, K: int, PI: torch.Tensor) -> torch.Tensor:
        L, QQ, N = a.shape
        device = a.device
        dr = a.dtype
        n = torch.arange(-K, K + 1, dtype=dr, device=device).view(1, 1, 1, -1)
        a_b = a.unsqueeze(-1)
        lam_b = lam.view(-1, 1, 1, 1)
        phase = torch.exp((2.0 * PI).to(dr) * 1j * (n * a_b).to(torch.complex64 if dr == torch.float32 else torch.complex128))
        expo = torch.exp(-PI.to(dr) * (n ** 2) / lam_b)
        return (expo.to(phase.dtype) * phase).sum(dim=-1)

    def _theta1d_shifted_recent(a: torch.Tensor, beta: torch.Tensor, lam: torch.Tensor, K: int, PI: torch.Tensor) -> torch.Tensor:
        L, QQ, N = a.shape
        device = a.device
        dr = a.dtype
        n0 = torch.round(beta)
        delta = beta - n0
        m = torch.arange(-K, K + 1, dtype=dr, device=device).view(1, 1, 1, -1)
        lam_b = lam.view(-1, 1, 1, 1)
        a_b = a.unsqueeze(-1)
        delta_b = delta.unsqueeze(-1)
        phase_n0 = torch.exp((2.0 * PI).to(dr) * 1j * (n0 * a).to(torch.complex64 if dr == torch.float32 else torch.complex128))
        phase_m = torch.exp((2.0 * PI).to(dr) * 1j * (m * a_b).to(torch.complex64 if dr == torch.float32 else torch.complex128))
        expo = torch.exp(-PI.to(dr) * ((m - delta_b) ** 2) / lam_b)
        inner = (expo.to(phase_m.dtype) * phase_m).sum(dim=-1)
        return phase_n0 * inner

    def _choose_K_unshifted(lam_max: float, eps_theta: float, dims: int, q2: int) -> int:
        per = max(eps_theta / max(1, dims * q2), 1e-16)
        val = math.sqrt(max(lam_max, 1.0e-12) * math.log(1.0 / per) / math.pi)
        K = int(math.ceil(val + 0.5))
        return max(K, 1)

    def _choose_K_shifted(lam_max: float, eps_theta: float, dims: int, q2: int, delta_max: float) -> int:
        per_star = max(eps_theta / max(1, dims * q2), 1.0e-16)
        val = math.sqrt(max(lam_max, 1.0e-12) * math.log(1.0 / per_star) / math.pi)
        Kf = delta_max - 1.0 + val + 0.5
        K = int(math.ceil(Kf))
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
    K_pos = _choose_K_unshifted(lam_for_K, eps_theta, dims=2 * N, q2=QQ)

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
        theta_R = _theta1d_phase(aR_eff, lam_sel, K_pos, PI)
        theta_I = _theta1d_phase(aI_eff, lam_sel, K_pos, PI)
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

        delta_R = beta_R - torch.round(beta_R)
        delta_I = beta_I - torch.round(beta_I)
        delta_abs_max_R = float(delta_R.abs().max().item()) if delta_R.numel() > 0 else 0.0
        delta_abs_max_I = float(delta_I.abs().max().item()) if delta_I.numel() > 0 else 0.0
        delta_abs_max = max(delta_abs_max_R, delta_abs_max_I)
        K_neg = _choose_K_shifted(float(lam_sel.max().item()), eps_theta, dims=2 * N, q2=QQ, delta_max=delta_abs_max)

        theta_R = _theta1d_shifted_recent(aR, beta_R, lam_sel, K_neg, PI)
        theta_I = _theta1d_shifted_recent(aI, beta_I, lam_sel, K_neg, PI)

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
