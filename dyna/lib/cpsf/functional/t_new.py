import torch
import math

from dyna.lib.cpsf.functional.core_math import delta_vec_d

@torch.jit.script
def flat3(
    *,
    x: torch.Tensor, 
    B: int, 
    M: int,
) -> torch.Tensor: 
    return x.reshape(B * M, x.shape[-1])

@torch.jit.script
def flat2(
    *,
    x: torch.Tensor, 
    B: int, 
    M: int,
) -> torch.Tensor: 
    return x.reshape(B * M)

@torch.jit.script
def _choose_K_unshifted(
    *,
    lam_max: float,
    eps_theta: float,
    N: int,
    QQ: int,
) -> int:
    per = max(eps_theta / max(1, N * QQ), 1e-16)
    val = math.sqrt(max(lam_max, 1.0e-12) * math.log(1.0 / per) / math.pi)
    K = int(math.ceil(val + 0.5))
    return max(K, 1)

@torch.jit.script
def _choose_K_shifted(
    *,
    lam_sel: torch.Tensor,
    delta_R: torch.Tensor,
    delta_I: torch.Tensor,
    N: int,
    QQ: int,
    eps_theta: float,
    tiny: float,
    PI: torch.Tensor,
    dtype_r: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    lam_max = lam_sel.max()
    delta_abs_max = torch.maximum(delta_R.abs().max(), delta_I.abs().max())

    eps_star = torch.tensor(eps_theta / (2 * N * QQ), dtype=dtype_r, device=device)
    one_r = torch.tensor(1.0, dtype=dtype_r, device=device)
    two_r = torch.tensor(2.0, dtype=dtype_r, device=device)
    half_r = torch.tensor(0.5, dtype=dtype_r, device=device)

    eps_star = eps_star.clamp_min(tiny).clamp_max(one_r - tiny)

    val = torch.sqrt(lam_max / PI * torch.log(one_r / eps_star))
    K0 = torch.ceil(delta_abs_max - one_r + val + half_r).to(torch.int64) + 1
    K0 = torch.clamp(K0, min=1)

    A = torch.sqrt(PI / lam_max)
    y_eps = (eps_star * torch.sqrt(PI / lam_max)).clamp_min(tiny).clamp_max(two_r - tiny)
    u_eps = -torch.special.ndtri(y_eps * half_r) / math.sqrt(2.0)

    u0 = A * (K0.to(dtype_r) + one_r - delta_abs_max)
    deltaK = torch.clamp((u_eps - u0) / A, min=torch.tensor(0.0, dtype=dtype_r, device=device))

    K_neg = torch.clamp(K0 + torch.ceil(deltaK).to(torch.int64), min=1)

    return K_neg

@torch.jit.script
def _theta1d_phase(
    *,
    a: torch.Tensor, 
    lam: torch.Tensor, 
    K: int, 
    PI: torch.Tensor,
    dtype_r: torch.dtype,
    dtype_c: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    n = torch.arange(-K, K + 1, dtype=dtype_r, device=device).view(1, 1, 1, -1)
    a_b = a.unsqueeze(-1)
    lam_b = lam.view(-1, 1, 1, 1)
    phase = torch.exp((2.0 * PI).to(dtype_r) * 1j * (n * a_b).to(dtype_c))
    expo = torch.exp(-PI.to(dtype_r) * (n ** 2) / lam_b)
    return (expo.to(phase.dtype) * phase).sum(dim=-1)

@torch.jit.script
def _theta1d_shifted(
    *,
    a: torch.Tensor,
    beta: torch.Tensor,
    lam: torch.Tensor,
    K: int,
    PI: torch.Tensor,
    dtype_r: torch.dtype,
    dtype_c: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    n0 = torch.floor(beta + 0.5)
    delta = beta - n0
    m = torch.arange(-K, K + 1, dtype=dtype_r, device=device).view(1, 1, 1, -1)
    lam_b = lam.view(-1, 1, 1, 1)
    a_b = a.unsqueeze(-1)
    delta_b = delta.unsqueeze(-1)
    phase_n0 = torch.exp((2.0 * PI).to(dtype_r) * 1j * (n0 * a).to(dtype_c))
    phase_m = torch.exp((2.0 * PI).to(dtype_r) * 1j * (m * a_b).to(dtype_c))
    expo = torch.exp(-PI.to(dtype_r) * ((m - delta_b) ** 2) / lam_b)
    inner = (expo.to(phase_m.dtype) * phase_m).sum(dim=-1)
    return phase_n0 * inner

@torch.jit.script
def _gh_nodes_weights(
    *,
    n: int, 
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.arange(1, n, device=device, dtype=dtype)
    off = torch.sqrt(k * torch.tensor(0.5, dtype=dtype, device=device))
    J = torch.zeros((n, n), device=device, dtype=dtype)
    J = J + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    evals, evecs = torch.linalg.eigh(J)
    w0 = evecs[0, :] ** 2
    PI = torch.tensor(math.pi, dtype=dtype, device=device)
    weights = torch.sqrt(PI) * w0
    return evals, weights

@torch.jit.script
def _frac_unit(
    *,
    x: torch.Tensor,
) -> torch.Tensor:
    return torch.remainder(x + 0.5, 1.0) - 0.5

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
    q_order: int = 7,
) -> torch.Tensor:
    device = z.device
    dtype_r = z.real.dtype
    dtype_c = z.dtype
    B, M, N = vec_d_j.shape

    PI = torch.tensor(math.pi, dtype=dtype_r, device=device)
    tiny = torch.tensor(torch.finfo(dtype_r).tiny, dtype=dtype_r, device=device)

    z_b = z.unsqueeze(1).expand(B, M, N)
    d_b = vec_d.unsqueeze(1).expand(B, M, N)
    dz = z_b - z_j
    dz_re = _frac_unit(x=dz.real)
    dz_im = _frac_unit(x=dz.imag)

    delta_d = delta_vec_d(d_b, vec_d_j)
    dd2 = (delta_d.real**2 + delta_d.imag**2).sum(dim=-1)
    C_dir = torch.exp(-PI * dd2 / sigma_perp).to(dtype_c)

    denom = torch.sqrt((vec_d_j.real**2 + vec_d_j.imag**2).sum(dim=-1).clamp_min(tiny))
    b = (vec_d_j / denom.unsqueeze(-1)).to(dtype_c)
    b_conj = b.conj()

    gh_nodes, gh_w = _gh_nodes_weights(n=q_order, device=device, dtype=dtype_r)
    Q = gh_nodes.numel()
    Xr = gh_nodes.view(Q, 1).expand(Q, Q).reshape(-1)
    Xi = gh_nodes.view(1, Q).expand(Q, Q).reshape(-1)
    Wr2d = (gh_w.view(Q, 1) * gh_w.view(1, Q)).reshape(-1)
    QQ = Xr.numel()

    lam_base = 1.0 / sigma_perp
    lam_for_K = float(lam_base.max().item())
    K_pos = _choose_K_unshifted(
        lam_max=lam_for_K,
        eps_theta=error_budget,
        N=2 * N,
        QQ=QQ,
    )

    dz_re_f = flat3(x=dz_re, B=B, M=M)
    dz_im_f = flat3(x=dz_im, B=B, M=M)
    b_conj_f = flat3(x=b_conj, B=B, M=M)

    sigma_par_f = flat2(x=sigma_par, B=B, M=M)
    sigma_perp_f = flat2(x=sigma_perp, B=B, M=M)
    lam_base_f = flat2(x=lam_base, B=B, M=M)

    mu = sigma_par_f - sigma_perp_f
    mask_pos = (mu >= 0.0)
    mask_neg = ~mask_pos

    Theta_pos_flat = torch.zeros(B * M, dtype=dtype_c, device=device)

    Xr_b = Xr.view(1, QQ, 1).to(dtype_r)
    Xi_b = Xi.view(1, QQ, 1).to(dtype_r)
    Wr_b = Wr2d.view(1, QQ).to(dtype_r)

    pref_global = (sigma_par_f * (sigma_perp_f ** (N - 1))).to(dtype_r).to(dtype_c)

    if mask_pos.any():
        idx = mask_pos
        lam_sel = lam_base_f[idx]
        mu_sel = mu[idx].clamp_min(0.0)
        dz_re_sel = dz_re_f[idx, :]
        dz_im_sel = dz_im_f[idx, :]
        bconj_sel = b_conj_f[idx, :]
        s = torch.sqrt(PI * mu_sel)
        xi = s.view(-1, 1, 1).to(dtype_c) * torch.as_tensor(Xr_b + 1j * Xi_b).to(dtype_c)
        gamma = xi * bconj_sel.to(dtype_c).unsqueeze(1)
        aR_eff = dz_re_sel.unsqueeze(1).expand(-1, QQ, -1) + (gamma.real / PI)
        aI_eff = dz_im_sel.unsqueeze(1).expand(-1, QQ, -1) - (gamma.imag / PI)

        theta_R = _theta1d_phase(
            a=aR_eff,
            lam=lam_sel,
            K=K_pos,
            PI=PI,
            dtype_r=aR_eff.dtype,
            dtype_c=gamma.dtype,
            device=aR_eff.device,
        )
        theta_I = _theta1d_phase(
            a=aI_eff,
            lam=lam_sel,
            K=K_pos,
            PI=PI,
            dtype_r=aI_eff.dtype,
            dtype_c=gamma.dtype,
            device=aI_eff.device,
        )

        mag_log = torch.log(theta_R.abs().clamp_min(tiny)).sum(dim=-1) + torch.log(theta_I.abs().clamp_min(tiny)).sum(dim=-1)
        ang_sum = torch.angle(theta_R).sum(dim=-1) + torch.angle(theta_I).sum(dim=-1)
        f_l = torch.exp(mag_log).to(dtype_c) * torch.exp(1j * ang_sum)
        acc = (f_l * Wr_b.to(dtype_c)).sum(dim=-1) * (1.0 / math.pi)
        Theta_pos_flat[idx] = pref_global[idx] * acc

    if mask_neg.any():
        idx = mask_neg
        lam_sel = lam_base_f[idx]
        mu_abs = (sigma_perp_f[idx] - sigma_par_f[idx]).clamp_min(0.0)
        dz_re_sel = dz_re_f[idx, :]
        dz_im_sel = dz_im_f[idx, :]
        bconj_sel = b_conj_f[idx, :]
        s = torch.sqrt(PI * mu_abs)
        xi = s.view(-1, 1, 1).to(dtype_c) * torch.as_tensor(Xr_b + 1j * Xi_b).to(dtype_c)
        gamma = xi * bconj_sel.to(dtype_c).unsqueeze(1)
        cR = (gamma.real / PI)
        cI = (-gamma.imag / PI)
        beta_R = lam_sel.view(-1, 1, 1) * cR
        beta_I = lam_sel.view(-1, 1, 1) * cI
        aR = dz_re_sel.unsqueeze(1).expand(-1, QQ, -1)
        aI = dz_im_sel.unsqueeze(1).expand(-1, QQ, -1)

        n0_R = torch.floor(beta_R + 0.5)
        n0_I = torch.floor(beta_I + 0.5)
        delta_R = beta_R - n0_R
        delta_I = beta_I - n0_I
        
        K_neg = _choose_K_shifted(
            N=N,
            QQ=QQ,
            lam_sel=lam_sel,
            delta_I=delta_I,
            delta_R=delta_R,
            eps_theta=error_budget,
            tiny=tiny,
            PI=PI,
            device=device,
            dtype_r=dtype_r,
        )
        theta_R = _theta1d_shifted(
            a=aR,
            beta=beta_R,
            lam=lam_sel,
            K=K_neg,
            PI=PI,
            dtype_r=aR.real.dtype,
            dtype_c=gamma.dtype,
            device=aR.device,
        )
        theta_I = _theta1d_shifted(
            a=aI,
            beta=beta_I,
            lam=lam_sel,
            K=K_neg,
            PI=PI,
            dtype_r=aI.real.dtype,
            dtype_c=gamma.dtype,
            device=aI.device,
        )

        gain_log = (PI * lam_sel.view(-1, 1, 1)) * (cR.pow(2) + cI.pow(2))
        gain_log = gain_log.sum(dim=-1)

        mag_log = torch.log(theta_R.abs().clamp_min(tiny)).sum(dim=-1) + torch.log(theta_I.abs().clamp_min(tiny)).sum(dim=-1)
        mag_log = mag_log + gain_log
        ang_sum = torch.angle(theta_R).sum(dim=-1) + torch.angle(theta_I).sum(dim=-1)
        f_l = torch.exp(mag_log).to(dtype_c) * torch.exp(1j * ang_sum)
        acc = (f_l * Wr_b.to(dtype_c)).sum(dim=-1) * (1.0 / math.pi)
        Theta_pos_flat[idx] = pref_global[idx] * acc

    Theta_pos = Theta_pos_flat.view(B, M)
    eta_j = C_dir * Theta_pos
    w = (alpha_j * eta_j.real).unsqueeze(-1).to(T_hat_j.dtype)
    T = (w * T_hat_j).sum(dim=1)
    return T
