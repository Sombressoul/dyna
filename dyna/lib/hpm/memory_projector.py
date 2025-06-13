import torch
from memory_field import MemoryField
from typing import Union

class MemoryProjector:
    def __init__(self, memory: MemoryField, max_steps: int = 128, min_tau_u: float = 1e-4, min_sigma_u: float = 1e-4):
        self.memory = memory
        self.max_steps = max_steps
        self.min_tau_u = min_tau_u
        self.min_sigma_u = min_sigma_u

    def _preprocess_inputs(
        self,
        phi_u: torch.Tensor,
        v_u: torch.Tensor,
        tau_u: torch.Tensor,
        sigma_u: torch.Tensor,
        normalize_v_u: bool,
        normalized_coords: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        assert phi_u.shape[-1] == self.memory.n_dim
        assert v_u.shape == phi_u.shape
        assert tau_u.shape == phi_u.shape[:-1], "tau_u shape mismatch"
        assert sigma_u.shape == phi_u.shape[:-1], "sigma_u shape mismatch"

        original_shape = phi_u.shape[:-1]
        D = self.memory.n_dim

        phi_u = phi_u.reshape(-1, D)
        v_u = v_u.reshape(-1, D)
        tau_u = tau_u.reshape(-1)
        sigma_u = sigma_u.reshape(-1)

        shape_tensor = torch.tensor(self.memory.shape, device=phi_u.device, dtype=torch.float32)

        if normalized_coords:
            if (phi_u < 0).any() or (phi_u > 1).any() or (v_u < -1).any() or (v_u > 1).any():
                raise ValueError("Normalized coordinates must be in [0,1] for phi_u and [-1,1] for v_u")
            phi_u = phi_u * shape_tensor
            v_u = v_u * shape_tensor
        else:
            if (phi_u < 0).any() or (phi_u >= shape_tensor).any():
                raise ValueError("phi_u coordinates out of memory bounds")

        if normalize_v_u:
            v_u_norm = torch.norm(v_u, dim=-1, keepdim=True)
            if (v_u_norm < 1e-6).any():
                raise ValueError("v_u norm too small: likely degenerate direction vector")
            v_u_unit = v_u / v_u_norm
        else:
            v_u_norm = torch.ones((phi_u.shape[0], 1), device=v_u.device)
            v_u_unit = v_u

        tau_u = torch.clamp(tau_u, min=self.min_tau_u)
        sigma_u = torch.clamp(sigma_u, min=self.min_sigma_u)

        return phi_u, v_u_unit, v_u_norm, tau_u, sigma_u, shape_tensor, original_shape

    def _trace(
        self,
        phi_u: torch.Tensor,
        v_u_unit: torch.Tensor,
        v_u_norm: torch.Tensor,
        tau_u: torch.Tensor,
        sigma_u: torch.Tensor,
        shape_tensor: torch.Tensor,
        delta_t: float,
        bidirectional: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D = phi_u.shape
        voxel_size = self.memory.voxel_size

        if bidirectional:
            steps = torch.arange(-self.max_steps, self.max_steps, device=phi_u.device).float() * delta_t
        else:
            steps = torch.arange(self.max_steps, device=phi_u.device).float() * delta_t

        l_u = phi_u[:, None, :] + v_u_unit[:, None, :] * steps[None, :, None] / voxel_size

        valid_mask = ((l_u >= 0.0) & (l_u < shape_tensor)).all(dim=-1, keepdim=True)
        if not valid_mask.any():
            t_shift = torch.zeros((B, steps.numel()), device=phi_u.device)
        else:
            first_valid = valid_mask.float().cumsum(dim=1).eq(1)
            t_shift = steps[None, :].repeat(B, 1)
            t_shift = t_shift - t_shift[first_valid.squeeze(-1)].unsqueeze(1)

        valid_mask &= (~((~valid_mask.squeeze(-1)).float().cumsum(dim=1) > 0)).unsqueeze(-1)

        l_u_flat = l_u.reshape(-1, D)
        l_u_normalized = l_u_flat / shape_tensor
        indices = self.memory.c2i_f(l_u_normalized).view(B, -1)
        values = self.memory.get(l_u_normalized, normalized=True).view(B, -1, self.memory.channels)

        t = torch.abs(t_shift) if bidirectional else t_shift
        t_real = t * v_u_norm * voxel_size
        delta_t_i = delta_t * v_u_norm * voxel_size
        t_real = torch.nan_to_num(t_real, nan=1e6, posinf=1e6, neginf=0.0)

        points_vec = l_u - phi_u[:, None, :]
        proj_len = (points_vec * v_u_unit[:, None, :]).sum(dim=-1, keepdim=True)
        proj = proj_len * v_u_unit[:, None, :]
        d_perp_sq = ((points_vec - proj) ** 2).sum(dim=-1, keepdim=True) * (voxel_size ** 2)
        d_perp_sq = torch.nan_to_num(d_perp_sq, nan=1e6, posinf=1e6, neginf=0.0)

        atten_long = torch.exp(-t_real / tau_u[:, None]).unsqueeze(-1)
        weights = torch.exp(-d_perp_sq / (2.0 * sigma_u[:, None, None] ** 2)) * atten_long * valid_mask * delta_t_i.unsqueeze(-1)
        return values, indices, weights.view(B, -1)

    def project(
        self,
        phi_u: torch.Tensor,
        v_u: torch.Tensor,
        tau_u: torch.Tensor,
        sigma_u: torch.Tensor,
        delta_t: float = 1.0,
        normalize_output: bool = True,
        normalize_v_u: bool = False,
        normalized_coords: bool = False,
        bidirectional: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi_u, v_u_unit, v_u_norm, tau_u, sigma_u, shape_tensor, original_shape = self._preprocess_inputs(
            phi_u, v_u, tau_u, sigma_u, normalize_v_u, normalized_coords
        )

        values, indices, weights = self._trace(
            phi_u, v_u_unit, v_u_norm, tau_u, sigma_u, shape_tensor, delta_t, bidirectional
        )
        mask = weights > 0
        mask = mask.view(*original_shape, -1)

        weighted = values * weights.unsqueeze(-1) * mask.unsqueeze(-1)
        projection = weighted.sum(dim=1)
        if normalize_output:
            projection = projection / (weights.sum(dim=1, keepdim=True) + 1e-8)

        projection = projection.view(*original_shape, -1)
        indices = indices.view(*original_shape, -1)
        weights = weights.view(*original_shape, -1)
        phi_u = phi_u.view(*original_shape, -1)
        v_u_unit = v_u_unit.view(*original_shape, -1)
        tau_u = tau_u.view(*original_shape, -1)
        sigma_u = sigma_u.view(*original_shape, -1)

        return projection, indices.masked_fill(~mask, -1), weights, phi_u, v_u_unit, tau_u, sigma_u
