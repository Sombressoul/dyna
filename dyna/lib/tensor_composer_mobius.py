import torch
import torch.nn as nn
import math

from typing import Union, List

import dyna


class TensorComposerMobius(nn.Module):
    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],     # Output spatial size (e.g. [H, W])
        context_length: int,                            # Dimensionality of the input context vector
        context_use_bias: bool = True,                  # Use bias in context projection layer
        n_subspaces: int = 16,                          # Number of independent weight components (subspaces)
        rank_subspace: int = 16,                        # Rank of internal modulation basis (basis vectors per subspace)
        rank_transformations: int = 16,                 # Number of interpolation control vectors (z) per axis (for subspace components)
        asymmetry: float = 1.0e-4,                      # min |z|^2 to avoid /0
        eps: float = 1.0e-12,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        # Transmitted parameters.
        self.output_shape = output_shape
        self.context_length = context_length
        self.context_use_bias = context_use_bias
        self.n_subspaces = n_subspaces
        self.rank_subspace = rank_subspace
        self.rank_transformations = rank_transformations
        self.asymmetry = asymmetry
        self.eps = eps
        self.dtype_weights = dtype_weights

        # Calculated parameters.
        self.latent_dim = int(math.sqrt(self.context_length * self.n_subspaces * 2))

        # Context->Modulations transformation layer.
        self.context_transform = nn.Linear(
            in_features=self.context_length,
            out_features=self.latent_dim * 2 * 2, # Linear/Modulation; Re/Im
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )

        # Initialize mod_i: base modulation filters along height
        # Shape: [1, C, R, H, 2] -> (batch_dim, subspaces, basis_rank, height, complex[2])
        self.space_i = nn.Parameter(
            data=torch.nn.init.uniform_(
                torch.empty(
                    [1, self.n_subspaces, self.rank_subspace, self.output_shape[0], 2],
                    dtype=self.dtype_weights,
                ),
                a=-1.0,
                b=+1.0,
            ),
        )

        # Initialize mod_j: base modulation filters along width
        # Shape: [1, C, R, W, 2] -> (batch_dim, subspaces, basis_rank, width, complex[2])
        self.space_j = nn.Parameter(
            data=torch.nn.init.uniform_(
                torch.empty(
                    [1, self.n_subspaces, self.rank_subspace, self.output_shape[1], 2],
                    dtype=self.dtype_weights,
                ),
                a=-1.0,
                b=+1.0,
            ),
        )

        # Inversions filter for Mobius-like transformation
        self.inversions = nn.Parameter(
            data=torch.nn.init.uniform_(
                torch.empty(
                    [1, self.n_subspaces, 1, 1, 2],
                    dtype=self.dtype_weights,
                ),
                a=-1.0,
                b=+1.0,
            ),
        )

        # -------------------------------------------------------------
        # Transformation decoder
        # Direct Tucker G kernel (identity-based, no projection)
        # -------------------------------------------------------------
        T = 2 + 2 * self.rank_transformations
        self.direct_G_Re = nn.Parameter(
            torch.empty([T, self.n_subspaces, self.latent_dim, self.rank_subspace], dtype=self.dtype_weights)
        )
        self.direct_G_Im = nn.Parameter(
            torch.empty([T, self.n_subspaces, self.latent_dim, self.rank_subspace], dtype=self.dtype_weights)
        )
        torch.nn.init.xavier_uniform_(self.direct_G_Re)
        torch.nn.init.xavier_uniform_(self.direct_G_Im)

        # Trainable weights for resulting weights ranking
        self.w_mix = nn.Parameter(
            data=torch.cat(
                [
                    torch.nn.init.ones_(
                        tensor=torch.empty(
                            [self.latent_dim, self.n_subspaces*self.rank_subspace, 1],
                            dtype=self.dtype_weights,
                        )
                    ),
                    torch.nn.init.zeros_(
                        tensor=torch.empty(
                            [self.latent_dim, self.n_subspaces*self.rank_subspace, 1],
                            dtype=self.dtype_weights,
                        )
                    ),
                ],
                dim=-1,
            )
        )

        # Trainable projection planes.
        with torch.no_grad():
            projection_planes = torch.empty(
                [1, self.n_subspaces, *self.output_shape, 2],
                dtype=torch.float32, # precision compatibility for Dirichlet init
            )
            self.dirichlet_init_(projection_planes[..., 0], 0.5)
            projection_planes[..., 1].normal_(0.0, 0.001)
            projection_planes = projection_planes.to(dtype=self.dtype_weights)

        self.projections = nn.Parameter(data=projection_planes)


    def dirichlet_init_(
        self,
        t: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        t_shape = t.shape
        t_flat = t.reshape([-1, t_shape[-1]])
        alpha = torch.full_like(t_flat, alpha)
        sample = torch.distributions.Dirichlet(alpha).sample()
        t.copy_(sample.reshape(t_shape))
        return t


    def complex_mul(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        re = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        im = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
        return torch.stack([re, im], dim=-1)
    

    def complex_div(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        denom = b[..., 0] ** 2 + b[..., 1] ** 2
        denom = denom.clamp_min(eps)
        re = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]) / denom
        im = (a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1]) / denom
        return torch.stack([re, im], dim=-1)


    def transform_space(
        self,
        space: torch.Tensor,
        shift: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        # -------------------------------------
        # Apply shift and modulation to space
        # -------------------------------------
        #   Dimension legend:
        #       B = batch                               # batch dimension
        #       T = 2 + 2 * rank_transformations        # shifts + z-controls
        #       C = n_subspaces                         # independent weight subspaces
        #       R = rank_subspace                       # independent ranks within subspaces
        #       Rz = rank_transformations               # low-res control points
        #
        # Broadcast shift to space spatial dimension and shift the space (space = [1, C, R, H/W, 2])
        shift = shift.unsqueeze(-2) # [B, C, R, 2] -> [B, C, R, 1, 2]
        space = space.expand(shift.shape[0], -1, -1, -1, -1) # Expand batch dim
        space = space + shift # [B, C, R, H/W, 2]

        # Rearrange z so that Rz is the last dimension (required by interpolate)
        z = z.permute([0, 4, 2, 3, 1]).contiguous() # [B, Rz, C, R, 2] -> [B, 2, C, R, Rz]
        B, _, C, R, Rz = z.shape
        z = z.reshape([B, 2 * C * R, Rz]) # flatten channel dims -> [B, channels, Rz]
        z = nn.functional.interpolate(
            z,
            size=space.shape[-2], # Interpolate to target dimension length
            mode="linear",
            align_corners=False,
        )
        z = z.reshape([B, 2, C, R, space.shape[-2]]) # Restore original channels
        z = z.permute([0, 2, 3, 4, 1]).contiguous() # Restore original axes

        # Complex multiplication between shifted space itself and its modulator
        space = self.complex_mul(space, z) # [B, C, R, H/W, 2]

        return space


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Save current x.dtype for future casting the resulting weights to the same dtype
        input_dtype = x.dtype

        # Cast input to internal weight dtype if needed
        x = x if x.dtype == self.dtype_weights else x.to(self.dtype_weights)

        slice_subspaces = self.latent_dim * 2 # Determine cutoff point for latent modulation

        # Projection of input context into latent modulation space with elementwise gated modulation of linear component
        x_transformed = self.context_transform(x)
        x_transformed = dyna.functional.backward_gradient_normalization(x_transformed)
        x_modulated = x_transformed[::, :slice_subspaces:] * dyna.functional.siglog(x_transformed[::, slice_subspaces::])
        x_modulated = x_modulated.reshape([x_modulated.shape[0], self.latent_dim, 2]) # [B, latent_dim, complex[2]]

        # Bilinear projection: apply latent representation to transformation kernels
        # Backward-normalize weights
        G_Re = dyna.functional.backward_gradient_normalization(self.direct_G_Re)  # [T, C, L, R]
        G_Im = dyna.functional.backward_gradient_normalization(self.direct_G_Im)

        # Complex latetn vector
        x_Re, x_Im = x_modulated[..., 0], x_modulated[..., 1]  # [B, L]

        # Einsum bilinear projection: [B, L] x [T, C, L, R] -> [B, T, C, R]
        Re = torch.einsum("bl,tclr->btcr", x_Re, G_Re) - torch.einsum("bl,tclr->btcr", x_Im, G_Im)
        Im = torch.einsum("bl,tclr->btcr", x_Re, G_Im) + torch.einsum("bl,tclr->btcr", x_Im, G_Re)

        # Expand L axis (copy x over L positions)
        Re = Re.unsqueeze(3).expand(-1, -1, -1, self.latent_dim, -1)  # [B, T, C, L, R]
        Im = Im.unsqueeze(3).expand(-1, -1, -1, self.latent_dim, -1)
        # TODO: Above is the point for further performance gains. Simply copying over L is dumb.

        # Normalize grads on backward pass
        Re = dyna.functional.backward_gradient_normalization(Re)
        Im = dyna.functional.backward_gradient_normalization(Im)

        # Normalize to unit circle direction
        cos_theta = Re / (Re**2 + Im**2 + self.eps).sqrt()
        sin_theta = Im / (Re**2 + Im**2 + self.eps).sqrt()
        mag = (Re ** 2 + Im ** 2 + self.eps).sqrt()

        # Weighted average direction using magnitude as importance weight; 3 - is latent_dim
        cos_weighted = (cos_theta * mag).sum(dim=3) / (mag.sum(dim=3) + self.eps)
        sin_weighted = (sin_theta * mag).sum(dim=3) / (mag.sum(dim=3) + self.eps)
        mag_mean = mag.mean(dim=3)

        # Reconstruct complex output using weighted phase and average amplitude
        Re_out = cos_weighted * mag_mean
        Im_out = sin_weighted * mag_mean

        # Final transformation tensor: [B, T, C, R, 2] -> complex output weights
        transformations = torch.stack([Re_out, Im_out], dim=-1)

        # Shift and modulate space i
        space_i_shift = transformations[::, 0] # [B, C, R, 2]
        space_i_z = transformations[::, 2:2+self.rank_transformations] # [B, Rz, C, R, 2]
        space_i = self.space_i
        space_i = dyna.functional.backward_gradient_normalization(space_i)
        space_i = self.transform_space(space=space_i, shift=space_i_shift, z=space_i_z)

        # Shift and modulate space j
        space_j_shift = transformations[::, 1]
        space_j_z = transformations[::, 2+self.rank_transformations:2+(self.rank_transformations*2)]
        space_j = self.space_j
        space_j = dyna.functional.backward_gradient_normalization(space_j)
        space_j = self.transform_space(space=space_j, shift=space_j_shift, z=space_j_z)

        # Mobius‑inversion binding
        inversions = self.inversions
        inversions = dyna.functional.backward_gradient_normalization(inversions)
        # 1) gamma * i
        gamma_I = self.complex_mul(space_i, inversions) # [B, C, R, H, 2]
        # 2) gamma / j
        gamma = inversions.expand_as(space_j)
        gamma_div_J = self.complex_div(gamma, space_j, self.asymmetry)
        # 3) calculate components weights
        w_mix = self.w_mix
        w_mix = dyna.functional.backward_gradient_normalization(w_mix)
        theta_Re = torch.einsum("bl,lk->bk", x_modulated[..., 0],  w_mix[..., 0]).sub_(
            torch.einsum("bl,lk->bk", x_modulated[..., 1], w_mix[..., 1])
        )
        theta_Im = torch.einsum("bl,lk->bk", x_modulated[..., 0], w_mix[..., 1]).add_(
            torch.einsum("bl,lk->bk", x_modulated[..., 1], w_mix[..., 0])
        )
        beta = torch.softmax((theta_Re**2 + theta_Im**2 + self.eps).sqrt().view([x_modulated.shape[0], self.n_subspaces, self.rank_subspace]), dim=-1)
        beta = beta[..., None, None]
        # 4) reweight gammas and sum over rank_subspaces
        gamma_I = (beta * gamma_I).sum(dim=2)
        gamma_J = (beta * gamma_div_J).sum(dim=2)
        # 5) einsum outer product
        w_Re = torch.einsum('bch,bcw->bchw', gamma_I[...,0], gamma_J[...,0]).sub_(
            torch.einsum('bch,bcw->bchw', gamma_I[...,1], gamma_J[...,1])
        )
        w_Im = torch.einsum('bch,bcw->bchw', gamma_I[...,0], gamma_J[...,1]).add_(
            torch.einsum('bch,bcw->bchw', gamma_I[...,1], gamma_J[...,0])
        )

        # Final projection.
        proj = self.projections
        proj = dyna.functional.backward_gradient_normalization(proj)
        denom = torch.sqrt((proj**2).sum(dim=-1, keepdim=True) + self.eps)
        theta_cos = proj[..., 0:1] / denom
        theta_sin = proj[..., 1:2] / denom
        weights = (w_Re.unsqueeze(-1) * theta_cos + w_Im.unsqueeze(-1) * theta_sin).squeeze(-1)
        weights = weights.sum(dim=1) # Sum over subspaces

        x = weights if weights.dtype == input_dtype else weights.to(input_dtype)

        return x
