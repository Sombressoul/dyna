import torch
import torch.nn as nn
import math

from typing import Union, List

import dyna


class WeightsLib2DMobius(nn.Module):
    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],     # Output spatial size (e.g. [H, W])
        context_length: int,                            # Dimensionality of the input context vector
        context_use_bias: bool = True,                  # Use bias in context projection layer
        n_subspaces: int = 16,                          # Number of independent weight components (subspaces)
        rank_subspace: int = 16,                        # Rank of internal modulation basis (basis vectors per subspace)
        rank_transformations: int = 16,                 # Number of interpolation control vectors (z) per axis (for subspace components)
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
            data=torch.nn.init.normal_(
                torch.empty(
                    [1, self.n_subspaces, self.rank_subspace, self.output_shape[0], 2],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=1.0,
            ),
        )

        # Initialize mod_j: base modulation filters along width
        # Shape: [1, C, R, W, 2] -> (batch_dim, subspaces, basis_rank, width, complex[2])
        self.space_j = nn.Parameter(
            data=torch.nn.init.normal_(
                torch.empty(
                    [1, self.n_subspaces, self.rank_subspace, self.output_shape[1], 2],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=1.0,
            ),
        )

        # -------------------------------------------------------------
        # Transformation decoder (factorised bilinear-form, low rank K)  
        # -------------------------------------------------------------
        #   A-factor : [T, C, L, K, 2 C, 2]   – projects latent component lᵢ
        #   B-factor : [T, C, K, L,     2]    – projects back to latent lⱼ
        #
        #   where
        #       T = 2 + 2 * rank_transformations      # shifts + z-controls
        #       C = n_subspaces                       # independent weight subspaces
        #       L = latent_dim                        # width of latent modulation
        #       K = int(math.sqrt(L))                 # chosen low-rank dimension
        #
        #   Both factors keep the complex channel (Re/Im) in the last axis.
        T = 2 + 2 * self.rank_transformations
        K = int(math.sqrt(self.latent_dim))
        self.spaces_transformations_factor_A = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(
                    [T, self.n_subspaces, self.latent_dim, K, 2 * self.n_subspaces, 2],
                    dtype=self.dtype_weights,
                )
            )
        )
        self.spaces_transformations_factor_B = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(
                    [T, self.n_subspaces, K, self.latent_dim, 2 * self.n_subspaces, 2],
                    dtype=self.dtype_weights,
                )
            )
        )

        pass

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
        x_modulated = x_transformed[::, :slice_subspaces:] * dyna.functional.siglog(x_transformed[::, slice_subspaces::])
        x_modulated = x_modulated.reshape([x_modulated.shape[0], self.latent_dim, 2]) # [B, latent_dim, complex[2]]

        # Complex bilinear projection: apply latent representation to transformation kernels
        # Contraction over latent Li
        i_Re = torch.einsum("bl,tclki->btclki", x_modulated[..., 0], self.spaces_transformations_factor_A[..., 0]).sub_(
            torch.einsum("bl,tclki->btclki", x_modulated[..., 1], self.spaces_transformations_factor_A[..., 1])
        ).contiguous()
        i_Im = torch.einsum("bl,tclki->btclki", x_modulated[..., 0], self.spaces_transformations_factor_A[..., 1]).add_(
            torch.einsum("bl,tclki->btclki", x_modulated[..., 1], self.spaces_transformations_factor_A[..., 0])
        ).contiguous()

        # Contraction over latent Lj
        j_Re = torch.einsum("bl,tcklj->btclkj", x_modulated[..., 0], self.spaces_transformations_factor_B[..., 0]).sub_(
            torch.einsum("bl,tcklj->btclkj", x_modulated[..., 1], self.spaces_transformations_factor_B[..., 1])
        ).contiguous()
        j_Im = torch.einsum("bl,tcklj->btclkj", x_modulated[..., 0], self.spaces_transformations_factor_B[..., 1]).add_(
            torch.einsum("bl,tcklj->btclkj", x_modulated[..., 1], self.spaces_transformations_factor_B[..., 0])
        ).contiguous()

        # Complex product, collapse K
        Re = (i_Re * j_Re - i_Im * j_Im).sum(dim=4)  # [B,T,C,L,M]
        Im = (i_Re * j_Im + i_Im * j_Re).sum(dim=4)  # [B,T,C,L,M]

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

        # Final transformation tensor: [B, T, C, 2C, 2] -> complex output weights
        transformations = torch.stack([Re_out, Im_out], dim=-1)


        print(f"{x_modulated.shape=}")
        print(f"{self.spaces_transformations_factor_A.shape=}")
        print(f"{self.spaces_transformations_factor_A.numel()=}")
        print(f"{self.spaces_transformations_factor_B.shape=}")
        print(f"{self.spaces_transformations_factor_B.numel()=}")
        print(f"{i_Re.shape=}")
        print(f"{i_Im.shape=}")
        print(f"{j_Re.shape=}")
        print(f"{j_Im.shape=}")
        print(f"{Re.shape=}")
        print(f"{Im.shape=}")
        print(f"{cos_theta.shape=}")
        print(f"{sin_theta.shape=}")
        print(f"{mag.shape=}")
        print(f"{cos_weighted.shape=}")
        print(f"{sin_weighted.shape=}")
        print(f"{mag_mean.shape=}")
        print(f"{Re_out.shape=}")
        print(f"{Im_out.shape=}")
        print(f"{transformations.shape=}")
        exit()

        # TODO: Apply transformations to space_i and space_j to generate final dynamic weights

        weights = torch.empty_like(x) # TEMP PLACEHOLDER

        x = weights if weights.dtype == input_dtype else weights.to(input_dtype)

        return x
