import torch
import torch.nn as nn
import math

from typing import Union, List


class WeightsLib2DDelta(nn.Module):
    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],
        context_length: int,
        context_use_bias: bool = True,
        rank: int = 4,
        initialization_std: float = 1.0e-5,
        repulsion_strength: float = 0.1,
        noise_strength: float = 0.01,
        eps: float = 1.0e-12,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.output_shape = output_shape
        self.context_length = context_length
        self.context_use_bias = context_use_bias
        self.rank = rank
        self.initialization_std = initialization_std
        self.repulsion_strength = repulsion_strength
        self.noise_strength = noise_strength
        self.eps = max(eps, 6.0e-8) if dtype_weights == torch.float16 else eps
        self.dtype_weights = dtype_weights

        self.context_transform = nn.Linear(
            in_features=self.context_length,
            out_features=24 * self.rank + (self.rank * 2) + 2,
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )
        nn.init.xavier_uniform_(self.context_transform.weight)
        self.weights = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [
                        1,
                        self.rank,
                        2,
                        2,
                        *self.output_shape,
                        2,
                    ],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=self.initialization_std,
            ).contiguous()
        )
        pass

    def addmul(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.stack(
            [
                bias[..., 0] + (z1[..., 0] * z2[..., 0] - z1[..., 1] * z2[..., 1]),
                bias[..., 1] + (z1[..., 0] * z2[..., 1] + z1[..., 1] * z2[..., 0]),
            ],
            dim=-1,
        )
        return z

    def lerp(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.stack(
            [
                (1 - c[..., 0]) * z1[..., 0] + c[..., 0] * z2[..., 0] - c[..., 1] * (z2[..., 1] - z1[..., 1]),
                (1 - c[..., 0]) * z1[..., 1] + c[..., 0] * z2[..., 1] + c[..., 1] * (z2[..., 0] - z1[..., 0]),
            ],
            dim=-1,
        )
        return z

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype

        x = x if x.dtype == self.dtype_weights else x.to(self.dtype_weights)

        slice_attention = 12 * self.rank + self.rank + 1
        slice_modulation = 12 * self.rank

        x_transformed = self.context_transform(x)
        x_transformed = x_transformed[::, 0:slice_attention] * torch.tanh(x_transformed[::, slice_attention::])
        mod = x_transformed[::, 0:slice_modulation]
        mod = mod.reshape([mod.shape[0], self.rank, 3, 2, 1, 1, 2]).expand([-1, -1, -1, -1, *self.output_shape, -1])

        shift_lerp = torch.tensor([0.5, 0.0], dtype=x.dtype, device=x.device)
        shift_lerp = shift_lerp.reshape([1, 1, 1, 2])
        shift_scale = torch.tensor([1.0, 0.0], dtype=x.dtype, device=x.device)
        shift_scale = shift_scale.reshape([1, 1, 1, 2])

        param_lerp = mod[::, ::, 0] + shift_lerp
        param_bias = mod[::, ::, 1]
        param_scale = mod[::, ::, 2] + shift_scale

        weights = self.weights.expand([mod.shape[0], -1, -1, -1, -1, -1, -1])
        weights = self.lerp(weights[::, ::, 0], weights[::, ::, 1], param_lerp)
        weights = self.addmul(weights, param_scale, param_bias)

        denom = (weights[::, ::, 1] ** 2).sum(-1).clamp(min=self.eps).sqrt().unsqueeze(-1)
        theta = weights[::, ::, 1] / denom
        weights = (weights[::, ::, 0] * theta).sum(dim=-1)

        # Cosine similarity decorellation.
        weights_flat = weights.view(weights.shape[0], weights.shape[1], -1)
        weights_stable = torch.where(weights_flat.abs() < self.eps, weights_flat.sign() * self.eps, weights_flat)
        weights_flat = weights_flat + (weights_stable - weights_flat).detach()
        weights_flat = torch.nn.functional.normalize(weights_flat, p=2, dim=-1)
        mat_sim = torch.matmul(weights_flat, weights_flat.transpose(1, 2))
        mat_diagonal = torch.eye(mat_sim.shape[1], dtype=torch.bool, device=mat_sim.device)
        mat_sim = mat_sim * (~mat_diagonal)
        repulsion = torch.matmul(mat_sim, weights_flat) / (self.rank - 1)
        weights_flat = weights_flat - repulsion * (self.repulsion_strength / math.sqrt(self.rank))
        weights_stable = torch.where(weights_flat.abs() < self.eps, weights_flat.sign() * self.eps, weights_flat)
        weights_flat = weights_flat + (weights_stable - weights_flat).detach()
        weights_flat = torch.nn.functional.normalize(weights_flat, p=2, dim=-1)
        weights_flat_mean = weights_flat.mean(dim=-1, keepdim=True).detach()
        weights_flat_std = weights_flat.std(dim=-1, keepdim=True).clamp(min=self.eps).detach()
        vector_noise = (torch.randn_like(weights_flat) * weights_flat_std + weights_flat_mean) * self.noise_strength
        weights_flat = weights_flat + vector_noise if self.training else weights_flat
        weights = weights_flat.reshape(weights.shape)

        # Add null-weight for attention drain.
        attention_drain = torch.zeros(
            size=[weights.shape[0], 1, *self.output_shape],
            dtype=self.dtype_weights,
            device=weights.device,
        )
        weights = torch.cat([weights, attention_drain], dim=1)

        # Weight components ranking.
        weight_rank = x_transformed[::, slice_modulation::]
        weight_rank_active = weight_rank[..., :-1]
        weight_rank_drain = weight_rank[..., -1:]
        weight_rank = torch.cat([
            weight_rank_active,
            weight_rank_drain + (1.0 / math.sqrt(self.rank)),
        ], dim=-1)
        weight_rank = torch.nn.functional.normalize(weight_rank, p=2, dim=-1)
        weight_rank = torch.softmax(weight_rank, dim=-1)

        weights = weights * weight_rank.reshape([*weight_rank.shape, *[1]*len(weights.shape[2::])])
        weights = weights.sum(1)

        x = weights if weights.dtype == input_dtype else weights.to(input_dtype)

        return x
