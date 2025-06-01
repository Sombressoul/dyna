import torch
import torch.nn as nn
import math

from typing import Union, List

from dyna.functional import siglog, backward_gradient_normalization
from dyna.module.signal_stabilization_compressor import SignalStabilizationCompressor, SignalStabilizationCompressorMode


class WeightsLib2DDelta(nn.Module):
    """
    WeightsLib2DDelta
    -----------------

    A 2D dynamic weight generation module with modulation, decorellation, and adaptive weighting.

    This module uses learned context vectors to generate a bank of rank-weighted spatial filters.
    Core mechanisms include:
    - Context-driven parameterization of weight interpolation, scaling, and bias
    - Complex-valued transformations with learnable modulation
    - Projection onto normalized orientation fields
    - Stabilization via SignalStabilizationCompressor
    - Repulsion and similarity penalty to encourage weight diversity
    - Entropy-influenced softmax weighting of ranks
    - Optional Gaussian noise for weights and rank logits during training

    Inputs:
    - x: Tensor of shape [batch_size, context_length]

    Returns:
    - Tensor of shape [batch_size, *output_shape] representing dynamically constructed weights
    """
    # TODO: experiment with torch.nn.init.orthogonal_() initialization of self.weights
    # TODO: export decorellation\repulsion logic into external modules
    # TODO: try JIT - wrap heavy operations into @torch.jit.script

    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],
        context_length: int,
        context_use_bias: bool = True,
        rank: int = 4,
        initialization_std: float = 1.0e-4,
        weights_repulsion_strength: float = 0.3,
        weights_noise_strength: float = 0.01,
        rank_noise_strength: float = 0.01,
        similarity_penalty_strength: float = 0.1,
        eps: float = 1.0e-12,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.output_shape = output_shape
        self.context_length = context_length
        self.context_use_bias = context_use_bias
        self.rank = rank
        self.initialization_std = initialization_std
        self.weights_repulsion_strength = weights_repulsion_strength
        self.weights_noise_strength = weights_noise_strength
        self.rank_noise_strength = rank_noise_strength
        self.similarity_penalty_strength = similarity_penalty_strength
        self.eps = max(eps, 6.0e-8) if dtype_weights == torch.float16 else eps
        self.dtype_weights = dtype_weights

        self.stabilizer = SignalStabilizationCompressor(
            bgn_input=True,
            bgn_mid=True,
            bgn_output=True,
            mode=SignalStabilizationCompressorMode.GATING,
            trainable=False,
            leak=1.0e-3,
            eps=self.eps,
        )
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
        self.extras = nn.Parameter(
            data=torch.cat(
                [
                    torch.tensor([5.0]), # temperature
                ],
                dim=0,
            )
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
        x_transformed = x_transformed[::, 0:slice_attention] * siglog(x_transformed[::, slice_attention::])
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

        denom = (weights[::, ::, 1] ** 2).sum(-1).add(self.eps).sqrt().unsqueeze(-1)
        theta = weights[::, ::, 1] / denom
        theta = backward_gradient_normalization(theta)
        weights = (weights[::, ::, 0] * theta).sum(dim=-1).contiguous()

        # Weights decorellation.
        weights = self.stabilizer(weights)
        weights_flat = weights.reshape(weights.shape[0], weights.shape[1], -1)
        mat_sim = torch.matmul(weights_flat, weights_flat.transpose(1, 2))
        weights_flat = backward_gradient_normalization(weights_flat)
        mat_diagonal = torch.eye(mat_sim.shape[1], dtype=torch.bool, device=mat_sim.device)
        mat_sim = mat_sim * (~mat_diagonal)
        repulsion = torch.matmul(mat_sim, weights_flat) / max((self.rank - 1), 1.0)
        similarity_penalty = mat_sim.pow(2).mean(dim=[-1], keepdim=True)
        similarity_penalty_grad_delta = similarity_penalty - similarity_penalty.detach()
        weights_flat = weights_flat + similarity_penalty_grad_delta * self.similarity_penalty_strength
        weights_flat = weights_flat - repulsion * (self.weights_repulsion_strength / math.sqrt(self.rank))
        weights_stable = torch.where(weights_flat.abs() < self.eps, weights_flat.sign() * self.eps, weights_flat)
        weights_flat = weights_flat + (weights_stable - weights_flat).detach()
        weights_flat_mean = weights_flat.mean(dim=-1, keepdim=True).detach()
        weights_flat_std = weights_flat.std(dim=-1, keepdim=True).add(self.eps).detach()
        vector_noise = (torch.randn_like(weights_flat) * weights_flat_std + weights_flat_mean) * self.weights_noise_strength
        weights_flat = weights_flat + vector_noise if self.training else weights_flat
        weights = weights_flat.reshape(weights.shape)

        # Add null-weight for attention drain.
        attention_drain = torch.zeros(
            size=[weights.shape[0], 1, *self.output_shape],
            dtype=self.dtype_weights,
            device=weights.device,
        )
        weights = torch.cat([weights, attention_drain], dim=1)

        # Ranks decorellation and components weighting.
        weight_rank = x_transformed[::, slice_modulation::]
        weight_rank_active = weight_rank[..., :-1]
        weight_rank_drain = weight_rank[..., -1:]
        weight_rank = torch.cat([
            weight_rank_active,
            weight_rank_drain + (1.0 / math.sqrt(self.rank)),
        ], dim=-1)
        weight_rank_noise = torch.randn_like(weight_rank) * weight_rank.std(dim=-1, keepdim=True).add(self.eps) * self.rank_noise_strength
        weight_rank_logits = weight_rank + weight_rank_noise
        weight_rank_probs = torch.softmax(weight_rank_logits, dim=-1)
        weight_rank_entropy = -(weight_rank_probs * weight_rank_probs.add(self.eps).log()).sum(dim=-1, keepdim=True)
        weight_rank_grad_delta = weight_rank_entropy - weight_rank_entropy.detach()
        weight_rank_logits = weight_rank_logits + weight_rank_grad_delta
        weight_rank = siglog(weight_rank_logits / self.extras[0])
        weight_rank = torch.softmax(weight_rank, dim=-1)
        weight_rank = backward_gradient_normalization(weight_rank)

        # Weighting.
        weights = weights * weight_rank.reshape([*weight_rank.shape, *[1]*len(weights.shape[2::])])
        weights = weights.sum(1)

        x = weights if weights.dtype == input_dtype else weights.to(input_dtype)

        return x
