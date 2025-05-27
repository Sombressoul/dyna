import torch
import torch.nn as nn

from typing import Union, List


class WeightsLib2DGamma(nn.Module):
    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],
        context_length: int,
        context_use_bias: bool = True,
        rank: int = 4,
        entropy_regularization: bool = False,
        initialization_std: float = 1.0e-3,
        eps: float = 1.0e-12,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.output_shape = output_shape
        self.context_length = context_length
        self.context_use_bias = context_use_bias
        self.rank = rank
        self.entropy_regularization = entropy_regularization
        self.initialization_std = initialization_std
        self.eps = max(eps, 6.0e-8) if dtype_weights == torch.float16 else eps
        self.dtype_weights = dtype_weights

        self.context_transform = nn.Linear(
            in_features=self.context_length,
            out_features=24 * self.rank + (self.rank * 2),
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )
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

    def _log_var(
        self,
        x: torch.Tensor,
        name: str = None,
        is_breakpoint: bool = True,
    ) -> None:
        if name is not None:
            print(f"Param: {name}")
        
        print(f"{x.shape=}")
        print(f"{x.min()=}")
        print(f"{x.max()=}")
        print(f"{x.mean()=}")
        print(f"{x.abs().min()=}")
        print(f"{x.abs().max()=}")
        print(f"{x.abs().mean()=}")
        print(f"{x.std()=}")

        if is_breakpoint:
            exit()
        
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

        x_transformed = self.context_transform(x)
        x_transformed = x_transformed[::, 0:(12 * self.rank + self.rank)] * torch.tanh(x_transformed[::, (12 * self.rank + self.rank)::])
        mod = x_transformed[::, 0:(12 * self.rank)]
        mod = mod.reshape([mod.shape[0], self.rank, 3, 2, 1, 1, 2]).expand([-1, -1, -1, -1, *self.output_shape, -1])

        shift_lerp = torch.tensor([0.5, 0.0], dtype=x.dtype, device=x.device)
        shift_lerp = shift_lerp.reshape([1, 1, 1, 2])
        shift_scale = torch.tensor([1.0, 0.0], dtype=x.dtype, device=x.device)
        shift_scale = shift_scale.reshape([1, 1, 1, 2])

        param_l = mod[::, ::, 0] + shift_lerp
        param_b = mod[::, ::, 1]
        param_s = mod[::, ::, 2] + shift_scale

        weights = self.weights.expand([mod.shape[0], -1, -1, -1, -1, -1, -1])
        weights = self.lerp(weights[::, ::, 0], weights[::, ::, 1], param_l)
        weights = self.addmul(weights, param_s, param_b)

        denom = (weights[::, ::, 1] ** 2).sum(-1).clamp(min=self.eps).sqrt().unsqueeze(-1)
        theta = weights[::, ::, 1] / denom
        weights = (weights[::, ::, 0] * theta).sum(dim=-1)

        weight_rank = x_transformed[::, (12 * self.rank)::]
        
        if self.entropy_regularization:
            probs = weights.abs() / weights.abs().sum(dim=[-2, -1], keepdim=True).clamp(min=self.eps)
            entropy = -(probs * probs.clamp(min=self.eps).log()).sum(dim=[-2, -1])
            entropy = entropy / entropy.mean(dim=-1, keepdim=True).clamp(min=self.eps)
            weight_rank = weight_rank + entropy.detach() # Scale, but do not increase entropy!

        weight_rank = torch.nn.functional.normalize(weight_rank, p=2, dim=-1) # Less peaky softmax -> better generalization.
        weight_rank = torch.softmax(weight_rank, dim=-1)

        weights = weights * weight_rank.reshape([*weight_rank.shape, *[1]*len(weights.shape[2::])])
        weights = weights.sum(1)

        x = weights if weights.dtype == input_dtype else weights.to(input_dtype)

        return x
