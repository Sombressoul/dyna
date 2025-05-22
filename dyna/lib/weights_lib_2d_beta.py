import torch
import torch.nn as nn

from typing import Union, List


class WeightsLib2DBeta(nn.Module):
    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],
        context_length: int,
        context_use_bias: bool = True,
        initialization_std: float = 1.0e-3,
        eps: float = 1.0e-12,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.output_shape = output_shape
        self.context_use_bias = context_use_bias
        self.initialization_std = initialization_std
        self.eps = max(eps, 6.0e-8) if dtype_weights == torch.float16 else eps
        self.dtype_weights = dtype_weights

        self.context_transform_input = nn.Linear(
            in_features=context_length,
            out_features=12,
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )
        self.context_transform_gate = nn.Linear(
            in_features=context_length,
            out_features=12,
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )
        self.weights_base_a = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [
                        1,
                        *self.output_shape,
                        2,
                    ],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=self.initialization_std,
            ).contiguous()
        )
        self.weights_base_b = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [
                        1,
                        *self.output_shape,
                        2,
                    ],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=self.initialization_std,
            ).contiguous()
        )
        self.weights_projection_a = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [
                        1,
                        *self.output_shape,
                        2,
                    ],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=self.initialization_std,
            ).contiguous()
        )
        self.weights_projection_b = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [
                        1,
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
        is_breakpoint: bool = True,
    ) -> None:
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

    def add(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.stack(
            [
                z1[..., 0] + z2[..., 0],
                z1[..., 1] + z2[..., 1],
            ],
            dim=-1,
        )
        return z

    def mul(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.stack(
            [
                z1[..., 0] * z2[..., 0] - z1[..., 1] * z2[..., 1],
                z1[..., 0] * z2[..., 1] + z1[..., 1] * z2[..., 0],
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

        x_transformed = self.context_transform_input(x)
        x_gate = self.context_transform_gate(x)
        mod = x_transformed * torch.nn.functional.tanh(x_gate) * 2.0
        mod = mod.reshape([mod.shape[0], 1, 1, 12]).repeat([1, *self.output_shape, 1])

        scale_shift = torch.tensor([1.0, 0.0], dtype=x.dtype, device=x.device)
        scale_shift = scale_shift.reshape([1, 1, 1, 2])

        weights_l = mod[..., 0:2]
        weights_b = mod[..., 2:4]
        weights_s = scale_shift + mod[..., 4:6]

        proj_l = mod[..., 6:8]
        proj_b = mod[..., 8:10]
        proj_s = scale_shift + mod[..., 10:12]

        target_shape = [mod.shape[0], 1, 1, 1]
        weights_base_a = self.weights_base_a.repeat(target_shape)
        weights_base_b = self.weights_base_b.repeat(target_shape)
        weights_base = torch.lerp(weights_base_a, weights_base_b, weights_l)
        weights_base = self.add(weights_base, weights_b)
        weights_base = self.mul(weights_base, weights_s)

        weights_projection_a = self.weights_projection_a.repeat(target_shape)
        weights_projection_b = self.weights_projection_b.repeat(target_shape)
        weight_projection = torch.lerp(weights_projection_a, weights_projection_b, proj_l)
        weight_projection = self.add(weight_projection, proj_b)
        weight_projection = self.mul(weight_projection, proj_s)

        denom = torch.sqrt((weight_projection ** 2).sum(-1).add(self.eps)).unsqueeze(-1)
        theta = weight_projection / denom
        weights = (weights_base * theta).sum(dim=-1)

        x = weights if weights.dtype == input_dtype else weights.to(input_dtype)

        return x
