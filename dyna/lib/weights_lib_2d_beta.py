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
            out_features=4,
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )
        self.context_transform_gate = nn.Linear(
            in_features=context_length,
            out_features=4,
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x if x.dtype == self.dtype_weights else x.to(self.dtype_weights)

        x_transformed = self.context_transform_input(x)
        x_gate = self.context_transform_gate(x)
        context = x_transformed * torch.nn.functional.tanh(x_gate) * 2.0
        context = context.reshape([context.shape[0], 1, 1, 4]).repeat([1, *self.output_shape, 1])

        target_shape = [context.shape[0], 1, 1, 1]
        weights_base_a = self.weights_base_a.repeat(target_shape)
        weights_base_b = self.weights_base_b.repeat(target_shape)
        weights_base = torch.lerp(weights_base_a, weights_base_b, context[..., 0:2])

        weights_projection_a = self.weights_projection_a.repeat(target_shape)
        weights_projection_b = self.weights_projection_b.repeat(target_shape)
        weight_projection = torch.lerp(weights_projection_a, weights_projection_b, context[..., 2::])

        denom = torch.sqrt((weight_projection ** 2).sum(-1).add(self.eps)).unsqueeze(-1)
        theta = weight_projection / denom
        weights = (weights_base * theta).sum(dim=-1)

        x = weights if weights.dtype == input_dtype else weights.to(input_dtype)

        return x
