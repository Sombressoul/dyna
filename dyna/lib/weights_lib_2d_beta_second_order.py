import torch
import torch.nn as nn

from typing import Union, List


class WeightsLib2DBetaSecondOrder(nn.Module):
    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],
        context_length: int,
        context_use_bias: bool = True,
        initialization_std: float = 1.0e-2,
        eps: float = 1.0e-12,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.output_shape = output_shape
        self.context_length = context_length
        self.context_use_bias = context_use_bias
        self.initialization_std = initialization_std
        self.eps = max(eps, 6.0e-8) if dtype_weights == torch.float16 else eps
        self.dtype_weights = dtype_weights

        self.context_transform = nn.Linear(
            in_features=self.context_length,
            out_features=480,
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )
        self.weights = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [
                        1,
                        2,
                        2,
                        20,
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
        x_input = x_transformed[::, 0:240, ...]
        x_gate = x_transformed[::, 240::, ...]
        mod = x_input * torch.tanh(x_gate) * 2.0
        mod = mod.reshape([mod.shape[0], 3, 2, 20, 1, 1, 2]).expand([-1, -1, -1, -1, *self.output_shape, -1])

        shift_lerp = torch.tensor([0.5, 0.0], dtype=x.dtype, device=x.device)
        shift_lerp = shift_lerp.reshape([1, 1, 1, 2])
        shift_scale = torch.tensor([1.0, 0.0], dtype=x.dtype, device=x.device)
        shift_scale = shift_scale.reshape([1, 1, 1, 2])

        # Calculate the first order components.
        param_l = mod[::, 0] + shift_lerp
        param_b = mod[::, 1]
        param_s = mod[::, 2] + shift_scale

        components = self.weights.expand([mod.shape[0], -1, -1, -1, -1, -1, -1])
        components = self.lerp(components[::, 0], components[::, 1], param_l)
        components = self.addmul(components, param_s, param_b)

        denom = (components[::, 0] ** 2).sum(-1).add(self.eps).sqrt().unsqueeze(-1)
        theta = components[::, 0] / denom
        components = (components[::, 1] * theta).sum(dim=-1)

        # Calculate the second order components.
        components = torch.stack([components[::, 0:10, ...], components[::, 10::, ...]], dim=-1)

        so_A = components[::, 0:2, ...]
        so_B = components[::, 2:4, ...]
        so_param_l = components[::, 4:6, ...] + shift_lerp.unsqueeze(0)
        so_param_b = components[::, 6:8, ...]
        so_param_s = components[::, 8::, ...] + shift_scale.unsqueeze(0)

        so_components = self.lerp(so_A, so_B, so_param_l)
        so_components = self.addmul(so_components, so_param_s, so_param_b)

        so_denom = (so_components[::, 0] ** 2).sum(-1).add(self.eps).sqrt().unsqueeze(-1)
        so_theta = so_components[::, 0] / so_denom
        weights = (so_components[::, 1] * so_theta).sum(dim=-1)

        x = weights if weights.dtype == input_dtype else weights.to(input_dtype)

        return x
