import torch
import torch.nn as nn

from typing import Union, List


class WeightsLib2DAlpha(nn.Module):
    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],
        context_length: int,
        context_rank: int = 4,
        context_use_bias: bool = True,
        context_conv_use_bias: bool = True,
        context_dropout_rate: float = 0.0,
        initialization_std: float = 1.0e-3,
        eps: float = 1.0e-12,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.output_shape = output_shape
        self.context_rank = context_rank
        self.context_use_bias = context_use_bias
        self.context_conv_use_bias = context_conv_use_bias
        self.context_dropout_rate = context_dropout_rate
        self.initialization_std = initialization_std
        self.eps = max(eps, 6.0e-8) if dtype_weights == torch.float16 else eps
        self.dtype_weights = dtype_weights

        ff_out_features = (self.output_shape[0] + self.output_shape[1]) * self.context_rank * 4  # 4 - real, imag, proj a, proj b 

        self.context_transform_input = nn.Linear(
            in_features=context_length,
            out_features=ff_out_features,
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )
        self.context_transform_input_gate = nn.Linear(
            in_features=context_length,
            out_features=ff_out_features,
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )
        self.context_transform = nn.Linear(
            in_features=ff_out_features,
            out_features=ff_out_features,
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )
        self.weights_static = nn.Parameter(
            data=torch.cat(
                [
                    torch.cat(
                        [
                            torch.nn.init.normal_(
                                tensor=torch.empty(
                                    [
                                        1, # base real
                                        *self.output_shape,
                                        1, # real
                                    ],
                                    dtype=self.dtype_weights,
                                ),
                                mean=0.0,
                                std=self.initialization_std,
                            ),
                            torch.nn.init.normal_(
                                tensor=torch.empty(
                                    [
                                        1, # base imag
                                        *self.output_shape,
                                        1, # imag
                                    ],
                                    dtype=self.dtype_weights,
                                ),
                                mean=0.0,
                                std=self.initialization_std,
                            ),
                        ],
                        dim=-1
                    ),
                    torch.cat(
                        [
                            torch.nn.init.normal_(
                                tensor=torch.empty(
                                    [
                                        1, # scale real
                                        *self.output_shape,
                                        1, # real
                                    ],
                                    dtype=self.dtype_weights,
                                ),
                                mean=1.0,
                                std=self.initialization_std,
                            ),
                            torch.nn.init.normal_(
                                tensor=torch.empty(
                                    [
                                        1, # scale imag
                                        *self.output_shape,
                                        1, # imag
                                    ],
                                    dtype=self.dtype_weights,
                                ),
                                mean=0.0,
                                std=self.initialization_std,
                            ),
                        ],
                        dim=-1,
                    ),
                ],
                dim=0,
            ).contiguous()
        )
        self.weights_mod = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [
                        4, # 4 - real, imag, proj a, proj b
                        *self.output_shape,
                    ],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=self.initialization_std,
            ).contiguous(),
        )
        self.mod_convolution = nn.Conv2d(
            in_channels=self.context_rank,
            out_channels=self.context_rank,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            bias=self.context_conv_use_bias,
            dtype=self.dtype_weights,
        )
        self.context_transform_input_dropout = nn.Dropout1d(
            p=self.context_dropout_rate,
        )

        pass

    def norm_layer(
        self,
        x: torch.Tensor,
        dim: Union[int, list[int]] = -1,
    ) -> torch.Tensor:
        return (x - x.mean(dim=dim, keepdim=True)) / (x.var(dim=dim, keepdim=True) + self.eps).sqrt()
    
    def norm_polar(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_abs = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + self.eps)
        h = x_abs / (x_abs.max(dim=-1, keepdim=True).values + self.eps)
        x_norm = x / (x_abs.unsqueeze(-1) + self.eps)
        r = h * x_norm[..., 0]
        i = h * x_norm[..., 1]
        x = torch.stack([r, i], dim=-1)
        return x

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

        x_transformed = self.context_transform_input_dropout(x)
        x_transformed = self.context_transform_input(x_transformed)
        x_gate = self.context_transform_input_dropout(x)
        x_gate = self.context_transform_input_gate(x_gate)
        x_stack = torch.stack([x_transformed, x_gate], dim=0)
        x_stack = x_stack.reshape([2, x_stack.shape[1], 4, -1])
        x_stack = self.norm_layer(x_stack, dim=[-1])
        context = x_stack[0] * torch.nn.functional.tanh(x_stack[1])
        context = context.flatten(1)
        context = self.context_transform(context)
        context = context.reshape(
            [
                context.shape[0],
                4,
                self.context_rank,
                self.output_shape[0] + self.output_shape[1],
            ],
        )
        mod = self.norm_layer(context, dim=-1)
        mod = mod.reshape([mod.shape[0], 4, self.context_rank, self.output_shape[0] + self.output_shape[1]])
        mod = torch.einsum(
            "...ri,...rj -> ...rij", 
            mod[..., 0:self.output_shape[0]], 
            mod[..., self.output_shape[0]::],
        )
        mod = torch.nn.functional.tanh(mod)
        mod_real = mod[::, 0, ...]
        mod_real = self.mod_convolution(mod_real)
        mod_real = mod_real.sum(dim=-3, keepdim=True)
        mod_other = mod[::, 1::, ...].sum(dim=-3)
        mod = torch.cat([mod_real, mod_other], dim=1)
        mod = self.weights_mod.unsqueeze(0) * mod

        A = self.weights_static.unsqueeze(0).repeat([mod.shape[0], 1, *[1]*(len(mod.shape)-1)])
        B = mod[::, 0:2:, ...].permute([0, 2, 3, 1])
        mod_scaled = torch.cat(
            tensors=[
                (A[::, 1, ..., 0] * B[..., 0] - A[::, 1, ..., 1] * B[..., 1]).unsqueeze(-1),
                (A[::, 1, ..., 0] * B[..., 1] - A[::, 1, ..., 1] * B[..., 0]).unsqueeze(-1),
            ],
            dim=-1,
        )
        mod_scaled_shapebuf = mod_scaled.shape
        mod_scaled = mod_scaled.reshape([mod_scaled_shapebuf[0], mod_scaled_shapebuf[1] * mod_scaled_shapebuf[2], 2])
        mod_scaled = self.norm_polar(mod_scaled).reshape(mod_scaled_shapebuf)
        mod_magnitude = torch.sqrt(mod_scaled[..., 0] ** 2 + mod_scaled[..., 1] ** 2 + self.eps)
        mod_magnitude = mod_magnitude.mean(dim=[-1, -2]).add(self.eps).sqrt()
        mod_magnitude = mod_magnitude.reshape([mod_magnitude.shape[0], *[1]*len(mod_scaled.shape[1::])])
        mod_scaled = mod_scaled / mod_magnitude
        mod_weights = A[::, 0, ...] + mod_scaled

        mod_proj = mod[::, 2::, ...].permute([0, 2, 3, 1])
        mod_proj_shapebuf = mod_scaled.shape
        mod_proj = mod_proj.reshape([mod_proj_shapebuf[0], mod_proj_shapebuf[1] * mod_proj_shapebuf[2], 2])
        mod_proj = self.norm_polar(mod_proj).reshape(mod_proj_shapebuf)

        proj_den = torch.sqrt(mod_proj[::, ..., 0] ** 2 + mod_proj[::, ..., 1] ** 2 + self.eps)
        theta_cos = mod_proj[::, ..., 0] / proj_den
        theta_sin = mod_proj[::, ..., 1] / proj_den
        weights = mod_weights[::, ..., 0] * theta_cos + mod_weights[::, ..., 1] * theta_sin

        x = weights if weights.dtype == input_dtype else weights.to(input_dtype)

        return x
