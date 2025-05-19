import torch
import torch.nn as nn

from typing import Union, List


class WeightsLib2DAlpha(nn.Module):
    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],
        context_length: int,
        context_rank: int = 4,
        context_use_bias: bool = False,
        eps: float = 1.0e-4,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.output_shape = output_shape
        self.context_rank = context_rank
        self.context_use_bias = context_use_bias
        self.eps = eps
        self.dtype_weights = dtype_weights

        ff_out_features = (self.output_shape[0] + self.output_shape[1]) * self.context_rank * 4  # 4 - real, imag, proj a, proj b 

        self.context_transform = nn.Linear(
            in_features=context_length,
            out_features=ff_out_features,
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )
        self.context_transform_gate = nn.Linear(
            in_features=context_length,
            out_features=ff_out_features,
            bias=self.context_use_bias,
            dtype=self.dtype_weights,
        )
        self.weights_static = nn.Parameter(
            data=torch.cat(
                [
                    torch.cat(
                        [
                            torch.nn.init.kaiming_uniform_(
                                tensor=torch.empty(
                                    [
                                        1, # base real
                                        *self.output_shape,
                                        1, # real
                                    ],
                                    dtype=self.dtype_weights,
                                ),
                            ),
                            torch.nn.init.kaiming_uniform_(
                                tensor=torch.empty(
                                    [
                                        1, # base imag
                                        *self.output_shape,
                                        1, # imag
                                    ],
                                    dtype=self.dtype_weights,
                                ),
                            ),
                        ],
                        dim=-1
                    ),
                    torch.cat(
                        [
                            torch.nn.init.uniform_(
                                tensor=torch.empty(
                                    [
                                        1, # scale real
                                        *self.output_shape,
                                        1, # real
                                    ],
                                    dtype=self.dtype_weights,
                                ),
                                a=+self.eps - 1.0,
                                b=-self.eps + 1.0,
                            ),
                            torch.nn.init.uniform_(
                                tensor=torch.empty(
                                    [
                                        1, # scale imag
                                        *self.output_shape,
                                        1, # imag
                                    ],
                                    dtype=self.dtype_weights,
                                ),
                                a=-self.eps,
                                b=+self.eps,
                            ),
                        ],
                        dim=-1,
                    ),
                ],
                dim=0,
            ).contiguous()
        )
        self.weights_mod = nn.Parameter(
            data=torch.nn.init.xavier_uniform_(
                tensor=torch.empty(
                    [
                        4, # 4 - real, imag, proj a, proj b
                        *self.output_shape,
                    ],
                    dtype=self.dtype_weights,
                ),
            ).contiguous(),
        )
        self.mod_convolution = nn.Conv2d(
            in_channels=self.context_rank,
            out_channels=self.context_rank,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            bias=True,
            dtype=self.dtype_weights,
        )

        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        input_dtype = x.dtype
        x = x if x.dtype == self.dtype_weights else x.to(self.dtype_weights)

        x_transformed = self.context_transform(x)
        x_gate = self.context_transform_gate(x)
        x_gate = x_gate * torch.nn.functional.sigmoid(x_gate)
        mod = x_transformed * x_gate
        mod = mod.reshape([mod.shape[0], 4, -1]) # split into meaningful components
        mod = mod / mod.abs().mean(dim=[-1]).add(self.eps).sqrt().unsqueeze(-1) # total mean sqrt norm
        mod = torch.reshape(
            input=mod,
            shape=[
                mod.shape[0],
                4, # 4 - real, imag, proj a, proj b
                self.context_rank,
                self.output_shape[0] + self.output_shape[1],
            ],
        )
        mod = torch.einsum(
            "...ri,...rj -> ...rij", 
            mod[..., 0:self.output_shape[0]], 
            mod[..., self.output_shape[0]::],
        )
        mod = mod.reshape([mod.shape[0] * 4, self.context_rank, *self.output_shape])
        mod = self.mod_convolution(mod).sum(dim=-3)
        mod = mod.reshape([mod.shape[0] // 4, 4, *self.output_shape])
        mod = self.weights_mod * mod

        A = self.weights_static.unsqueeze(0).repeat([mod.shape[0], 1, *[1]*(len(mod.shape)-1)])
        B = mod[::, 0:2:, ...].permute([0, 2, 3, 1])
        mod_scaled = torch.cat(
            tensors=[
                (A[::, 1, ..., 0] * B[..., 0] - A[::, 1, ..., 1] * B[..., 1]).unsqueeze(-1),
                (A[::, 1, ..., 0] * B[..., 1] - A[::, 1, ..., 1] * B[..., 0]).unsqueeze(-1),
            ],
            dim=-1,
        )
        mod_magnitude = torch.sqrt(mod_scaled[..., 0] ** 2 + mod_scaled[..., 1] ** 2 + self.eps)
        mod_magnitude = mod_magnitude.mean(dim=[-1, -2]).add(self.eps).sqrt()
        mod_magnitude = mod_magnitude.reshape([mod_magnitude.shape[0], *[1]*len(mod_scaled.shape[1::])])
        mod_scaled = mod_scaled / mod_magnitude
        mod_weights = A[::, 0, ...] + mod_scaled
        mod_proj = mod[::, 2::, ...].permute([0, 2, 3, 1])

        proj_den = torch.sqrt(mod_proj[::, ..., 0] ** 2 + mod_proj[::, ..., 1] ** 2 + self.eps)
        theta_cos = mod_proj[::, ..., 0] / proj_den
        theta_sin = mod_proj[::, ..., 1] / proj_den
        weights = (
            mod_weights[::, ..., 0] * theta_cos + mod_weights[::, ..., 1] * theta_sin
        )

        x = weights if weights.dtype == input_dtype else weights.to(input_dtype)

        return x
