import torch
import torch.nn as nn

from typing import Union, List

from dyna.functional import siglog

class WeightsLib2DAlpha(nn.Module):
    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],
        context_length: int,
        context_rank: int = 4,
        context_use_bias: bool = False,
        convolution_kernel_size: int = 3,
        convolution_kernel_bias: bool = False,
        eps: float = 1.0e-5,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.output_shape = output_shape
        self.context_rank = context_rank
        self.context_use_bias = context_use_bias
        self.convolution_kernel_size = convolution_kernel_size
        self.convolution_kernel_bias = convolution_kernel_bias
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
        self.context_transform_norm = nn.LayerNorm(
            normalized_shape=[ff_out_features],
            elementwise_affine=True,
            bias=False,
            dtype=self.dtype_weights,
        )
        self.context_transform_mod_norm = nn.LayerNorm(
            normalized_shape=[ff_out_features],
            elementwise_affine=True,
            bias=False,
            dtype=self.dtype_weights,
        )
        self.mod_convolution = nn.Conv2d(
            in_channels=4,
            out_channels=4,
            kernel_size=self.convolution_kernel_size,
            stride=1,
            padding=(self.convolution_kernel_size-1)//2,
            dilation=1,
            bias=self.convolution_kernel_bias,
            padding_mode="replicate",
            dtype=self.dtype_weights,
        )
        self.weights_base = nn.Parameter(
            data=torch.nn.init.xavier_uniform_(
                tensor=torch.empty(
                    [
                        *self.output_shape,
                        2, # real, imag
                    ],
                    dtype=self.dtype_weights,
                ),
            ).contiguous(),
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

        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        input_dtype = x.dtype
        x = x if x.dtype == self.dtype_weights else x.to(self.dtype_weights)

        x_transformed = self.context_transform(x)
        x_transformed = self.context_transform_norm(x_transformed)
        x_gate = self.context_transform_gate(x)
        x_gate = x_gate * torch.nn.functional.sigmoid(x_gate)
        mod = x_transformed * x_gate
        mod = self.context_transform_mod_norm(mod)
        mod = torch.reshape(
            input=mod,
            shape=[
                mod.shape[0],
                4, # 4 - real, imag, proj a, proj b
                self.context_rank,
                self.output_shape[0] + self.output_shape[1],
            ],
        )
        mod = self.weights_mod * torch.einsum(
            "...ri,...rj -> ...ij", 
            mod[..., 0:self.output_shape[0]], 
            mod[..., self.output_shape[0]::],
        )
        mod = self.mod_convolution(mod)

        A = self.weights_base.unsqueeze(0).repeat([mod.shape[0], *[1]*(len(mod.shape)-1)])
        B = mod[::, 0:2:, ...].permute([0, 2, 3, 1])
        mod_weights = torch.cat(
            tensors=[
                (A[..., 0] * B[..., 0] - A[..., 1] * B[..., 1]).unsqueeze(-1),
                (A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0]).unsqueeze(-1),
            ],
            dim=-1,
        )
        mod_proj = mod[::, 2::, ...].permute([0, 2, 3, 1])
        mods = torch.cat(
            [
                mod_weights.unsqueeze(1),
                mod_proj.unsqueeze(1),
            ],
            dim=1,
        )

        proj_den = torch.sqrt(mods[::, 1, ..., 0] ** 2 + mods[::, 1, ..., 1] ** 2 + self.eps)
        theta_cos = mods[::, 1, ..., 0] / proj_den
        theta_sin = mods[::, 1, ..., 1] / proj_den
        weights = (
            mods[::, 0, ..., 0] * theta_cos + mods[::, 0, ..., 1] * theta_sin
        )

        x = weights if weights.dtype == input_dtype else weights.to(input_dtype)

        return x
