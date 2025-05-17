import torch
import torch.nn as nn

from typing import Union, List, Callable

from dyna.functional import siglog
from dyna.module.dynamic_conv2d_alpha import DynamicConv2DAlpha


class Coder2DDynamicAlpha(nn.Module):
    def __init__(
        self,
        # Required:
        context_length: int,
        context_rank: int,
        conv_channels_in: int,
        conv_channels_out: int,
        conv_channels_intermediate: int,
        # Defaults:
        context_use_bias: bool = False,
        conv_kernel_small: Union[int, List[int]] = [3, 3],
        conv_kernel_large: Union[int, List[int]] = [5, 5],
        conv_kernel_refine: Union[int, List[int]] = [3, 3],
        conv_padding_mode: str = "replicate",
        interpolate_scale_factor: float = 1.0,
        interpolate_mode: str = "nearest-exact",
        interpolate_align_corners: bool = False,
        batch_norm_affine: bool = True,
        batch_norm_momentum: float = 1.0e-1,
        # Additional:
        activation_intermediate: Callable = siglog,
        eps: float = 1.0e-5,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        # Variables.
        acceptable_dtypes = [torch.float16, torch.bfloat16, torch.float32]

        # Basic arguments checks.
        assert type(context_length) == int, "context_length should be an integer."
        assert type(context_rank) == int, "context_rank should be an integer."
        assert type(conv_channels_in) == int, "conv_channels_in should be an integer."
        assert type(conv_channels_out) == int, "conv_channels_out should be an integer."
        assert type(conv_channels_intermediate) == int, "conv_channels_intermediate should be an integer."
        assert type(context_use_bias) == bool, "context_use_bias should be a boolean."
        assert type(conv_padding_mode) == str, "conv_padding_mode should be a string."
        assert type(interpolate_scale_factor) == float, "interpolate_scale_factor should be a float."
        assert type(interpolate_mode) == str, "interpolate_mode should be a string."
        assert type(interpolate_align_corners) == bool, "interpolate_align_corners should be a boolean."
        assert type(batch_norm_affine) == bool, "batch_norm_affine should be a boolean."
        assert type(batch_norm_momentum) == float, "batch_norm_momentum should be a float."
        assert callable(activation_intermediate), "activation_intermediate should be a function/callable."
        assert type(eps) == float, "eps should be a float."
        assert dtype_weights in acceptable_dtypes, f"dtype_weights should be one of: {acceptable_dtypes}"

        # Complex arguments check.
        # TODO: check conv_kernel_small
        # TODO: check conv_kernel_large

        # Components:
        # self.block_04_norm_pre = nn.BatchNorm2d(
        #     num_features=self.decoder_channels_conv,
        #     eps=self.eps,
        #     affine=True,
        #     momentum=0.1,
        #     dtype=self.dtype_weights,
        # )
        # self.block_04_norm_post = nn.BatchNorm2d(
        #     num_features=self.decoder_channels_contextual * 2,
        #     eps=self.eps,
        #     affine=True,
        #     momentum=0.1,
        #     dtype=self.dtype_weights,
        # )
        # self.block_04_upsample = torch.nn.Upsample(
        #     scale_factor=2,
        #     mode='nearest',
        # )
        # self.block_04_conv_pre = DynamicConv2DAlpha(
        #     in_channels=self.decoder_channels_conv,
        #     out_channels=self.decoder_channels_contextual,
        #     context_length=self.context_length,
        #     context_rank=self.mod_rank,
        #     context_use_bias=False,
        #     kernel_size=self.kernel_size_c_sml,
        #     stride=[1, 1],
        #     padding=self.padding_size_c_sml,
        #     dilation=[1, 1],
        #     transpose=False,
        #     output_padding=None,
        #     dtype_weights=self.dtype_weights,
        # )
        # self.block_04_conv_post = DynamicConv2DAlpha(
        #     in_channels=self.decoder_channels_conv,
        #     out_channels=self.decoder_channels_contextual,
        #     context_length=self.context_length,
        #     context_rank=self.mod_rank,
        #     context_use_bias=False,
        #     kernel_size=self.kernel_size_c_med,
        #     stride=[1, 1],
        #     padding=self.padding_size_c_med,
        #     dilation=[1, 1],
        #     transpose=False,
        #     output_padding=None,
        #     dtype_weights=self.dtype_weights,
        # )
        # self.block_04_conv_refine = DynamicConv2DAlpha(
        #     in_channels=self.decoder_channels_contextual * 2,
        #     out_channels=self.decoder_channels_conv,
        #     context_length=self.context_length,
        #     context_rank=self.mod_rank,
        #     context_use_bias=False,
        #     kernel_size=self.kernel_size_c_sml, # replace w/ specified refine kernel size
        #     stride=[1, 1],
        #     padding=self.padding_size_c_sml,
        #     dilation=[1, 1],
        #     transpose=False,
        #     output_padding=None,
        #     dtype_weights=self.dtype_weights,
        # )
        
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        
        # TODO: replace torch.nn.Upsample w/ torch.nn.functional.interpolate

        # Forward:
        # x = self.block_04_norm_pre(x)
        # x_pre = F.pad(
        #     input=x,
        #     pad=[1, 1, 1, 1],
        #     mode="replicate",
        # )
        # x_pre = self.block_04_conv_pre(x_pre, ctx)
        # x_pre = self.block_04_upsample(x_pre)
        # x_post = self.block_04_upsample(x)
        # x_post = F.pad(
        #     input=x_post,
        #     pad=[2, 2, 2, 2],
        #     mode="replicate",
        # )
        # x_post = self.block_04_conv_post(x_post, ctx)
        # x_up = torch.cat([x_pre, x_post], dim=1)
        # x_up = activation_fn_x(x_up)
        # x_up = F.pad(
        #     input=x_up,
        #     pad=[1, 1, 1, 1],
        #     mode="replicate",
        # )
        # x_up = self.block_04_norm_post(x_up)
        # x = self.block_04_conv_refine(x_up, ctx)
        # x = activation_fn_x(x)

        raise NotImplementedError
