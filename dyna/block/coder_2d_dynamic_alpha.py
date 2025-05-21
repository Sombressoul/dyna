import torch
import torch.nn as nn

from typing import Union, List, Callable

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
        context_use_bias: bool = True,
        context_conv_use_bias: bool = True,
        conv_kernel_small: Union[int, List[int]] = [3, 3],
        conv_kernel_large: Union[int, List[int]] = [5, 5],
        conv_kernel_refine: Union[int, List[int]] = [3, 3],
        conv_padding_mode: str = "replicate",
        conv_padding_value: Union[int, float] = 0.0,
        interpolate_scale_factor: float = 1.0,
        interpolate_mode: str = "nearest-exact",
        interpolate_align_corners: Union[bool, None] = None,
        interpolate_antialias: bool = False,
        batch_norm_affine: bool = True,
        batch_norm_momentum: float = 1.0e-1,
        # Additional:
        activation_internal: Callable = torch.tanh,
        eps: float = 1.0e-5,
        dtype_weights: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # Helper variables.
        dtypes = [torch.float16, torch.bfloat16, torch.float32]
        modes_padding = ["constant", "reflect", "replicate", "circular"]
        modes_interpolate = ["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"]
        modes_align_corners = ["linear", "bilinear", "bicubic", "trilinear"]

        # Helper functions.
        kernel_type_check = lambda x: type(x) == int or (all([type(em) == int and em > 1 for em in x]) if type(x) == list and len(x) == 2 else False)
        kernel_val_check = lambda x: x[0] % 2 != 0 and x[1] % 2 != 0
        kernel_format = lambda x: [x, x] if type(x) == int else x
        var_info = lambda x: print(f"Recieved: {type(x)=}, {x=}")

        # Arguments checks.
        assert type(context_length) == int, f"context_length should be an integer. {var_info(context_length)}"
        assert context_length > 0, f"context_length should be positive. {var_info(context_length)}"
        assert type(context_rank) == int, f"context_rank should be an integer. {var_info(context_rank)}"
        assert context_rank > 0, f"context_rank should be positive. {var_info(context_rank)}"
        assert type(conv_channels_in) == int, f"conv_channels_in should be an integer. {var_info(conv_channels_in)}"
        assert conv_channels_in > 0, f"conv_channels_in should be positive. {var_info(conv_channels_in)}"
        assert type(conv_channels_out) == int, f"conv_channels_out should be an integer. {var_info(conv_channels_out)}"
        assert conv_channels_out > 0, f"conv_channels_out should be positive. {var_info(conv_channels_out)}"
        assert type(conv_channels_intermediate) == int, f"conv_channels_intermediate should be an integer. {var_info(conv_channels_intermediate)}"
        assert conv_channels_intermediate > 0, f"conv_channels_intermediate should be positive. {var_info(conv_channels_intermediate)}"
        assert type(context_use_bias) == bool, f"context_use_bias should be a boolean. {var_info(context_use_bias)}"
        assert type(context_conv_use_bias) == bool, f"context_convolution_use_bias should be a boolean. {var_info(context_conv_use_bias)}"
        assert type(conv_padding_mode) == str, f"conv_padding_mode should be a string. {var_info(conv_padding_mode)}"
        assert conv_padding_mode in modes_padding, f"conv_padding_mode should be one of: {modes_padding}. {var_info(conv_padding_mode)}"
        assert type(conv_padding_value) in [int, float], f"conv_padding_value should be an integer or a float. {var_info(conv_padding_mode)}"
        assert type(interpolate_scale_factor) == float, f"interpolate_scale_factor should be a float. {var_info(interpolate_scale_factor)}"
        assert interpolate_scale_factor > 0.0, f"interpolate_scale_factor should be positive. {var_info(interpolate_scale_factor)}"
        assert type(interpolate_mode) == str, f"interpolate_mode should be a string. {var_info(interpolate_mode)}"
        assert interpolate_mode in modes_interpolate, f"interpolate_mode should one of: {modes_interpolate}. {var_info(interpolate_mode)}"
        assert type(interpolate_align_corners) in [bool, type(None)], f"interpolate_align_corners should be a boolean or None. {var_info(interpolate_align_corners)}"
        assert type(interpolate_antialias) == bool, f"interpolate_antialias should be a boolean. {var_info(interpolate_antialias)}"
        assert type(batch_norm_affine) == bool, f"batch_norm_affine should be a boolean. {var_info(batch_norm_affine)}"
        assert type(batch_norm_momentum) == float, f"batch_norm_momentum should be a float. {var_info(batch_norm_momentum)}"
        assert batch_norm_momentum >= 0, f"batch_norm_momentum should be positive or equal to 0.0. {var_info(batch_norm_momentum)}"
        assert callable(activation_internal), f"activation_internal should be a function/callable. {var_info(activation_internal)}"
        assert type(eps) == float, f"eps should be a float. {var_info(eps)}"
        assert dtype_weights in dtypes, f"dtype_weights should be one of: {dtypes}. {var_info(dtype_weights)}"
        assert kernel_type_check(conv_kernel_small), f"conv_kernel_small should be a positive integer > 1 or a list of two integers. {var_info(conv_kernel_small)}"
        assert kernel_type_check(conv_kernel_large), f"conv_kernel_large should be a positive integer > 1 or a list of two integers. {var_info(conv_kernel_large)}"
        assert kernel_type_check(conv_kernel_refine), f"conv_kernel_refine should be a positive integer > 1 or a list of two integers. {var_info(conv_kernel_refine)}"
        assert kernel_val_check(conv_kernel_small), f"conv_kernel_small values should be odd. {var_info(conv_kernel_small)}"
        assert kernel_val_check(conv_kernel_large), f"conv_kernel_large values should be odd. {var_info(conv_kernel_large)}"
        assert kernel_val_check(conv_kernel_refine), f"conv_kernel_refine values should be odd. {var_info(conv_kernel_refine)}"

        if interpolate_align_corners is not None:
            assert interpolate_mode in modes_align_corners, f"To use interpolate_align_corners, an interpolate_mode should be one of {modes_align_corners}. {var_info(modes_align_corners)}"

        # Init instance vars.
        self.context_length = context_length
        self.context_rank = context_rank
        self.conv_channels_in = conv_channels_in
        self.conv_channels_out = conv_channels_out
        self.conv_channels_intermediate = conv_channels_intermediate
        self.context_use_bias = context_use_bias
        self.context_conv_use_bias = context_conv_use_bias
        self.conv_kernel_small = kernel_format(conv_kernel_small)
        self.conv_kernel_large = kernel_format(conv_kernel_large)
        self.conv_kernel_refine = kernel_format(conv_kernel_refine)
        self.conv_padding_mode = conv_padding_mode
        self.conv_padding_value = conv_padding_value
        self.interpolate_scale_factor = interpolate_scale_factor
        self.interpolate_mode = interpolate_mode
        self.interpolate_align_corners = interpolate_align_corners
        self.interpolate_antialias = interpolate_antialias
        self.batch_norm_affine = batch_norm_affine
        self.batch_norm_momentum = batch_norm_momentum
        self.activation_internal = activation_internal
        self.eps = eps
        self.dtype_weights = dtype_weights

        # Init components:
        self.coder_block_norm_pre = nn.BatchNorm2d(
            num_features=self.conv_channels_in,
            eps=self.eps,
            affine=self.batch_norm_affine,
            momentum=self.batch_norm_momentum,
            dtype=self.dtype_weights,
        )
        self.coder_block_norm_post = nn.BatchNorm2d(
            num_features=self.conv_channels_intermediate * 2,
            eps=self.eps,
            affine=self.batch_norm_affine,
            momentum=self.batch_norm_momentum,
            dtype=self.dtype_weights,
        )
        self.coder_block_conv_small = DynamicConv2DAlpha(
            in_channels=self.conv_channels_in,
            out_channels=self.conv_channels_intermediate,
            context_length=self.context_length,
            context_rank=self.context_rank,
            context_use_bias=self.context_use_bias,
            context_conv_use_bias=self.context_conv_use_bias,
            kernel_size=self.conv_kernel_small,
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.coder_block_conv_large = DynamicConv2DAlpha(
            in_channels=self.conv_channels_in,
            out_channels=self.conv_channels_intermediate,
            context_length=self.context_length,
            context_rank=self.context_rank,
            context_use_bias=self.context_use_bias,
            context_conv_use_bias=self.context_conv_use_bias,
            kernel_size=self.conv_kernel_large,
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.coder_block_conv_refine = DynamicConv2DAlpha(
            in_channels=self.conv_channels_intermediate * 2,
            out_channels=self.conv_channels_out,
            context_length=self.context_length,
            context_rank=self.context_rank,
            context_use_bias=self.context_use_bias,
            context_conv_use_bias=self.context_conv_use_bias,
            kernel_size=self.conv_kernel_refine,
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        
        # Variables.
        dtype_source_x = x.dtype
        dtype_source_context = context.dtype
        
        # Helper functions.
        get_padding_by_kernel = lambda x: [*[(x[1] - 1) // 2]*2, *[(x[0] - 1) // 2]*2]
        pad = lambda x, kernel: torch.nn.functional.pad(
            input=x,
            pad=get_padding_by_kernel(kernel),
            mode=self.conv_padding_mode,
            value=self.conv_padding_value,
        )
        interpolate = lambda x: torch.nn.functional.interpolate(
            input=x,
            scale_factor=self.interpolate_scale_factor,
            mode=self.interpolate_mode,
            align_corners=self.interpolate_align_corners,
            recompute_scale_factor=False,
            antialias=self.interpolate_antialias,
        )

        # Prepare.
        x = x.to(self.dtype_weights) if x.dtype != self.dtype_weights else x
        context = context.to(self.dtype_weights) if dtype_source_context != self.dtype_weights else context

        # Forward pass.
        x = self.coder_block_norm_pre(x)
        x_small = x
        x_small = pad(x_small, self.conv_kernel_small)
        x_small = self.coder_block_conv_small(x_small, context)
        x_small = interpolate(x_small)
        x_large = x
        x_large = interpolate(x_large)
        x_large = pad(x_large, self.conv_kernel_large)
        x_large = self.coder_block_conv_large(x_large, context)
        x_refine = torch.cat([x_small, x_large], dim=-3)
        x_refine = self.activation_internal(x_refine)
        x_refine = self.coder_block_norm_post(x_refine)
        x_refine = pad(x_refine, self.conv_kernel_refine)
        x_refine = self.coder_block_conv_refine(x_refine, context)
        x_refine = self.activation_internal(x_refine)
        x = x_refine

        # Revert original dtype.
        x = x.to(dtype_source_x) if x.dtype != dtype_source_x else x
        context = context.to(dtype_source_context) if context.dtype != dtype_source_context else context

        return x
