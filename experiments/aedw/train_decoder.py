import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_warmup as warmup
import kornia
import math
import gc

from PIL import Image
from madgrad import MADGRAD

from typing import Optional, Union, Callable, List

script_dir = os.path.dirname(os.path.abspath(__file__))
evals_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(evals_dir)
sys.path.append(project_dir)

# torch.manual_seed(42)
torch.manual_seed(10056)

from dyna import DynamicConv2D, WeightsLib2D, siglog, siglog_parametric


class DecoderOnlyModel(nn.Module):
    def __init__(
        self,
        data_cache_ctx_len: int = None,
        data_cache_latents_len: int = None,
        data_cache_latents_shape: list[int] = None,
        dropout_rate_latents: float = 0.0,
        dropout_rate_context: float = 0.0,
        noisein_rate_latents: float = 0.0,
        noisein_rate_context: float = 0.0,
        noisein_rate_latents_input: float = 0.0,
        noisein_rate_latents_output: float = 0.0,
        noisein_rate_context_input: float = 0.0,
        noisein_rate_context_output: float = 0.0,
        noiseover_rate_latents: float = 0.0,
        noiseover_rate_context: float = 0.0,
        noiseover_rate_latents_input: float = 0.0,
        noiseover_rate_latents_output: float = 0.0,
        noiseover_rate_context_input: float = 0.0,
        noiseover_rate_context_output: float = 0.0,
        data_cache_ctx_bound: float = 0.01,
        data_cache_latents_bound: float = 0.01,
        context_through: bool = False,
        dtype_weights: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.data_cache_ctx_len = data_cache_ctx_len
        self.data_cache_latents_len = data_cache_latents_len
        self.data_cache_latents_shape = data_cache_latents_shape

        self.dropout_rate_latents = dropout_rate_latents
        self.dropout_rate_context = dropout_rate_context
        self.noisein_rate_latents = noisein_rate_latents
        self.noisein_rate_context = noisein_rate_context
        self.noisein_rate_latents_input = noisein_rate_latents_input
        self.noisein_rate_latents_output = noisein_rate_latents_output
        self.noisein_rate_context_input = noisein_rate_context_input
        self.noisein_rate_context_output = noisein_rate_context_output
        self.noiseover_rate_latents = noiseover_rate_latents
        self.noiseover_rate_context = noiseover_rate_context
        self.noiseover_rate_latents_input = noiseover_rate_latents_input
        self.noiseover_rate_latents_output = noiseover_rate_latents_output
        self.noiseover_rate_context_input = noiseover_rate_context_input
        self.noiseover_rate_context_output = noiseover_rate_context_output

        self.use_bias = False
        self.bias_static = 0.0
        self.context_length = 32
        self.mod_rank = 32
        self.transformations_rank = 32

        self.context_through = context_through

        self.branch_x_convolved_sml = True
        self.branch_x_convolved_med = True
        self.branch_x_convolved_lrg = True

        self.data_cache_ctx_bound = data_cache_ctx_bound
        self.data_cache_latents_bound = data_cache_latents_bound

        self.kernel_size_t = [3, 3]
        self.kernel_size_c_base = [3, 3]
        self.kernel_size_c_sml = [3, 3]
        self.kernel_size_c_med = [5, 5]
        self.kernel_size_c_lrg = [7, 7]
        self.padding_size_upsample = [0, 0, 0, 0]
        self.padding_size_c_base = [0, 0, 0, 0]
        self.padding_size_c_sml = [0, 0, 0, 0]
        self.padding_size_c_med = [0, 0, 0, 0]
        self.padding_size_c_lrg = [0, 0, 0, 0]

        self.padding_process_value = -1.0 * math.e
        self.padding_size_c_sml_process = [0, 0, 0, 0]
        self.padding_size_c_med_process = [1, 1, 1, 1]
        self.padding_size_c_lrg_process = [2, 2, 2, 2]

        self.eps = 1.0e-3
        self.q_levels = 16
        self.q_scale = math.e

        self.decoder_channels_out = 3
        self.decoder_channels_reduced = 16
        self.decoder_channels_up = 32
        self.decoder_channels_conv = 16

        self.dtype_weights = dtype_weights

        self.dropout_latents = nn.Dropout(p=self.dropout_rate_latents)
        self.dropout_context = nn.Dropout(p=self.dropout_rate_context)
        self.upsample_nearest = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample_bilinear = nn.Upsample(scale_factor=2, mode="bilinear")

        # ====> Block Input
        self.block_input_x_linear_a = nn.Linear(
            in_features=self.decoder_channels_reduced,
            out_features=self.decoder_channels_reduced,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_input_x_linear_b = nn.Linear(
            in_features=self.decoder_channels_reduced,
            out_features=self.decoder_channels_reduced,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_input_ctx_linear_a = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_input_ctx_linear_b = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_input_base_norm_context_pre = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=False,
            dtype=self.dtype_weights,
        )
        self.block_input_base_norm_context_post_a = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=False,
            dtype=self.dtype_weights,
        )
        self.block_input_base_norm_context_post_b = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=False,
            dtype=self.dtype_weights,
        )
        self.block_input_base_norm_latents_pre = nn.BatchNorm2d(
            num_features=self.data_cache_latents_shape[0],
            eps=self.eps,
            affine=True,
            momentum=0.1,
            dtype=self.dtype_weights,
        )
        self.block_input_base_norm_latents_post_a = nn.BatchNorm2d(
            num_features=self.data_cache_latents_shape[0],
            eps=self.eps,
            affine=True,
            momentum=0.1,
            dtype=self.dtype_weights,
        )
        self.block_input_base_norm_latents_post_b = nn.BatchNorm2d(
            num_features=self.data_cache_latents_shape[0],
            eps=self.eps,
            affine=True,
            momentum=0.1,
            dtype=self.dtype_weights,
        )

        # ====> Block 05
        self.block_05_conv_upsample = DynamicConv2D(
            in_channels=self.decoder_channels_reduced,
            out_channels=self.decoder_channels_up,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_base,
            stride=[2, 2],
            padding=self.padding_size_upsample,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=[0, 0],
            dtype_weights=self.dtype_weights,
        )
        self.block_05_conv_sml = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_sml,
            stride=[1, 1],
            padding=self.padding_size_c_sml,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_05_conv_med = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_med,
            stride=[1, 1],
            padding=self.padding_size_c_med,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_05_conv_lrg = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_lrg,
            stride=[1, 1],
            padding=self.padding_size_c_lrg,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_05_conv_out = DynamicConv2D(
            in_channels=self.decoder_channels_reduced,
            out_channels=self.decoder_channels_reduced,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_05_wl_lat = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.decoder_channels_conv * 3,
                self.decoder_channels_reduced,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_05_wl_ctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_05_ctx_linear_upsample = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_ctx_linear_wl_lat = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_ctx_linear_conv_out = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_ctx_linear_sml = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_ctx_linear_med = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_ctx_linear_lrg = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_norm_ctx_pre = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_norm_ctx_post = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_norm_conv_upsample = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_up],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_norm_conv_sml = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_norm_conv_med = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_norm_conv_lrg = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_norm_out_pre = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_reduced],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_norm_out_post = nn.BatchNorm2d(
            num_features=self.decoder_channels_reduced,
            eps=self.eps,
            affine=True,
            momentum=0.1,
            dtype=self.dtype_weights,
        )
        self.block_05_wl_x_to_ctx_x = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.decoder_channels_reduced,
                self.decoder_channels_reduced,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_05_wl_x_to_ctx_x_norm = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_reduced],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_05_wl_x_to_ctx_xctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length + self.decoder_channels_reduced,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_05_wl_x_to_ctx_xctx_norm = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )

        # ====> Block 04
        self.block_04_conv_upsample = DynamicConv2D(
            in_channels=self.decoder_channels_reduced,
            out_channels=self.decoder_channels_up,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_base,
            stride=[2, 2],
            padding=self.padding_size_upsample,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=[0, 0],
            dtype_weights=self.dtype_weights,
        )
        self.block_04_conv_sml = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_sml,
            stride=[1, 1],
            padding=self.padding_size_c_sml,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_04_conv_med = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_med,
            stride=[1, 1],
            padding=self.padding_size_c_med,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_04_conv_lrg = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_lrg,
            stride=[1, 1],
            padding=self.padding_size_c_lrg,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_04_conv_out = DynamicConv2D(
            in_channels=self.decoder_channels_reduced,
            out_channels=self.decoder_channels_reduced,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_04_wl_lat = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.decoder_channels_conv * 3,
                self.decoder_channels_reduced,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_04_wl_ctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_04_ctx_linear_upsample = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_ctx_linear_wl_lat = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_ctx_linear_conv_out = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_ctx_linear_sml = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_ctx_linear_med = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_ctx_linear_lrg = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_norm_ctx_pre = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_norm_ctx_post = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_norm_conv_upsample = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_up],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_norm_conv_sml = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_norm_conv_med = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_norm_conv_lrg = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_norm_out_pre = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_reduced],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_norm_out_post = nn.BatchNorm2d(
            num_features=self.decoder_channels_reduced,
            eps=self.eps,
            affine=True,
            momentum=0.1,
            dtype=self.dtype_weights,
        )
        self.block_04_wl_x_to_ctx_x = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.decoder_channels_reduced,
                self.decoder_channels_reduced,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_04_wl_x_to_ctx_x_norm = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_reduced],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_04_wl_x_to_ctx_xctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length + self.decoder_channels_reduced,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_04_wl_x_to_ctx_xctx_norm = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )

        # ====> Block 03
        self.block_03_conv_upsample = DynamicConv2D(
            in_channels=self.decoder_channels_reduced,
            out_channels=self.decoder_channels_up,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_base,
            stride=[2, 2],
            padding=self.padding_size_upsample,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=[0, 0],
            dtype_weights=self.dtype_weights,
        )
        self.block_03_conv_sml = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_sml,
            stride=[1, 1],
            padding=self.padding_size_c_sml,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_03_conv_med = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_med,
            stride=[1, 1],
            padding=self.padding_size_c_med,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_03_conv_lrg = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_lrg,
            stride=[1, 1],
            padding=self.padding_size_c_lrg,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_03_conv_out = DynamicConv2D(
            in_channels=self.decoder_channels_reduced,
            out_channels=self.decoder_channels_reduced,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_03_wl_lat = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.decoder_channels_conv * 3,
                self.decoder_channels_reduced,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_03_wl_ctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_03_ctx_linear_upsample = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_ctx_linear_wl_lat = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_ctx_linear_conv_out = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_ctx_linear_sml = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_ctx_linear_med = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_ctx_linear_lrg = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_norm_ctx_pre = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_norm_ctx_post = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_norm_conv_upsample = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_up],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_norm_conv_sml = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_norm_conv_med = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_norm_conv_lrg = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_norm_out_pre = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_reduced],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_norm_out_post = nn.BatchNorm2d(
            num_features=self.decoder_channels_reduced,
            eps=self.eps,
            affine=True,
            momentum=0.1,
            dtype=self.dtype_weights,
        )
        self.block_03_wl_x_to_ctx_x = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.decoder_channels_reduced,
                self.decoder_channels_reduced,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_03_wl_x_to_ctx_x_norm = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_reduced],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_03_wl_x_to_ctx_xctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length + self.decoder_channels_reduced,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_03_wl_x_to_ctx_xctx_norm = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )

        # ====> Block 02
        self.block_02_conv_upsample = DynamicConv2D(
            in_channels=self.decoder_channels_reduced,
            out_channels=self.decoder_channels_up,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_base,
            stride=[2, 2],
            padding=self.padding_size_upsample,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=[0, 0],
            dtype_weights=self.dtype_weights,
        )
        self.block_02_conv_sml = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_sml,
            stride=[1, 1],
            padding=self.padding_size_c_sml,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_02_conv_med = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_med,
            stride=[1, 1],
            padding=self.padding_size_c_med,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_02_conv_lrg = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_lrg,
            stride=[1, 1],
            padding=self.padding_size_c_lrg,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_02_conv_out = DynamicConv2D(
            in_channels=self.decoder_channels_reduced,
            out_channels=self.decoder_channels_reduced,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_02_wl_lat = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.decoder_channels_conv * 3,
                self.decoder_channels_reduced,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_02_wl_ctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_02_ctx_linear_upsample = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_ctx_linear_wl_lat = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_ctx_linear_conv_out = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_ctx_linear_sml = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_ctx_linear_med = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_ctx_linear_lrg = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_norm_ctx_pre = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_norm_ctx_post = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_norm_conv_upsample = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_up],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_norm_conv_sml = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_norm_conv_med = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_norm_conv_lrg = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_norm_out_pre = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_reduced],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_norm_out_post = nn.BatchNorm2d(
            num_features=self.decoder_channels_reduced,
            eps=self.eps,
            affine=True,
            momentum=0.1,
            dtype=self.dtype_weights,
        )
        self.block_02_wl_x_to_ctx_x = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.decoder_channels_reduced,
                self.decoder_channels_reduced,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_02_wl_x_to_ctx_x_norm = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_reduced],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_02_wl_x_to_ctx_xctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length + self.decoder_channels_reduced,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_02_wl_x_to_ctx_xctx_norm = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )

        # ====> Block 01
        self.block_01_conv_upsample = DynamicConv2D(
            in_channels=self.decoder_channels_reduced,
            out_channels=self.decoder_channels_up,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_base,
            stride=[2, 2],
            padding=self.padding_size_upsample,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=[0, 0],
            dtype_weights=self.dtype_weights,
        )
        self.block_01_conv_sml = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_sml,
            stride=[1, 1],
            padding=self.padding_size_c_sml,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_01_conv_med = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_med,
            stride=[1, 1],
            padding=self.padding_size_c_med,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_01_conv_lrg = DynamicConv2D(
            in_channels=self.decoder_channels_up,
            out_channels=self.decoder_channels_conv,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=self.kernel_size_c_lrg,
            stride=[1, 1],
            padding=self.padding_size_c_lrg,
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_01_conv_out = DynamicConv2D(
            in_channels=self.decoder_channels_reduced,
            out_channels=self.decoder_channels_reduced,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_01_wl_lat = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.decoder_channels_conv * 3,
                self.decoder_channels_reduced,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_01_wl_ctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_01_ctx_linear_upsample = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_ctx_linear_wl_lat = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_ctx_linear_conv_out = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_ctx_linear_sml = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_ctx_linear_med = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_ctx_linear_lrg = nn.Linear(
            in_features=self.context_length,
            out_features=self.context_length,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_norm_ctx_pre = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_norm_ctx_post = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_norm_conv_upsample = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_up],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_norm_conv_sml = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_norm_conv_med = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_norm_conv_lrg = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_conv],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_norm_out_pre = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_reduced],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_norm_out_post = nn.BatchNorm2d(
            num_features=self.decoder_channels_reduced,
            eps=self.eps,
            affine=True,
            momentum=0.1,
            dtype=self.dtype_weights,
        )
        self.block_01_wl_x_to_ctx_x = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.decoder_channels_reduced,
                self.decoder_channels_reduced,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_01_wl_x_to_ctx_x_norm = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_reduced],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_01_wl_x_to_ctx_xctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length + self.decoder_channels_reduced,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_01_wl_x_to_ctx_xctx_norm = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )

        # ====> Block out
        self.block_out_linear = nn.Linear(
            in_features=self.decoder_channels_reduced,
            out_features=self.decoder_channels_reduced,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_out_conv = DynamicConv2D(
            in_channels=self.decoder_channels_reduced,
            out_channels=self.decoder_channels_out,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_out_wl_ctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_out_norm_ctx_pre = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_out_norm_ctx_post = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_out_wl_x_to_ctx_x = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.decoder_channels_reduced,
                self.decoder_channels_reduced,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_out_wl_x_to_ctx_x_norm = nn.LayerNorm(
            normalized_shape=[self.decoder_channels_reduced],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )
        self.block_out_wl_x_to_ctx_xctx = WeightsLib2D(
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            transformations_rank=self.transformations_rank,
            output_shape=[
                self.context_length + self.decoder_channels_reduced,
                self.context_length,
            ],
            dtype_weights=self.dtype_weights,
        )
        self.block_out_wl_x_to_ctx_xctx_norm = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=True,
            dtype=self.dtype_weights,
        )

        # ====> Block data cache
        self.data_cache_ctx = nn.Parameter(
            data=torch.nn.init.uniform_(
                tensor=torch.empty(
                    [self.data_cache_ctx_len, self.context_length],
                    dtype=self.dtype_weights,
                ),
                a=-self.data_cache_ctx_bound,
                b=+self.data_cache_ctx_bound,
            ),
        )
        self.data_cache_latents = nn.Parameter(
            data=torch.nn.init.uniform_(
                tensor=torch.empty(
                    [self.data_cache_latents_len, *self.data_cache_latents_shape],
                    dtype=self.dtype_weights,
                ),
                a=-self.data_cache_latents_bound,
                b=+self.data_cache_latents_bound,
            ),
        )

        # ====> Block siglog parametric params
        self.siglog_params = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [89, 1],
                    dtype=self.dtype_weights,
                ),
                mean=(1.0 / math.e),
                std=1.0e-2,
            ),
        )

        pass

    def quantizer(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            l = self.q_levels
            s = self.q_scale
            shift = 1.0 / (l * 2)
            x_q = x / s
            x_q = (x_q * l).clamp(-l, +l)
            x_q = (x_q // 1.0).to(dtype=x.dtype, device=x.device)
            x_q = (x_q / l) + shift
            x_q = x_q * s
        x = x + (x_q - x).detach()
        return x

    def doubleLogNormAbs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.abs()
        x = x.sub(x.min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1))
        x = x.add(torch.e).log().add(self.eps).log()
        return x

    def doubleLogNormTanhAbs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x * F.tanh(x * 1.0e6)
        x = x.sub(x.min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1))
        x = x.add(torch.e).log().add(self.eps).log()
        return x

    def doubleLogNorm(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_mul = torch.where(x > 0.0, +1.0, -1.0).to(dtype=x.dtype, device=x.device)
        x = x.abs()
        x = x.sub(x.min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1))
        x = x.add(torch.e).log().add(self.eps).log()
        x = x.mul(x_mul)
        return x

    def rescale(
        self,
        x: torch.Tensor,
        min: float = -1.0,
        max: float = +1.0,
    ) -> torch.Tensor:
        assert min < max, f"min: {min}, max: {max}"

        scaling_range = max - min

        x_min = (
            x.flatten(1)
            .min(1)[0]
            .reshape([x.shape[0], *[1 for _ in range(len(x.shape) - 1)]])
        )
        x_max = (
            x.flatten(1)
            .max(1)[0]
            .reshape([x.shape[0], *[1 for _ in range(len(x.shape) - 1)]])
        )

        x = (x - x_min + self.eps) / (x_max - x_min + (self.eps * 2))
        x = x * scaling_range + min

        return x

    def partial_norm(
        self,
        x: torch.Tensor,
        fraction: float = 0.1,
        gamma: torch.Tensor = None,
        beta: torch.Tensor = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        sample_numel = x[0].numel()
        partial_size = int(sample_numel * fraction)
        partial_size = 1 if partial_size == 0 else partial_size
        indices = torch.randint(0, sample_numel, [x.shape[0], partial_size])
        indices = [indices[i].add_(sample_numel * i) for i in range(indices.shape[0])]
        indices = torch.cat(indices, dim=0)
        partial_x = x.reshape([-1])[indices].view([-1, partial_size])
        # It is better to use .view() above, but that may cause memory access errors.
        # I have to climb under the hood to fix this, but I don't want to. \/(o_O)\/
        norm_x = partial_x.norm(p="fro", dim=-1, keepdim=True)
        rms_x = norm_x * partial_size ** (-1.0 / 2)
        rms_x = rms_x.view([-1] + [1] * (len(x.shape) - 1))
        x = x / (rms_x + eps)
        x = x * gamma if gamma is not None else x
        x = x + beta if beta is not None else x
        return x

    def standardize(
        self,
        x: torch.Tensor,
        dims: list[int] = [-1],
        eps: float = 1e-6,
    ) -> torch.Tensor:
        x_mean = x.mean(dim=dims, keepdim=True)
        x_std = x.std(dim=dims, keepdim=True)
        return (x - x_mean) / (x_std + eps)

    def squash(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_sq = x * x + 1.0e-4
        x_abs = x_sq.sqrt()
        x = (x_sq / (1.0 + x_sq)) * (x / x_abs)
        return x

    def test_activation_fn(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_sl = siglog_parametric(x, alpha=1.0 / torch.e)
        x_th_sq = F.tanh(x) ** 2
        x_a = x_th_sq
        x_b = 1.0 - x_th_sq
        return (x_sl * x_a) + (x * x_b)

    def noisein(
        self,
        x: torch.Tensor,
        rate: float,
    ) -> torch.Tensor:
        if rate <= 0.0:
            return x

        x_shape = x.shape
        x = x.flatten(1)
        x_numel = x.shape[1]
        x_noise_numel = int(x_numel * rate)
        x_noise = torch.nn.init.normal_(
            tensor=torch.empty([x.shape[0], x_noise_numel], device=x.device),
            mean=x.mean().item(),
            std=x.std().item(),
        )
        target_indices = torch.randint(0, x_numel, [x.shape[0], x_noise_numel])
        target_indices = target_indices + torch.arange(
            0, math.prod(x.shape), x_numel
        ).unsqueeze(1)
        target_indices = target_indices.flatten(0)
        x = x.flatten(0)
        x_negative = torch.zeros_like(x)
        x_negative[target_indices] = (x[target_indices] * -1.0) + x_noise.flatten(0)
        x = x + x_negative
        x = x.reshape([*x_shape])
        return x

    def noiseover(
        self,
        x: torch.Tensor,
        rate: float,
    ) -> torch.Tensor:
        if rate <= 0.0:
            return x

        x_noise = torch.nn.init.normal_(
            tensor=torch.empty_like(x),
            mean=x.mean().item(),
            std=x.std().item(),
        )
        x = x + (x_noise * rate)

        return x

    def forward_pretrain(
        self,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        # Get base ctx
        ids = ids.to(device=self.data_cache_latents.device, dtype=torch.int32)
        ctx = self.data_cache_ctx[ids]

        # Block 05
        w_b05_wl_ctx = self.block_05_wl_ctx(ctx)
        w_b05_wl_lat = self.block_05_wl_lat(ctx)
        w_b05_upsample = self.block_05_conv_upsample.get_weights(ctx)
        w_b05_sml = self.block_05_conv_sml.get_weights(ctx)
        w_b05_med = self.block_05_conv_med.get_weights(ctx)
        w_b05_lrg = self.block_05_conv_lrg.get_weights(ctx)
        w_b05_out = self.block_05_conv_out.get_weights(ctx)

        # Block 04
        w_b04_wl_ctx = self.block_04_wl_ctx(ctx)
        w_b04_wl_lat = self.block_04_wl_lat(ctx)
        w_b04_upsample = self.block_04_conv_upsample.get_weights(ctx)
        w_b04_sml = self.block_04_conv_sml.get_weights(ctx)
        w_b04_med = self.block_04_conv_med.get_weights(ctx)
        w_b04_lrg = self.block_04_conv_lrg.get_weights(ctx)
        w_b04_out = self.block_04_conv_out.get_weights(ctx)

        # Block 03
        w_b03_wl_ctx = self.block_03_wl_ctx(ctx)
        w_b03_wl_lat = self.block_03_wl_lat(ctx)
        w_b03_upsample = self.block_03_conv_upsample.get_weights(ctx)
        w_b03_sml = self.block_03_conv_sml.get_weights(ctx)
        w_b03_med = self.block_03_conv_med.get_weights(ctx)
        w_b03_lrg = self.block_03_conv_lrg.get_weights(ctx)
        w_b03_out = self.block_03_conv_out.get_weights(ctx)

        # Block 02
        w_b02_wl_ctx = self.block_02_wl_ctx(ctx)
        w_b02_wl_lat = self.block_02_wl_lat(ctx)
        w_b02_upsample = self.block_02_conv_upsample.get_weights(ctx)
        w_b02_sml = self.block_02_conv_sml.get_weights(ctx)
        w_b02_med = self.block_02_conv_med.get_weights(ctx)
        w_b02_lrg = self.block_02_conv_lrg.get_weights(ctx)
        w_b02_out = self.block_02_conv_out.get_weights(ctx)

        # Block 01
        w_b01_wl_ctx = self.block_01_wl_ctx(ctx)
        w_b01_wl_lat = self.block_01_wl_lat(ctx)
        w_b01_upsample = self.block_01_conv_upsample.get_weights(ctx)
        w_b01_sml = self.block_01_conv_sml.get_weights(ctx)
        w_b01_med = self.block_01_conv_med.get_weights(ctx)
        w_b01_lrg = self.block_01_conv_lrg.get_weights(ctx)
        w_b01_out = self.block_01_conv_out.get_weights(ctx)

        # Combo blocks
        w_wl_ctx = torch.cat(
            [
                w_b05_wl_ctx.unsqueeze(1),
                w_b04_wl_ctx.unsqueeze(1),
                w_b03_wl_ctx.unsqueeze(1),
                w_b02_wl_ctx.unsqueeze(1),
                w_b01_wl_ctx.unsqueeze(1),
            ],
            dim=1,
        )
        w_wl_lat = torch.cat(
            [
                w_b05_wl_lat.unsqueeze(1),
                w_b04_wl_lat.unsqueeze(1),
                w_b03_wl_lat.unsqueeze(1),
                w_b02_wl_lat.unsqueeze(1),
                w_b01_wl_lat.unsqueeze(1),
            ],
            dim=1,
        )
        w_upsample = torch.cat(
            [
                w_b05_upsample.unsqueeze(1),
                w_b04_upsample.unsqueeze(1),
                w_b03_upsample.unsqueeze(1),
                w_b02_upsample.unsqueeze(1),
                w_b01_upsample.unsqueeze(1),
            ],
            dim=1,
        )
        w_sml = torch.cat(
            [
                w_b05_sml.unsqueeze(1),
                w_b04_sml.unsqueeze(1),
                w_b03_sml.unsqueeze(1),
                w_b02_sml.unsqueeze(1),
                w_b01_sml.unsqueeze(1),
            ],
            dim=1,
        )
        w_med = torch.cat(
            [
                w_b05_med.unsqueeze(1),
                w_b04_med.unsqueeze(1),
                w_b03_med.unsqueeze(1),
                w_b02_med.unsqueeze(1),
                w_b01_med.unsqueeze(1),
            ],
            dim=1,
        )
        w_lrg = torch.cat(
            [
                w_b05_lrg.unsqueeze(1),
                w_b04_lrg.unsqueeze(1),
                w_b03_lrg.unsqueeze(1),
                w_b02_lrg.unsqueeze(1),
                w_b01_lrg.unsqueeze(1),
            ],
            dim=1,
        )
        w_out = torch.cat(
            [
                w_b05_out.unsqueeze(1),
                w_b04_out.unsqueeze(1),
                w_b03_out.unsqueeze(1),
                w_b02_out.unsqueeze(1),
                w_b01_out.unsqueeze(1),
            ],
            dim=1,
        )

        # Out
        w_out_wl_ctx = self.block_out_wl_ctx(ctx).unsqueeze(1)
        w_out_conv = self.block_out_conv.get_weights(ctx).unsqueeze(1)

        return [
            w_wl_ctx,
            w_wl_lat,
            w_upsample,
            w_sml,
            w_med,
            w_lrg,
            w_out,
            w_out_wl_ctx,
            w_out_conv,
        ]

    def forward(
        self,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        ids = ids.to(device=self.data_cache_latents.device, dtype=torch.int32)
        context = self.data_cache_ctx[ids]
        x = self.data_cache_latents[ids]

        # activation_fn = lambda x: siglog_parametric(x, alpha=1.0 / math.e, smooth_grad=True, smoothing=0.1)
        # activation_fn = lambda x: self.squash(siglog_parametric(x, alpha=0.01))
        # activation_fn = lambda x: siglog_parametric(x, alpha=0.01)

        # activation_fn_x = lambda x: siglog_parametric(
        #     x,
        #     alpha=(0.25 / math.e),
        #     smooth_grad=True,
        #     smoothing=0.01,
        # ) # 31 calls
        # activation_fn_ctx = lambda ctx: siglog_parametric(
        #     ctx,
        #     alpha=(0.25 / math.e),
        #     smooth_grad=True,
        #     smoothing=0.01,
        # ) # 36 calls

        self.siglog_call_idx = 0

        def activation_call(
            x: torch.Tensor,
        ) -> torch.Tensor:
            x = siglog_parametric(
                x,
                alpha=self.siglog_params[self.siglog_call_idx, 0],
                smooth_grad=True,
                smoothing=0.01,
            )
            self.siglog_call_idx = self.siglog_call_idx + 1
            return x

        activation_fn_x = lambda x: activation_call(x)
        activation_fn_ctx = lambda ctx: activation_call(ctx)

        # Prepare inputs.
        base_x = self.block_input_base_norm_latents_pre(x)
        base_ctx = self.block_input_base_norm_context_pre(context)

        # Input noise-in/-over
        base_x = self.noisein(base_x, self.noisein_rate_latents_input)
        base_x = self.noiseover(base_x, self.noiseover_rate_latents_input)
        base_ctx = self.noisein(base_ctx, self.noisein_rate_context_input)
        base_ctx = self.noiseover(base_ctx, self.noiseover_rate_context_input)

        # Inter-Block dropout
        base_x = self.dropout_latents(base_x)
        base_ctx = self.dropout_context(base_ctx)

        # Prepare input x and ctx.
        base_x = base_x.permute([0, 2, 3, 1])
        base_x = self.block_input_x_linear_a(base_x)
        base_x = base_x.permute([0, 3, 1, 2])
        base_x = activation_fn_x(base_x)
        base_x = self.block_input_base_norm_latents_post_a(base_x)
        base_x = base_x.permute([0, 2, 3, 1])
        base_x = self.block_input_x_linear_b(base_x)
        base_x = base_x.permute([0, 3, 1, 2])
        base_x = activation_fn_x(base_x)
        base_x = self.block_input_base_norm_latents_post_b(base_x)
        base_ctx = self.block_input_ctx_linear_a(base_ctx)
        base_ctx = activation_fn_ctx(base_ctx)
        base_ctx = self.block_input_base_norm_context_post_a(base_ctx)
        base_ctx = self.block_input_ctx_linear_b(base_ctx)
        base_ctx = activation_fn_ctx(base_ctx)
        base_ctx = self.block_input_base_norm_context_post_b(base_ctx)

        # Set x and ctx.
        x = base_x
        ctx = base_ctx

        # Block 05
        ctx_x = x.flatten(2).permute([0, 2, 1])
        ctx_x = (
            ctx_x.unsqueeze(2) @ self.block_05_wl_x_to_ctx_x(ctx).unsqueeze(1)
        ).squeeze(2)
        ctx_x = activation_fn_ctx(ctx_x).mean(1)
        ctx_x = self.block_05_wl_x_to_ctx_x_norm(ctx_x)
        ctx_x = torch.cat([ctx_x, ctx], dim=1)
        ctx_x = (ctx_x.unsqueeze(1) @ self.block_05_wl_x_to_ctx_xctx(ctx)).squeeze(1)
        ctx_x = activation_fn_ctx(ctx_x)
        ctx_x = self.block_05_wl_x_to_ctx_xctx_norm(ctx_x)
        ctx = ctx + ctx_x
        ctx = self.block_05_norm_ctx_pre(ctx)
        ctx = self.noisein(ctx, self.noisein_rate_context)
        ctx = self.noiseover(ctx, self.noiseover_rate_context)
        ctx = (ctx.unsqueeze(1) @ self.block_05_wl_ctx(ctx)).squeeze(1)
        ctx = activation_fn_ctx(ctx)
        ctx = self.block_05_norm_ctx_post(ctx)
        ctx_upsample = self.block_05_ctx_linear_upsample(ctx)
        ctx_upsample = activation_fn_ctx(ctx_upsample)
        ctx_sml = self.block_05_ctx_linear_sml(ctx)
        ctx_sml = activation_fn_ctx(ctx_sml)
        ctx_med = self.block_05_ctx_linear_med(ctx)
        ctx_med = activation_fn_ctx(ctx_med)
        ctx_lrg = self.block_05_ctx_linear_lrg(ctx)
        ctx_lrg = activation_fn_ctx(ctx_lrg)
        ctx_wl_lat = self.block_05_ctx_linear_wl_lat(ctx)
        ctx_wl_lat = activation_fn_ctx(ctx_wl_lat)
        ctx_conv_out = self.block_05_ctx_linear_conv_out(ctx)
        ctx_conv_out = activation_fn_ctx(ctx_conv_out)
        x_upsampled = self.block_05_conv_upsample(x, ctx_upsample)
        x_upsampled = activation_fn_x(x_upsampled)
        x_upsampled = x_upsampled.permute([0, 2, 3, 1])
        x_upsampled = self.block_05_norm_conv_upsample(x_upsampled)
        x_upsampled = x_upsampled.permute([0, 3, 1, 2])
        x_convolved_sml = x_upsampled
        x_convolved_sml = self.dropout_latents(x_convolved_sml)
        x_convolved_sml = self.noisein(x_convolved_sml, self.noisein_rate_latents)
        x_convolved_sml = self.noiseover(x_convolved_sml, self.noiseover_rate_latents)
        x_convolved_sml = self.block_05_conv_sml(x_convolved_sml, ctx_sml)
        x_convolved_sml = activation_fn_x(x_convolved_sml)
        x_convolved_sml = x_convolved_sml.permute([0, 2, 3, 1])
        x_convolved_sml = self.block_05_norm_conv_sml(x_convolved_sml)
        x_convolved_sml = x_convolved_sml.permute([0, 3, 1, 2])
        x_convolved_sml = F.pad(
            input=x_convolved_sml,
            pad=self.padding_size_c_sml_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved_med = x_upsampled
        x_convolved_med = self.dropout_latents(x_convolved_med)
        x_convolved_med = self.noisein(x_convolved_med, self.noisein_rate_latents)
        x_convolved_med = self.noiseover(x_convolved_med, self.noiseover_rate_latents)
        x_convolved_med = self.block_05_conv_med(x_convolved_med, ctx_med)
        x_convolved_med = activation_fn_x(x_convolved_med)
        x_convolved_med = x_convolved_med.permute([0, 2, 3, 1])
        x_convolved_med = self.block_05_norm_conv_med(x_convolved_med)
        x_convolved_med = x_convolved_med.permute([0, 3, 1, 2])
        x_convolved_med = F.pad(
            input=x_convolved_med,
            pad=self.padding_size_c_med_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved_lrg = x_upsampled
        x_convolved_lrg = self.dropout_latents(x_convolved_lrg)
        x_convolved_lrg = self.noisein(x_convolved_lrg, self.noisein_rate_latents)
        x_convolved_lrg = self.noiseover(x_convolved_lrg, self.noiseover_rate_latents)
        x_convolved_lrg = self.block_05_conv_lrg(x_convolved_lrg, ctx_lrg)
        x_convolved_lrg = activation_fn_x(x_convolved_lrg)
        x_convolved_lrg = x_convolved_lrg.permute([0, 2, 3, 1])
        x_convolved_lrg = self.block_05_norm_conv_lrg(x_convolved_lrg)
        x_convolved_lrg = x_convolved_lrg.permute([0, 3, 1, 2])
        x_convolved_lrg = F.pad(
            input=x_convolved_lrg,
            pad=self.padding_size_c_lrg_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved = torch.cat(
            [
                (
                    x_convolved_sml
                    if self.branch_x_convolved_sml
                    else torch.zeros_like(x_convolved_sml)
                ),
                (
                    x_convolved_med
                    if self.branch_x_convolved_med
                    else torch.zeros_like(x_convolved_med)
                ),
                (
                    x_convolved_lrg
                    if self.branch_x_convolved_lrg
                    else torch.zeros_like(x_convolved_lrg)
                ),
            ],
            dim=1,
        )
        x_lat_w = self.block_05_wl_lat(ctx_wl_lat)
        x = x_convolved.permute([0, 2, 3, 1])
        x = torch.einsum("b...j,bjk->b...k", x, x_lat_w)
        x = activation_fn_x(x)
        x = self.block_05_norm_out_pre(x)
        x = x.permute([0, 3, 1, 2])
        x = self.block_05_conv_out(x, ctx_conv_out)
        x = activation_fn_x(x)
        x = self.block_05_norm_out_post(x)

        # Context-through mode.
        ctx = base_ctx if self.context_through else base_ctx + ctx

        # Inter-Block dropout
        x = self.dropout_latents(x)
        ctx = self.dropout_context(ctx)

        # Block 04
        ctx_x = x.flatten(2).permute([0, 2, 1])
        ctx_x = (
            ctx_x.unsqueeze(2) @ self.block_04_wl_x_to_ctx_x(ctx).unsqueeze(1)
        ).squeeze(2)
        ctx_x = activation_fn_ctx(ctx_x).mean(1)
        ctx_x = self.block_04_wl_x_to_ctx_x_norm(ctx_x)
        ctx_x = torch.cat([ctx_x, ctx], dim=1)
        ctx_x = (ctx_x.unsqueeze(1) @ self.block_04_wl_x_to_ctx_xctx(ctx)).squeeze(1)
        ctx_x = activation_fn_ctx(ctx_x)
        ctx_x = self.block_04_wl_x_to_ctx_xctx_norm(ctx_x)
        ctx = ctx + ctx_x
        ctx = self.block_04_norm_ctx_pre(ctx)
        ctx = self.noisein(ctx, self.noisein_rate_context)
        ctx = self.noiseover(ctx, self.noiseover_rate_context)
        ctx = (ctx.unsqueeze(1) @ self.block_04_wl_ctx(ctx)).squeeze(1)
        ctx = activation_fn_ctx(ctx)
        ctx = self.block_04_norm_ctx_post(ctx)
        ctx_upsample = self.block_04_ctx_linear_upsample(ctx)
        ctx_upsample = activation_fn_ctx(ctx_upsample)
        ctx_sml = self.block_04_ctx_linear_sml(ctx)
        ctx_sml = activation_fn_ctx(ctx_sml)
        ctx_med = self.block_04_ctx_linear_med(ctx)
        ctx_med = activation_fn_ctx(ctx_med)
        ctx_lrg = self.block_04_ctx_linear_lrg(ctx)
        ctx_lrg = activation_fn_ctx(ctx_lrg)
        ctx_wl_lat = self.block_04_ctx_linear_wl_lat(ctx)
        ctx_wl_lat = activation_fn_ctx(ctx_wl_lat)
        ctx_conv_out = self.block_04_ctx_linear_conv_out(ctx)
        ctx_conv_out = activation_fn_ctx(ctx_conv_out)
        x_upsampled = self.block_04_conv_upsample(x, ctx_upsample)
        x_upsampled = activation_fn_x(x_upsampled)
        x_upsampled = x_upsampled.permute([0, 2, 3, 1])
        x_upsampled = self.block_04_norm_conv_upsample(x_upsampled)
        x_upsampled = x_upsampled.permute([0, 3, 1, 2])
        x_convolved_sml = x_upsampled
        x_convolved_sml = self.dropout_latents(x_convolved_sml)
        x_convolved_sml = self.noisein(x_convolved_sml, self.noisein_rate_latents)
        x_convolved_sml = self.noiseover(x_convolved_sml, self.noiseover_rate_latents)
        x_convolved_sml = self.block_04_conv_sml(x_convolved_sml, ctx_sml)
        x_convolved_sml = activation_fn_x(x_convolved_sml)
        x_convolved_sml = x_convolved_sml.permute([0, 2, 3, 1])
        x_convolved_sml = self.block_04_norm_conv_sml(x_convolved_sml)
        x_convolved_sml = x_convolved_sml.permute([0, 3, 1, 2])
        x_convolved_sml = F.pad(
            input=x_convolved_sml,
            pad=self.padding_size_c_sml_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved_med = x_upsampled
        x_convolved_med = self.dropout_latents(x_convolved_med)
        x_convolved_med = self.noisein(x_convolved_med, self.noisein_rate_latents)
        x_convolved_med = self.noiseover(x_convolved_med, self.noiseover_rate_latents)
        x_convolved_med = self.block_04_conv_med(x_convolved_med, ctx_med)
        x_convolved_med = activation_fn_x(x_convolved_med)
        x_convolved_med = x_convolved_med.permute([0, 2, 3, 1])
        x_convolved_med = self.block_04_norm_conv_med(x_convolved_med)
        x_convolved_med = x_convolved_med.permute([0, 3, 1, 2])
        x_convolved_med = F.pad(
            input=x_convolved_med,
            pad=self.padding_size_c_med_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved_lrg = x_upsampled
        x_convolved_lrg = self.dropout_latents(x_convolved_lrg)
        x_convolved_lrg = self.noisein(x_convolved_lrg, self.noisein_rate_latents)
        x_convolved_lrg = self.noiseover(x_convolved_lrg, self.noiseover_rate_latents)
        x_convolved_lrg = self.block_04_conv_lrg(x_convolved_lrg, ctx_lrg)
        x_convolved_lrg = activation_fn_x(x_convolved_lrg)
        x_convolved_lrg = x_convolved_lrg.permute([0, 2, 3, 1])
        x_convolved_lrg = self.block_04_norm_conv_lrg(x_convolved_lrg)
        x_convolved_lrg = x_convolved_lrg.permute([0, 3, 1, 2])
        x_convolved_lrg = F.pad(
            input=x_convolved_lrg,
            pad=self.padding_size_c_lrg_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved = torch.cat(
            [
                (
                    x_convolved_sml
                    if self.branch_x_convolved_sml
                    else torch.zeros_like(x_convolved_sml)
                ),
                (
                    x_convolved_med
                    if self.branch_x_convolved_med
                    else torch.zeros_like(x_convolved_med)
                ),
                (
                    x_convolved_lrg
                    if self.branch_x_convolved_lrg
                    else torch.zeros_like(x_convolved_lrg)
                ),
            ],
            dim=1,
        )
        x_lat_w = self.block_04_wl_lat(ctx_wl_lat)
        x = x_convolved.permute([0, 2, 3, 1])
        x = torch.einsum("b...j,bjk->b...k", x, x_lat_w)
        x = activation_fn_x(x)
        x = self.block_04_norm_out_pre(x)
        x = x.permute([0, 3, 1, 2])
        x = self.block_04_conv_out(x, ctx_conv_out)
        x = activation_fn_x(x)
        x = self.block_04_norm_out_post(x)

        # Context-through mode.
        ctx = base_ctx if self.context_through else base_ctx + ctx

        # Inter-Block dropout
        x = self.dropout_latents(x)
        ctx = self.dropout_context(ctx)

        # Block 03
        ctx_x = x.flatten(2).permute([0, 2, 1])
        ctx_x = (
            ctx_x.unsqueeze(2) @ self.block_03_wl_x_to_ctx_x(ctx).unsqueeze(1)
        ).squeeze(2)
        ctx_x = activation_fn_ctx(ctx_x).mean(1)
        ctx_x = self.block_03_wl_x_to_ctx_x_norm(ctx_x)
        ctx_x = torch.cat([ctx_x, ctx], dim=1)
        ctx_x = (ctx_x.unsqueeze(1) @ self.block_03_wl_x_to_ctx_xctx(ctx)).squeeze(1)
        ctx_x = activation_fn_ctx(ctx_x)
        ctx_x = self.block_03_wl_x_to_ctx_xctx_norm(ctx_x)
        ctx = ctx + ctx_x
        ctx = self.block_03_norm_ctx_pre(ctx)
        ctx = self.noisein(ctx, self.noisein_rate_context)
        ctx = self.noiseover(ctx, self.noiseover_rate_context)
        ctx = (ctx.unsqueeze(1) @ self.block_03_wl_ctx(ctx)).squeeze(1)
        ctx = activation_fn_ctx(ctx)
        ctx = self.block_03_norm_ctx_post(ctx)
        ctx_upsample = self.block_03_ctx_linear_upsample(ctx)
        ctx_upsample = activation_fn_ctx(ctx_upsample)
        ctx_sml = self.block_03_ctx_linear_sml(ctx)
        ctx_sml = activation_fn_ctx(ctx_sml)
        ctx_med = self.block_03_ctx_linear_med(ctx)
        ctx_med = activation_fn_ctx(ctx_med)
        ctx_lrg = self.block_03_ctx_linear_lrg(ctx)
        ctx_lrg = activation_fn_ctx(ctx_lrg)
        ctx_wl_lat = self.block_03_ctx_linear_wl_lat(ctx)
        ctx_wl_lat = activation_fn_ctx(ctx_wl_lat)
        ctx_conv_out = self.block_03_ctx_linear_conv_out(ctx)
        ctx_conv_out = activation_fn_ctx(ctx_conv_out)
        x_upsampled = self.block_03_conv_upsample(x, ctx_upsample)
        x_upsampled = activation_fn_x(x_upsampled)
        x_upsampled = x_upsampled.permute([0, 2, 3, 1])
        x_upsampled = self.block_03_norm_conv_upsample(x_upsampled)
        x_upsampled = x_upsampled.permute([0, 3, 1, 2])
        x_convolved_sml = x_upsampled
        x_convolved_sml = self.dropout_latents(x_convolved_sml)
        x_convolved_sml = self.noisein(x_convolved_sml, self.noisein_rate_latents)
        x_convolved_sml = self.noiseover(x_convolved_sml, self.noiseover_rate_latents)
        x_convolved_sml = self.block_03_conv_sml(x_convolved_sml, ctx_sml)
        x_convolved_sml = activation_fn_x(x_convolved_sml)
        x_convolved_sml = x_convolved_sml.permute([0, 2, 3, 1])
        x_convolved_sml = self.block_03_norm_conv_sml(x_convolved_sml)
        x_convolved_sml = x_convolved_sml.permute([0, 3, 1, 2])
        x_convolved_sml = F.pad(
            input=x_convolved_sml,
            pad=self.padding_size_c_sml_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved_med = x_upsampled
        x_convolved_med = self.dropout_latents(x_convolved_med)
        x_convolved_med = self.noisein(x_convolved_med, self.noisein_rate_latents)
        x_convolved_med = self.noiseover(x_convolved_med, self.noiseover_rate_latents)
        x_convolved_med = self.block_03_conv_med(x_convolved_med, ctx_med)
        x_convolved_med = activation_fn_x(x_convolved_med)
        x_convolved_med = x_convolved_med.permute([0, 2, 3, 1])
        x_convolved_med = self.block_03_norm_conv_med(x_convolved_med)
        x_convolved_med = x_convolved_med.permute([0, 3, 1, 2])
        x_convolved_med = F.pad(
            input=x_convolved_med,
            pad=self.padding_size_c_med_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved_lrg = x_upsampled
        x_convolved_lrg = self.dropout_latents(x_convolved_lrg)
        x_convolved_lrg = self.noisein(x_convolved_lrg, self.noisein_rate_latents)
        x_convolved_lrg = self.noiseover(x_convolved_lrg, self.noiseover_rate_latents)
        x_convolved_lrg = self.block_03_conv_lrg(x_convolved_lrg, ctx_lrg)
        x_convolved_lrg = activation_fn_x(x_convolved_lrg)
        x_convolved_lrg = x_convolved_lrg.permute([0, 2, 3, 1])
        x_convolved_lrg = self.block_03_norm_conv_lrg(x_convolved_lrg)
        x_convolved_lrg = x_convolved_lrg.permute([0, 3, 1, 2])
        x_convolved_lrg = F.pad(
            input=x_convolved_lrg,
            pad=self.padding_size_c_lrg_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved = torch.cat(
            [
                (
                    x_convolved_sml
                    if self.branch_x_convolved_sml
                    else torch.zeros_like(x_convolved_sml)
                ),
                (
                    x_convolved_med
                    if self.branch_x_convolved_med
                    else torch.zeros_like(x_convolved_med)
                ),
                (
                    x_convolved_lrg
                    if self.branch_x_convolved_lrg
                    else torch.zeros_like(x_convolved_lrg)
                ),
            ],
            dim=1,
        )
        x_lat_w = self.block_03_wl_lat(ctx_wl_lat)
        x = x_convolved.permute([0, 2, 3, 1])
        x = torch.einsum("b...j,bjk->b...k", x, x_lat_w)
        x = activation_fn_x(x)
        x = self.block_03_norm_out_pre(x)
        x = x.permute([0, 3, 1, 2])
        x = self.block_03_conv_out(x, ctx_conv_out)
        x = activation_fn_x(x)
        x = self.block_03_norm_out_post(x)

        # Context-through mode.
        ctx = base_ctx if self.context_through else base_ctx + ctx

        # Inter-Block dropout
        x = self.dropout_latents(x)
        ctx = self.dropout_context(ctx)

        # Block 02
        ctx_x = x.flatten(2).permute([0, 2, 1])
        ctx_x = (
            ctx_x.unsqueeze(2) @ self.block_02_wl_x_to_ctx_x(ctx).unsqueeze(1)
        ).squeeze(2)
        ctx_x = activation_fn_ctx(ctx_x).mean(1)
        ctx_x = self.block_02_wl_x_to_ctx_x_norm(ctx_x)
        ctx_x = torch.cat([ctx_x, ctx], dim=1)
        ctx_x = (ctx_x.unsqueeze(1) @ self.block_02_wl_x_to_ctx_xctx(ctx)).squeeze(1)
        ctx_x = activation_fn_ctx(ctx_x)
        ctx_x = self.block_02_wl_x_to_ctx_xctx_norm(ctx_x)
        ctx = ctx + ctx_x
        ctx = self.block_02_norm_ctx_pre(ctx)
        ctx = self.noisein(ctx, self.noisein_rate_context)
        ctx = self.noiseover(ctx, self.noiseover_rate_context)
        ctx = (ctx.unsqueeze(1) @ self.block_02_wl_ctx(ctx)).squeeze(1)
        ctx = activation_fn_ctx(ctx)
        ctx = self.block_02_norm_ctx_post(ctx)
        ctx_upsample = self.block_02_ctx_linear_upsample(ctx)
        ctx_upsample = activation_fn_ctx(ctx_upsample)
        ctx_sml = self.block_02_ctx_linear_sml(ctx)
        ctx_sml = activation_fn_ctx(ctx_sml)
        ctx_med = self.block_02_ctx_linear_med(ctx)
        ctx_med = activation_fn_ctx(ctx_med)
        ctx_lrg = self.block_02_ctx_linear_lrg(ctx)
        ctx_lrg = activation_fn_ctx(ctx_lrg)
        ctx_wl_lat = self.block_02_ctx_linear_wl_lat(ctx)
        ctx_wl_lat = activation_fn_ctx(ctx_wl_lat)
        ctx_conv_out = self.block_02_ctx_linear_conv_out(ctx)
        ctx_conv_out = activation_fn_ctx(ctx_conv_out)
        x_upsampled = self.block_02_conv_upsample(x, ctx_upsample)
        x_upsampled = activation_fn_x(x_upsampled)
        x_upsampled = x_upsampled.permute([0, 2, 3, 1])
        x_upsampled = self.block_02_norm_conv_upsample(x_upsampled)
        x_upsampled = x_upsampled.permute([0, 3, 1, 2])
        x_convolved_sml = x_upsampled
        x_convolved_sml = self.dropout_latents(x_convolved_sml)
        x_convolved_sml = self.noisein(x_convolved_sml, self.noisein_rate_latents)
        x_convolved_sml = self.noiseover(x_convolved_sml, self.noiseover_rate_latents)
        x_convolved_sml = self.block_02_conv_sml(x_convolved_sml, ctx_sml)
        x_convolved_sml = activation_fn_x(x_convolved_sml)
        x_convolved_sml = x_convolved_sml.permute([0, 2, 3, 1])
        x_convolved_sml = self.block_02_norm_conv_sml(x_convolved_sml)
        x_convolved_sml = x_convolved_sml.permute([0, 3, 1, 2])
        x_convolved_sml = F.pad(
            input=x_convolved_sml,
            pad=self.padding_size_c_sml_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved_med = x_upsampled
        x_convolved_med = self.dropout_latents(x_convolved_med)
        x_convolved_med = self.noisein(x_convolved_med, self.noisein_rate_latents)
        x_convolved_med = self.noiseover(x_convolved_med, self.noiseover_rate_latents)
        x_convolved_med = self.block_02_conv_med(x_convolved_med, ctx_med)
        x_convolved_med = activation_fn_x(x_convolved_med)
        x_convolved_med = x_convolved_med.permute([0, 2, 3, 1])
        x_convolved_med = self.block_02_norm_conv_med(x_convolved_med)
        x_convolved_med = x_convolved_med.permute([0, 3, 1, 2])
        x_convolved_med = F.pad(
            input=x_convolved_med,
            pad=self.padding_size_c_med_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved_lrg = x_upsampled
        x_convolved_lrg = self.dropout_latents(x_convolved_lrg)
        x_convolved_lrg = self.noisein(x_convolved_lrg, self.noisein_rate_latents)
        x_convolved_lrg = self.noiseover(x_convolved_lrg, self.noiseover_rate_latents)
        x_convolved_lrg = self.block_02_conv_lrg(x_convolved_lrg, ctx_lrg)
        x_convolved_lrg = activation_fn_x(x_convolved_lrg)
        x_convolved_lrg = x_convolved_lrg.permute([0, 2, 3, 1])
        x_convolved_lrg = self.block_02_norm_conv_lrg(x_convolved_lrg)
        x_convolved_lrg = x_convolved_lrg.permute([0, 3, 1, 2])
        x_convolved_lrg = F.pad(
            input=x_convolved_lrg,
            pad=self.padding_size_c_lrg_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved = torch.cat(
            [
                (
                    x_convolved_sml
                    if self.branch_x_convolved_sml
                    else torch.zeros_like(x_convolved_sml)
                ),
                (
                    x_convolved_med
                    if self.branch_x_convolved_med
                    else torch.zeros_like(x_convolved_med)
                ),
                (
                    x_convolved_lrg
                    if self.branch_x_convolved_lrg
                    else torch.zeros_like(x_convolved_lrg)
                ),
            ],
            dim=1,
        )
        x_lat_w = self.block_02_wl_lat(ctx_wl_lat)
        x = x_convolved.permute([0, 2, 3, 1])
        x = torch.einsum("b...j,bjk->b...k", x, x_lat_w)
        x = activation_fn_x(x)
        x = self.block_02_norm_out_pre(x)
        x = x.permute([0, 3, 1, 2])
        x = self.block_02_conv_out(x, ctx_conv_out)
        x = activation_fn_x(x)
        x = self.block_02_norm_out_post(x)

        # Context-through mode.
        ctx = base_ctx if self.context_through else base_ctx + ctx

        # Inter-Block dropout
        x = self.dropout_latents(x)
        ctx = self.dropout_context(ctx)

        # Block 01
        ctx_x = x.flatten(2).permute([0, 2, 1])
        ctx_x = (
            ctx_x.unsqueeze(2) @ self.block_01_wl_x_to_ctx_x(ctx).unsqueeze(1)
        ).squeeze(2)
        ctx_x = activation_fn_ctx(ctx_x).mean(1)
        ctx_x = self.block_01_wl_x_to_ctx_x_norm(ctx_x)
        ctx_x = torch.cat([ctx_x, ctx], dim=1)
        ctx_x = (ctx_x.unsqueeze(1) @ self.block_01_wl_x_to_ctx_xctx(ctx)).squeeze(1)
        ctx_x = activation_fn_ctx(ctx_x)
        ctx_x = self.block_01_wl_x_to_ctx_xctx_norm(ctx_x)
        ctx = ctx + ctx_x
        ctx = self.block_01_norm_ctx_pre(ctx)
        ctx = self.noisein(ctx, self.noisein_rate_context)
        ctx = self.noiseover(ctx, self.noiseover_rate_context)
        ctx = (ctx.unsqueeze(1) @ self.block_01_wl_ctx(ctx)).squeeze(1)
        ctx = activation_fn_ctx(ctx)
        ctx = self.block_01_norm_ctx_post(ctx)
        ctx_upsample = self.block_01_ctx_linear_upsample(ctx)
        ctx_upsample = activation_fn_ctx(ctx_upsample)
        ctx_sml = self.block_01_ctx_linear_sml(ctx)
        ctx_sml = activation_fn_ctx(ctx_sml)
        ctx_med = self.block_01_ctx_linear_med(ctx)
        ctx_med = activation_fn_ctx(ctx_med)
        ctx_lrg = self.block_01_ctx_linear_lrg(ctx)
        ctx_lrg = activation_fn_ctx(ctx_lrg)
        ctx_wl_lat = self.block_01_ctx_linear_wl_lat(ctx)
        ctx_wl_lat = activation_fn_ctx(ctx_wl_lat)
        ctx_conv_out = self.block_01_ctx_linear_conv_out(ctx)
        ctx_conv_out = activation_fn_ctx(ctx_conv_out)
        x_upsampled = self.block_01_conv_upsample(x, ctx_upsample)
        x_upsampled = activation_fn_x(x_upsampled)
        x_upsampled = x_upsampled.permute([0, 2, 3, 1])
        x_upsampled = self.block_01_norm_conv_upsample(x_upsampled)
        x_upsampled = x_upsampled.permute([0, 3, 1, 2])
        x_convolved_sml = x_upsampled
        x_convolved_sml = self.dropout_latents(x_convolved_sml)
        x_convolved_sml = self.noisein(x_convolved_sml, self.noisein_rate_latents)
        x_convolved_sml = self.noiseover(x_convolved_sml, self.noiseover_rate_latents)
        x_convolved_sml = self.block_01_conv_sml(x_convolved_sml, ctx_sml)
        x_convolved_sml = activation_fn_x(x_convolved_sml)
        x_convolved_sml = x_convolved_sml.permute([0, 2, 3, 1])
        x_convolved_sml = self.block_01_norm_conv_sml(x_convolved_sml)
        x_convolved_sml = x_convolved_sml.permute([0, 3, 1, 2])
        x_convolved_sml = F.pad(
            input=x_convolved_sml,
            pad=self.padding_size_c_sml_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved_med = x_upsampled
        x_convolved_med = self.dropout_latents(x_convolved_med)
        x_convolved_med = self.noisein(x_convolved_med, self.noisein_rate_latents)
        x_convolved_med = self.noiseover(x_convolved_med, self.noiseover_rate_latents)
        x_convolved_med = self.block_01_conv_med(x_convolved_med, ctx_med)
        x_convolved_med = activation_fn_x(x_convolved_med)
        x_convolved_med = x_convolved_med.permute([0, 2, 3, 1])
        x_convolved_med = self.block_01_norm_conv_med(x_convolved_med)
        x_convolved_med = x_convolved_med.permute([0, 3, 1, 2])
        x_convolved_med = F.pad(
            input=x_convolved_med,
            pad=self.padding_size_c_med_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved_lrg = x_upsampled
        x_convolved_lrg = self.dropout_latents(x_convolved_lrg)
        x_convolved_lrg = self.noisein(x_convolved_lrg, self.noisein_rate_latents)
        x_convolved_lrg = self.noiseover(x_convolved_lrg, self.noiseover_rate_latents)
        x_convolved_lrg = self.block_01_conv_lrg(x_convolved_lrg, ctx_lrg)
        x_convolved_lrg = activation_fn_x(x_convolved_lrg)
        x_convolved_lrg = x_convolved_lrg.permute([0, 2, 3, 1])
        x_convolved_lrg = self.block_01_norm_conv_lrg(x_convolved_lrg)
        x_convolved_lrg = x_convolved_lrg.permute([0, 3, 1, 2])
        x_convolved_lrg = F.pad(
            input=x_convolved_lrg,
            pad=self.padding_size_c_lrg_process,
            mode="constant",
            value=self.padding_process_value,
        )
        x_convolved = torch.cat(
            [
                (
                    x_convolved_sml
                    if self.branch_x_convolved_sml
                    else torch.zeros_like(x_convolved_sml)
                ),
                (
                    x_convolved_med
                    if self.branch_x_convolved_med
                    else torch.zeros_like(x_convolved_med)
                ),
                (
                    x_convolved_lrg
                    if self.branch_x_convolved_lrg
                    else torch.zeros_like(x_convolved_lrg)
                ),
            ],
            dim=1,
        )
        x_lat_w = self.block_01_wl_lat(ctx_wl_lat)
        x = x_convolved.permute([0, 2, 3, 1])
        x = torch.einsum("b...j,bjk->b...k", x, x_lat_w)
        x = activation_fn_x(x)
        x = self.block_01_norm_out_pre(x)
        x = x.permute([0, 3, 1, 2])
        x = self.block_01_conv_out(x, ctx_conv_out)
        x = activation_fn_x(x)
        x = self.block_01_norm_out_post(x)

        # Context-through mode.
        ctx = base_ctx if self.context_through else base_ctx + ctx

        # Output noise-in/-over
        x = self.noisein(x, self.noisein_rate_latents_output)
        x = self.noiseover(x, self.noiseover_rate_latents_output)
        ctx = self.noisein(ctx, self.noisein_rate_context_output)
        ctx = self.noiseover(ctx, self.noiseover_rate_context_output)

        # Inter-Block dropout
        x = self.dropout_latents(x)
        ctx = self.dropout_context(ctx)

        # Out
        ctx_x = x.flatten(2).permute([0, 2, 1])
        ctx_x = (
            ctx_x.unsqueeze(2) @ self.block_out_wl_x_to_ctx_x(ctx).unsqueeze(1)
        ).squeeze(2)
        ctx_x = activation_fn_ctx(ctx_x).mean(1)
        ctx_x = self.block_out_wl_x_to_ctx_x_norm(ctx_x)
        ctx_x = torch.cat([ctx_x, ctx], dim=1)
        ctx_x = (ctx_x.unsqueeze(1) @ self.block_out_wl_x_to_ctx_xctx(ctx)).squeeze(1)
        ctx_x = activation_fn_ctx(ctx_x)
        ctx_x = self.block_out_wl_x_to_ctx_xctx_norm(ctx_x)
        ctx = ctx + ctx_x
        ctx = self.block_out_norm_ctx_pre(ctx)
        ctx = self.noisein(ctx, self.noisein_rate_context)
        ctx = self.noiseover(ctx, self.noiseover_rate_context)
        ctx = (ctx.unsqueeze(1) @ self.block_out_wl_ctx(ctx)).squeeze(1)
        ctx = activation_fn_ctx(ctx)
        ctx = self.block_out_norm_ctx_post(ctx)
        x = self.noisein(x, self.noisein_rate_latents)
        x = self.noiseover(x, self.noiseover_rate_latents)
        x = self.block_out_linear(x.permute([0, 2, 3, 1])).permute([0, 3, 1, 2])
        x = activation_fn_x(x)
        x = self.block_out_conv(x, ctx)
        x = F.sigmoid(x)

        return x


def generate_data_from_images(
    shape: list[int],
    images_path_src: str,
    images_sample_count: int,
    starting_from: int = 0,
) -> torch.Tensor:
    data = torch.empty([images_sample_count, 3, *shape], dtype=torch.uint8)
    dir_contents = os.listdir(images_path_src)

    assert len(dir_contents) >= images_sample_count + starting_from, " ".join(
        [
            f"Not enough images in '{images_path_src}'.",
            f"Got: {len(dir_contents)}.",
            f"Need: {images_sample_count + starting_from}.",
        ]
    )

    for i in range(images_sample_count):
        image_name = dir_contents[starting_from + i]
        image_path = os.path.join(images_path_src, image_name)

        try:
            image = Image.open(image_path)
        except ValueError:
            print(f"Failed to load image '{image_path}'.")
            exit()

        image = image.resize([shape[1], shape[0]], Image.LANCZOS)
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0) * 255.0
        image = image.to(dtype=torch.uint8)
        data[i] = image

    return data


def generate_images_from_data(
    data: torch.Tensor,
    images_path_dst: str,
    prefix: str,
) -> None:
    data = data.to(dtype=torch.float16)
    for i in range(data.shape[0]):
        image = data[i].squeeze(0)
        # image = image.clamp(0.0, 1.0)
        image = (image - image.min()) / (image.max() - image.min())
        image = transforms.ToPILImage()(image)
        image_name = f"{prefix}_mat_{i}.png"
        image_path = os.path.join(images_path_dst, image_name)
        image.save(image_path)
    pass


def data_rgb_to_lab(
    data: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        lab = kornia.color.rgb_to_lab(data)
        lab[:, 0, ...].div_(100.0)
        lab[:, 1:, ...].add_(128.0).div_(255.0)
    lab.to(data.device)
    return lab


def data_lab_to_rgb(
    data: torch.Tensor,
) -> torch.Tensor:
    rgb = data
    with torch.no_grad():
        rgb[:, 0, ...].mul_(100.0)
        rgb[:, 1:, ...].mul_(255.0).sub_(128.0)
        rgb = kornia.color.lab_to_rgb(rgb)
    rgb.to(data.device)
    return rgb


def pretrain(
    model: DecoderOnlyModel,
    epochs: int = 1000,
    batch_size: int = 32,
) -> None:
    max_id = model.data_cache_ctx.shape[0]

    x = model.forward_pretrain(torch.tensor([0]))
    targets = []
    for response in x:
        target = torch.nn.init.normal_(
            tensor=torch.empty_like(response).repeat(
                [
                    max_id + 1,
                    *[1 for _ in response.shape[1:]],
                ]
            ),
            mean=0.0,
            std=0.1,
        )
        targets.append(target)

    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=1.0e-5,
        weight_decay=1.0e-6,
        momentum_decay=5.0e-3,
        decoupled_weight_decay=True,
    )

    for epoch_id in range(epochs):
        epoch_ids = torch.randperm(max_id)

        optimizer.zero_grad()
        epoch_loss = 0.0

        for _ in range(0, max_id, batch_size):
            batch_ids = epoch_ids[0:batch_size]
            epoch_ids = epoch_ids[batch_size:]

            batch_response = model.forward_pretrain(batch_ids)
            batch_loss = 0.0
            for response_id in range(len(batch_response)):
                response = batch_response[response_id]
                target = targets[response_id][batch_ids]
                loss = F.mse_loss(response, target)
                batch_loss += loss
            batch_loss = batch_loss / ((max_id + 1) // batch_size)
            batch_loss = batch_loss / len(batch_response)
            batch_loss.backward()
            epoch_loss += batch_loss.item()

        print(f"Epoch {epoch_id + 1}/{epochs}, Loss: {epoch_loss}")

        optimizer.step()

    pass


def train(
    data: torch.Tensor,
    total_steps: int,
    batch_size: int,
    grad_accumulation_steps: int,
    sliding_batch: bool,
    loss_channels_weights: torch.Tensor,
    use_regularization_model: bool,
    use_regularization_ctx: bool,
    use_regularization_latents: bool,
    regularization_alpha_model: float,
    regularization_alpha_ctx: float,
    regularization_alpha_latents: float,
    regularization_low_weights_model_bound: Union[float, List[float]],
    regularization_low_weights_model_alpha: Union[float, List[float]],
    regularization_low_weights_fn: Union[Callable, List[Callable]],
    weights_hysteresis_loop: bool,
    weights_hysteresis_loop_zero_bound: float,
    weights_hysteresis_loop_zero_jump: float,
    loss_weights_main_reg: List[float],
    grad_min_clip_value: float,
    grad_max_clip_value: float,
    grad_clip_norm: float,
    clip_grad_by: Optional[str],
    freeze_ctx_nth_epoch: Optional[int],
    freeze_ctx_epochs: Optional[int],
    freeze_latents_nth_epoch: Optional[int],
    freeze_latents_epochs: Optional[int],
    freeze_model_nth_epoch: Optional[int],
    freeze_model_epochs: Optional[int],
    model: DecoderOnlyModel,
    optimizer: torch.optim.Optimizer,
    log_nth_update_step: int,
    images_path_dst: str = None,
    save_nth_iteration: int = 100,
    savers: list = [],
    to_save: list[bool] = [],
    warmup_scheduler: Optional[Callable] = None,
    warmup_epochs: Optional[int] = None,
    lr_scheduler: Optional[Callable] = None,
) -> None:
    with torch.no_grad():
        data_lab = data.cpu().to(dtype=model.dtype_weights)
        data_lab = data_lab / 255.0
        data_lab = data_rgb_to_lab(data_lab)
        data_lab = data_lab.to(dtype=torch.float16, device=data.device)
        data = data.cpu().detach()
        gc.collect()

    loss_channels_weights = loss_channels_weights.to(data_lab.device)
    accumulation_step = 0
    epoch_ids = torch.randperm(data_lab.shape[0])
    epoch_idx = 0
    last_grad_ctx = 0.0
    last_grad_latents = 0.0
    loss_logging_accumulator = []
    loss_main_logging_accumulator = []
    loss_reg_logging_accumulator = []
    stdr_logging_accumulator = []

    frozen_ctx_last_epoch_id = 0
    epoch_frozen_ctx = None
    frozen_latents_last_epoch_id = 0
    epoch_frozen_latents = None
    frozen_model_last_epoch_id = 0
    epoch_frozen_model = None
    weights_update_step_idx = 0

    kldiv_loss_fn = nn.KLDivLoss(reduction="none", log_target=True)

    initially_decoded_samples = model(torch.arange(0, min(4, data_lab.shape[0]), 1))
    generate_images_from_data(
        data=data_lab_to_rgb(initially_decoded_samples),
        images_path_dst=images_path_dst,
        prefix="initial_state",
    )

    params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n# --------------------------------------------------- #\n")
    print(f"Model type: {type(model)}")
    print(f"Model parameters: {params_model}")
    print(f"Output samples shape: {initially_decoded_samples.shape[1:]}")
    print("\n# --------------------------------------------------- #\n")

    print(f"Starting training. Epoch #{epoch_idx}")
    for step_idx in range(total_steps):
        if len(epoch_ids) < batch_size:
            epoch_idx = epoch_idx + 1
            print(f"\n# ==============> New epoch: #{epoch_idx}")
            epoch_ids = torch.randperm(data_lab.shape[0])

        if freeze_ctx_nth_epoch is not None and freeze_ctx_nth_epoch > 0:
            if epoch_frozen_ctx is None:
                if epoch_idx % freeze_ctx_nth_epoch == 0:
                    epoch_frozen_ctx = 1
                    frozen_ctx_last_epoch_id = epoch_idx
                    model_freeze_ctx(model)
            else:
                if frozen_ctx_last_epoch_id != epoch_idx:
                    frozen_ctx_last_epoch_id = epoch_idx
                    if epoch_frozen_ctx >= freeze_ctx_epochs:
                        model_unfreeze_ctx(model)
                        epoch_frozen_ctx = None
                    else:
                        epoch_frozen_ctx = epoch_frozen_ctx + 1

        if freeze_latents_nth_epoch is not None and freeze_latents_nth_epoch > 0:
            if epoch_frozen_latents is None:
                if epoch_idx % freeze_latents_nth_epoch == 0:
                    epoch_frozen_latents = 1
                    frozen_latents_last_epoch_id = epoch_idx
                    model_freeze_latents(model)
            else:
                if frozen_latents_last_epoch_id != epoch_idx:
                    frozen_latents_last_epoch_id = epoch_idx
                    if epoch_frozen_latents >= freeze_latents_epochs:
                        model_unfreeze_latents(model)
                        epoch_frozen_latents = None
                    else:
                        epoch_frozen_latents = epoch_frozen_latents + 1

        if freeze_model_nth_epoch is not None and freeze_model_nth_epoch > 0:
            if epoch_frozen_model is None:
                if epoch_idx % freeze_model_nth_epoch == 0:
                    epoch_frozen_model = 1
                    frozen_model_last_epoch_id = epoch_idx
                    model_freeze_model(model)
            else:
                if frozen_model_last_epoch_id != epoch_idx:
                    frozen_model_last_epoch_id = epoch_idx
                    if epoch_frozen_model >= freeze_model_epochs:
                        model_unfreeze_model(model)
                        epoch_frozen_model = None
                    else:
                        epoch_frozen_model = epoch_frozen_model + 1

        batch_ids = epoch_ids[0:batch_size]
        epoch_ids = epoch_ids[(1 if sliding_batch else batch_size) :]

        sample = data_lab[batch_ids]
        sample = sample.to(dtype=model.dtype_weights)

        accumulation_step = accumulation_step + 1
        decoded = model(batch_ids)

        loss_channels_weights = loss_channels_weights.reshape([1, 3, 1, 1])
        loss_base_decoded = decoded * loss_channels_weights
        loss_base_targets = sample * loss_channels_weights

        # Loss A
        # loss_main = (loss_base_targets - loss_base_decoded).std().sqrt()

        # Loss B
        # loss_main = F.l1_loss(loss_base_decoded, loss_base_targets)

        # Loss C
        # loss_main = F.mse_loss(loss_base_decoded, loss_base_targets)

        # Loss D
        loss_base_targets = (loss_base_targets + 0.01) / (1.02)
        loss_base_decoded = F.log_softmax(loss_base_decoded.flatten(2), dim=-1)
        loss_base_targets = F.log_softmax(loss_base_targets.flatten(2), dim=-1)
        loss_main = kldiv_loss_fn(loss_base_decoded, loss_base_targets)
        loss_main = loss_main.sum().div(math.prod(loss_base_targets.shape[0:2]))

        # Regularizations
        null_placeholder = torch.tensor([0.0]).to(
            dtype=loss_main.dtype,
            device=loss_main.device,
        )
        reg_term_model = (
            get_regularization_term_model(
                model=model,
                alpha=regularization_alpha_model,
            )
            if use_regularization_model
            else null_placeholder.clone()
        )
        reg_term_ctx = (
            get_regularization_term_ctx(
                model=model,
                alpha=regularization_alpha_ctx,
            )
            if use_regularization_ctx
            else null_placeholder.clone()
        )
        reg_term_latents = (
            get_regularization_term_latents(
                model=model,
                alpha=regularization_alpha_latents,
            )
            if use_regularization_latents
            else null_placeholder.clone()
        )

        if regularization_low_weights_fn is not None:
            if type(regularization_low_weights_fn) == list:
                reg_sum = 0.0
                for fn, bound, alpha in zip(
                    regularization_low_weights_fn,
                    regularization_low_weights_model_bound,
                    regularization_low_weights_model_alpha,
                ):
                    reg_sum = reg_sum + fn(model, bound, alpha)
                reg_term_low_weights_model = reg_sum
            else:
                reg_term_low_weights_model = regularization_low_weights_fn(
                    model=model,
                    bound=regularization_low_weights_model_bound,
                    alpha=regularization_low_weights_model_alpha,
                )
        else:
            reg_term_low_weights_model = null_placeholder.clone()

        loss_reg = (
            reg_term_model
            + reg_term_ctx
            + reg_term_latents
            + reg_term_low_weights_model
        )

        loss = (
            loss_main * loss_weights_main_reg[0] + loss_reg * loss_weights_main_reg[1]
        )
        loss_logging_accumulator.append(loss.detach().item())
        stdr_logging_accumulator.append((sample - decoded).detach().std())
        loss = loss / grad_accumulation_steps
        loss.backward()

        if accumulation_step == grad_accumulation_steps:
            weights_update_step_idx = weights_update_step_idx + 1
            accumulation_step = 0

            last_grad_ctx = (
                model.data_cache_ctx.grad.abs().mean().item()
                if model.data_cache_ctx.requires_grad
                and model.data_cache_ctx.requires_grad is not None
                else -1.0
            )
            last_grad_latents = (
                model.data_cache_latents.grad.abs().mean().item()
                if model.data_cache_latents.requires_grad
                and model.data_cache_latents.grad is not None
                else -1.0
            )

            grad_clip_val = -1.0
            if clip_grad_by is not None:
                grad_max_clip_value = (
                    grad_max_clip_value if grad_max_clip_value is not None else 1.0
                )
                if "ctx" in clip_grad_by:
                    grad_clip_val = model.data_cache_ctx.grad.abs().mean().item()
                    grad_clip_val = min(grad_clip_val, grad_max_clip_value)
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_val)
                if "latents" in clip_grad_by:
                    grad_clip_val = model.data_cache_latents.grad.abs().mean().item()
                    grad_clip_val = min(grad_clip_val, grad_max_clip_value)
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_val)
                if "mean" in clip_grad_by:
                    mean_grads = []
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            mean_grads.append(param.grad.abs().mean().item())
                    grad_clip_val = sum(mean_grads) / len(mean_grads)
                    grad_clip_val = min(grad_clip_val, grad_max_clip_value)
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_val)
            elif grad_min_clip_value is not None and grad_max_clip_value is not None:
                if grad_min_clip_value == grad_max_clip_value:
                    grad_clip_val = grad_min_clip_value
                else:
                    mean_grad = [p.grad.abs().mean().item() for p in model.parameters()]
                    mean_grad = sum(mean_grad) / len(mean_grad)
                    grad_clip_val = max(grad_min_clip_value, mean_grad)
                    grad_clip_val = min(grad_clip_val, grad_max_clip_value)
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_val)
            elif grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    if epoch_idx > warmup_epochs:
                        lr_scheduler.step(epoch_idx - warmup_epochs)

            optimizer.step()
            optimizer.zero_grad()

            if weights_hysteresis_loop:
                weights_hysteresis_loop_zero(
                    model=model,
                    bound=weights_hysteresis_loop_zero_bound,
                    jump=weights_hysteresis_loop_zero_jump,
                )

        # print(f"step loss: {loss.item()}")

        loss_main_logging_accumulator.append(loss_main.item())
        loss_reg_logging_accumulator.append(loss_reg.item())

        if (
            weights_update_step_idx > 0
            and (weights_update_step_idx) % log_nth_update_step == 0
        ):
            weights_update_step_idx = 0
            print(
                "\n# ==============> "
                + "\n".join(
                    [
                        f"Iteration #{step_idx+1}:",
                        f"Loss main: {loss_main.item()}",
                        f"Loss reg.: {loss_reg.item()}; model={reg_term_model.item():.5f}, ctx={reg_term_ctx.item():.5f}, latents={reg_term_latents.item():.5f}",
                        f"Loss m/r scaled: {loss_main.item()/loss_weights_main_reg[0]:.5f}/{loss_reg.item()/loss_weights_main_reg[1]:.5f}",
                        f"Loss total: {loss.item()}",
                        f"Mean grad ctx/latents: {last_grad_ctx:.10f}/{last_grad_latents:.10f}",
                        f"Grad clipping value: {grad_clip_val:.10f}",
                        f"Grad clipping norm: {(grad_clip_norm if grad_clip_norm is not None else -1.0):.10f}",
                        f"LR: {optimizer.param_groups[0]['lr']:.10f}",
                        f"StdR: {(sample - decoded).detach().std()}",
                        f"Mean loss (main): {(sum(loss_main_logging_accumulator)/len(loss_main_logging_accumulator)):.5f}",
                        f"Mean loss (reg): {(sum(loss_reg_logging_accumulator)/len(loss_reg_logging_accumulator)):.5f}",
                        f"Mean StdR: {(sum(stdr_logging_accumulator)/len(stdr_logging_accumulator)):.5f}",
                        f"Mean loss (total): {(sum(loss_logging_accumulator)/len(loss_logging_accumulator)):.5f}",
                    ]
                )
                + "\n# <=============="
            )
            # exit()
            stdr_logging_accumulator = []
            loss_main_logging_accumulator = []
            loss_reg_logging_accumulator = []
            loss_logging_accumulator = []
            generate_images_from_data(
                data=data_lab_to_rgb(decoded[-1].unsqueeze(0).clamp(0.0, 1.0)),
                images_path_dst=images_path_dst,
                prefix=f"output_e{epoch_idx:0>4d}_i{(step_idx+1):0>7d}",
            )
        if (step_idx + 1) % save_nth_iteration == 0:
            for j, saver in enumerate(savers):
                if to_save[j]:
                    saver(step_idx + 1)
            pass

        del sample
        del decoded
        del loss_base_decoded
        del loss_base_targets

    print("\n# --------------------------------------------------- #\n")

    generate_images_from_data(
        data=data_lab_to_rgb(decoded[-1].unsqueeze(0).clamp(0.0, 1.0)),
        images_path_dst=images_path_dst,
        prefix=f"output_final_i{step_idx+1}",
    )

    pass


def get_regularization_term_model(
    model: nn.Module,
    alpha: float,
) -> torch.Tensor:
    sum = 0.0
    for name, param in model.named_parameters():
        if "data_cache" not in name:
            sum = sum + (param**2).sum()
    return alpha * sum


def get_regularization_term_ctx(
    model: nn.Module,
    alpha: float,
) -> torch.Tensor:
    sum = 0.0
    for name, param in model.named_parameters():
        if "data_cache_ctx" in name:
            sum = sum + (param**2).sum()
    return alpha * (sum / model.data_cache_ctx.shape[0])


def get_regularization_term_latents(
    model: nn.Module,
    alpha: float,
) -> torch.Tensor:
    sum = 0.0
    for name, param in model.named_parameters():
        if "data_cache_latents" in name:
            sum = sum + (param**2).sum()
    return alpha * (sum / model.data_cache_latents.shape[0])


def get_regularization_term_low_weights_model_alpha(
    model: nn.Module,
    bound: float,
    alpha: float,
) -> torch.Tensor:
    sum = 0.0
    for name, param in model.named_parameters():
        if any(
            [
                "data_cache" in name,
                "weights_lib.mod_i" in name,
                "weights_lib.mod_j" in name,
            ]
        ):
            continue
        vars = param.abs().clamp(0.0, bound).sub(bound).abs()
        sum = sum + vars.sum()
    return alpha * sum


def get_regularization_term_low_weights_model_beta(
    model: nn.Module,
    bound: float,
    alpha: float,
) -> torch.Tensor:
    sum = 0.0
    for name, param in model.named_parameters():
        if any(
            [
                "data_cache" in name,
                "weights_lib.mod_i" in name,
                "weights_lib.mod_j" in name,
            ]
        ):
            continue
        vars = param.abs().clamp(0.0, bound).sub(bound).abs()
        varsum = vars.mul(1.0 / bound).sqrt()
        varsum = varsum / ((varsum[varsum > 0.0]).numel() + 1.0)
        varsum = varsum.sum()
        sum = sum + varsum
    return alpha * sum


def weights_hysteresis_loop_zero(
    model: nn.Module,
    bound: float,
    jump: float,
) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "data_cache" in name:
                continue
            cond_to_neg = (param > -0.0) & (param < +bound)
            cond_to_pos = (param < +0.0) & (param > -bound)
            if cond_to_neg.any():
                param[cond_to_neg] = param[cond_to_neg] - jump
            if cond_to_pos.any():
                param[cond_to_pos] = param[cond_to_pos] + jump


def count_vals(
    model: nn.Module,
    target: float,
    excl: str = None,
    incl: str = None,
    abs: bool = True,
    mode: str = "gt",
) -> int:
    assert mode in ["gt", "lt", "eq"], f"mode must be 'gt' or 'lt', got {mode}"
    with torch.no_grad():
        cond_fn = lambda x: torch.where(
            ((x.abs() if abs else x) < target)
            if mode == "lt"
            else (
                ((x.abs() if abs else x) > target)
                if mode == "gt"
                else ((x.abs() if abs else x) == target)
            )
        )
        cnt_fn = lambda x, acc, cond: acc + x[cond].numel()
        sum_fn = lambda x, acc, cond: acc + (
            x[cond].abs().sum() if abs else x[cond].sum()
        )
        cnt = 0
        sum = 0
        cond = None
        for name, param in model.named_parameters():
            if excl is not None and excl in name:
                continue
            else:
                cond = cond_fn(param)
            if incl is not None:
                cnt = cnt_fn(param, cnt, cond) if incl in name else cnt
                sum = sum_fn(param, sum, cond) if incl in name else sum
            else:
                cnt = cnt_fn(param, cnt, cond)
                sum = sum_fn(param, sum, cond)
    return dict(cnt=cnt, sum=sum)


def count_params(
    model: nn.Module,
) -> int:
    sum = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "data_cache" in name:
                continue
            sum = sum + param.numel()
    return sum


def model_info(
    model: nn.Module,
) -> None:
    info = [
        f"{n: <64s} -> "
        + ", ".join(
            [
                f"{p.abs().min().item()=:.6f}",
                f"{p.abs().max().item()=:.6f}",
                f"{p.abs().mean().item()=:.6f}",
                f"{p.min().item()=:.6f}",
                f"{p.max().item()=:.6f}",
                f"{p.mean().item()=:.6f}",
                f"{p.std().item()=:.6f}",
            ]
        )
        for n, p in model.named_parameters()
    ]
    print("\n".join(info))
    pass


def model_freeze_model(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache" in name:
            continue
        param.requires_grad = False
        print(f"freeze_model: {name}")
    pass


def model_unfreeze_model(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache" in name:
            continue
        param.requires_grad = True
        print(f"unfreeze_model: {name}")
    pass


def model_freeze_latents(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache_latents" in name:
            param.requires_grad = False
            print(f"freeze_latents: {name}")
    pass


def model_unfreeze_latents(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache_latents" in name:
            param.requires_grad = True
            print(f"unfreeze_latents: {name}")
    pass


def model_freeze_ctx(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache_ctx" in name:
            param.requires_grad = False
            print(f"freeze_ctx: {name}")
    pass


def model_unfreeze_ctx(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache_ctx" in name:
            param.requires_grad = True
            print(f"unfreeze_ctx: {name}")
    pass


def model_freeze_all(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = False
    print(f"model_freeze_all")
    pass


def model_unfreeze_all(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = True
    print(f"model_unfreeze_all")
    pass


def model_expand_cache(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache_latents" in name:
            current = param.data.clone()
            new = torch.nn.init.normal_(
                torch.empty(
                    [2048, 8, 8, 8],
                    dtype=current.dtype,
                    device=current.device,
                ),
                mean=0.0,
                std=1.0e-4,
            )
            new[0 : current.shape[0]] = current
            param.data = new
            print(f"model_expand_cache: {name}")
        if "data_cache_ctx" in name:
            current = param.data.clone()
            new = torch.nn.init.normal_(
                torch.empty(
                    [2048, 32],
                    dtype=current.dtype,
                    device=current.device,
                ),
                mean=0.0,
                std=1.0e-4,
            )
            new[0 : current.shape[0]] = current
            param.data = new
            print(f"model_expand_cache: {name}")
    pass


def model_constant_ctx(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    v = 0.0001
    for name, param in model.named_parameters():
        if "data_cache_ctx" in name:
            param.data = torch.nn.init.constant_(
                param.data,
                val=v,
            )
            print(f"model_constant_ctx: {v}")
    pass


def model_constant_latents(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    v = 0.1
    for name, param in model.named_parameters():
        if "data_cache_latents" in name:
            param.data = torch.nn.init.constant_(
                param.data,
                val=v,
            )
            print(f"model_constant_latents: {v}")
    pass


def model_same_latents(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache_latents" in name:
            sample = param.data[0].clone().unsqueeze(0)
            sample = nn.init.uniform_(
                sample,
                a=-0.1,
                b=+0.1,
            )
            param.data = sample.repeat([param.data.shape[0], 1, 1, 1])
            print(f"model_same_latents")
    pass


def model_perturb_small_weights(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    a: float = 0.0001,
    b: float = 0.0010,
) -> None:
    print(f"model_perturb_small_weights: {a=}, {b=}")
    for name, param in model.named_parameters():
        if "data_cache" not in name:
            delta = torch.nn.init.uniform_(
                torch.empty_like(param.data),
                a=a,
                b=b,
            )
            delta = delta * torch.randn_like(param.data).sign()
            cnd = torch.where(param.data.abs() < a)
            param.data[cnd] = (param.data + delta)[cnd]
            cnd = torch.where(param.data.abs() < a)
            param.data[cnd] = delta[cnd]
            print(f"model_perturb_small_weights: {name}")
    pass


def model_reinit_weights_same_distribution(
    model: DecoderOnlyModel,
    target: str = None,
) -> None:
    for name, param in model.named_parameters():
        if target is not None and target not in name:
            continue
        mean = param.data.mean()
        std = param.data.std()
        param.data = torch.nn.init.normal_(
            param.data,
            mean=mean,
            std=std,
        )
        print(
            " ".join(
                [
                    f"model_reinit_weights_same_distribution: {name};",
                    f"mean/std: {mean.item()}/{std.item()}",
                ]
            )
        )
    pass


def model_data_cache_double(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    model.data_cache_ctx = nn.Parameter(
        torch.cat(
            [
                model.data_cache_ctx,
                torch.zeros_like(model.data_cache_ctx),
            ],
            dim=0,
        )
    )
    model.data_cache_latents = nn.Parameter(
        torch.cat(
            [
                model.data_cache_latents,
                torch.zeros_like(model.data_cache_latents),
            ],
            dim=0,
        )
    )
    print(f"change_data_cache: {model.data_cache_ctx.shape}")
    print(f"change_data_cache: {model.data_cache_latents.shape}")
    pass


def model_quantize_weights(
    model: DecoderOnlyModel,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    levels = 2**8
    offset = 1.0 / (levels * 2)
    scale = 1.5

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "data_cache" in name:
                continue
            sign = torch.where(param.data > 0.0, +1.0, -1.0)
            sign = sign.to(dtype=dtype, device=device)
            d_max = param.data.abs().max()
            d_s = ((param.data / d_max) * (levels - 1)) // 1.0
            d_s = ((d_s / levels) + offset).mul(sign).mul(scale)
            d_s = d_s.to(dtype=dtype, device=device)
            param.data = d_s
            print(
                "; ".join(
                    [
                        f"quantize_weights: {name}",
                        f"levels={levels}",
                        f"offset={offset}",
                        f"scale={scale}",
                    ]
                )
            )
    pass


def eval_infer(
    model: DecoderOnlyModel,
    start_from: int = 0,
    batches: int = 1,
    batch_size: int = 16,
) -> None:
    training = model.training
    dropout_buffer = model.dropout_rate_latents

    if training:
        model.eval()
        model.dropout_rate_latents = 0.0

    for batch_id in range(0, batches, 1):
        id_start = start_from
        id_end = start_from + batch_size + batch_size * batch_id
        with torch.no_grad():
            model_output = model(torch.arange(id_start, id_end, 1))
            data_rgb = data_lab_to_rgb(model_output)
            generate_images_from_data(data_rgb, images_path_dst, f"infer_b{batch_id}")

    if training:
        model.train()
        model.dropout_rate_latents = dropout_buffer

    return None


def eval_ctx_rand(
    model: DecoderOnlyModel,
) -> None:
    with torch.no_grad():
        ctx = model.data_cache_ctx.data.clone()
        model.data_cache_ctx.data = torch.randn_like(ctx)
        model_output = model(torch.arange(0, 16, 1))
        model.data_cache_ctx.data = ctx
        data_rgb = data_lab_to_rgb(model_output)
        generate_images_from_data(data_rgb, images_path_dst, "ctx_rand")
    pass


def eval_latents_rand(
    model: DecoderOnlyModel,
) -> None:
    with torch.no_grad():
        latents = model.data_cache_latents.data.clone()
        model.data_cache_latents.data = torch.randn_like(latents)
        model_output = model(torch.arange(0, 16, 1))
        model.data_cache_latents.data = latents
        data_rgb = data_lab_to_rgb(model_output)
        generate_images_from_data(data_rgb, images_path_dst, "latents_rand")
    pass


def eval_latents_ones(
    model: DecoderOnlyModel,
) -> None:
    with torch.no_grad():
        latents = model.data_cache_latents.data.clone()
        model.data_cache_latents.data = torch.ones_like(latents)
        model_output = model(torch.arange(0, 16, 1))
        model.data_cache_latents.data = latents
        data_rgb = data_lab_to_rgb(model_output)
        generate_images_from_data(data_rgb, images_path_dst, "latents_ones")
    pass


def downscale_weights(
    model: DecoderOnlyModel,
    lim: float = 1.0,
) -> None:
    for name, param in model.named_parameters():
        pmax = param.data.abs().max()
        if pmax > lim:
            param.data = param.data / pmax
            print(f"weights_clamp: {name}")
    pass


def optim_change_momentum(
    optim: torch.optim.Optimizer,
    momentum: float,
) -> None:
    for group in optim.param_groups:
        group["momentum"] = momentum
        for p in group["params"]:
            state = optim.state[p]
            state["x0"] = torch.zeros_like(state["s"]).detach()

    print(f"optim_change_momentum: {momentum}")
    return None


def model_change_data_cache_latents(
    model: DecoderOnlyModel,
    device: torch.device,
    dtype: torch.dtype,
    latents_size: list[int],
) -> None:
    for name, param in model.named_parameters():
        if "data_cache_latents" in name:
            param.data = torch.randn(latents_size, device=device, dtype=dtype)
            print(f"model_change_data_cache_latents: {name}")
            continue
    return None


def model_fill_data_cache_latents(
    model: DecoderOnlyModel,
    value: float = 1.0e-5,
) -> None:
    model.data_cache_latents.data = torch.nn.init.constant_(
        model.data_cache_latents.data, val=value
    )
    print(f"model_fill_data_cache_latents: filled with {value}")
    pass


def model_fill_data_cache_context(
    model: DecoderOnlyModel,
    value: float = 1.0e-5,
) -> None:
    model.data_cache_ctx.data = torch.nn.init.constant_(
        model.data_cache_ctx.data, val=value
    )
    print(f"model_fill_data_cache_context: filled with {value}")
    pass


def model_change_data_cache_ctx(
    model: DecoderOnlyModel,
    device: torch.device,
    dtype: torch.dtype,
    ctx_size: list[int],
) -> None:
    for name, param in model.named_parameters():
        if "data_cache_ctx" in name:
            param.data = torch.randn(ctx_size, device=device, dtype=dtype)
            print(f"model_change_data_cache_ctx: {name}")
            continue
    return None


def model_freeze_block(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    block_name: str = "no_block_passed",
) -> None:
    for name, param in model.named_parameters():
        if block_name in name:
            param.requires_grad = False
            print(f"model_freeze_block: {name}")
    pass


def model_unfreeze_block(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    block_name: str = "no_block_passed",
) -> None:
    for name, param in model.named_parameters():
        if block_name in name:
            param.requires_grad = True
            print(f"model_unfreeze_block: {name}")
    pass


def renormalize_weigths(
    model: DecoderOnlyModel,
    include_data_cache: bool = False,
) -> None:
    max_std = torch.tensor([0.0], device=model.data_cache_latents.device)

    for name, param in model.named_parameters():
        if not include_data_cache and "data_cache" in name:
            continue
        std = param.data.std()
        if std > max_std:
            max_std = std

    for name, param in model.named_parameters():
        if not include_data_cache and "data_cache" in name:
            continue
        param.data = param.data / max_std

    pass


def model_perturb_weights(
    model: DecoderOnlyModel,
    rate: float = 0.025,
    include_data_cache: bool = False,
) -> None:

    for name, param in model.named_parameters():
        if not include_data_cache and "data_cache" in name:
            continue
        std = param.data.std()
        if std == 0.0 or std != std:
            continue
        noise = torch.randn_like(param.data) * param.data.std() * rate
        param.data = param.data + noise
        print(f"model_perturb_weights (rate={rate:.5f}): {name}")

    pass


def model_extend_data_cache(
    model: DecoderOnlyModel,
    size: int,
    std: float = None,
) -> None:

    assert size > 0, "size must be positive"
    assert (
        size > model.data_cache_ctx.data.shape[0]
    ), "size must be greater than current size"

    current_ctx = model.data_cache_ctx.data.clone()
    current_latents = model.data_cache_latents.data.clone()

    model.data_cache_ctx.data = torch.randn(
        [size, *model.data_cache_ctx.data.shape[1:]],
        dtype=current_ctx.dtype,
        device=current_ctx.device,
    )
    model.data_cache_latents.data = torch.randn(
        [size, *model.data_cache_latents.data.shape[1:]],
        dtype=current_latents.dtype,
        device=current_latents.device,
    )

    if std is None:
        model.data_cache_ctx.data = (
            model.data_cache_ctx.data - model.data_cache_ctx.data.mean()
        ) + current_ctx.mean()
        model.data_cache_latents.data = (
            model.data_cache_latents.data - model.data_cache_latents.data.mean()
        ) + current_latents.mean()

        model.data_cache_ctx.data = (
            model.data_cache_ctx.data / model.data_cache_ctx.data.std()
        ) * current_ctx.std()
        model.data_cache_latents.data = (
            model.data_cache_latents.data / model.data_cache_latents.data.std()
        ) * current_latents.std()
    else:
        model.data_cache_ctx.data = model.data_cache_ctx.data * std
        model.data_cache_latents.data = model.data_cache_latents.data * std

    model.data_cache_ctx.data[0 : current_ctx.shape[0]] = current_ctx.clone()
    model.data_cache_latents.data[0 : current_latents.shape[0]] = (
        current_latents.clone()
    )

    print(f"model_extend_data_cache: {size}")

    del current_ctx, current_latents

    pass


def clear() -> None:
    os.system("clear")
    pass


def model_siglog_alpha_fix(
    model: DecoderOnlyModel,
    lower_bound: float = 0.1,
    mean: float = 1.0 / math.e,
    std: float = 0.01,
) -> None:
    req_grad = model.siglog_params.requires_grad

    model.siglog_params.requires_grad = False

    fill_t = torch.nn.init.normal(
        model.siglog_params[model.siglog_params <= lower_bound].clone(),
        mean=mean,
        std=std,
    )

    model.siglog_params[model.siglog_params <= lower_bound] = fill_t
    model.siglog_params.requires_grad = req_grad

    print(f"model_siglog_alpha_fix: {lower_bound=}, {mean=}, {std=}")
    print(f"model_siglog_alpha_fix: values fixed: {fill_t.numel()}")
    pass


if __name__ == "__main__":
    train_mode = True

    load_model = False
    load_optim = False
    drop_ctx_cache = False
    drop_latents_cache = False
    onload_model_fn = [
        # model_freeze_model,
        # model_data_cache_double,
        # model_quantize_weights,
        # lambda m, d, t: model_change_data_cache_latents(m, d, t, [1024, 8, 8, 8]),
        # model_freeze_model,
        # model_freeze_ctx,
        # model_expand_cache,
        # model_constant_ctx,
        # model_constant_latents,
        # model_same_latents,
        # model_freeze_latents,
        # model_data_cache_double,
        # model_data_cache_double,
        # lambda m, d, t: model_change_data_cache_latents(m, d, t, [512, 16, 16, 16]),
        # lambda m, d, t: model_change_data_cache_ctx(m, d, t, [512, 32]),
        # model_unfreeze_model,
        # model_unfreeze_latents,
        # lambda m, d, t: model_perturb_small_weights(m, a=0.001, b=0.005),
        # lambda m, d, t: model_perturb_weights(m, 0.10, False),
        # lambda m, d, t: model_siglog_alpha_fix(m, 0.1, 1.0 / math.e, 0.01),
        # model_perturb_small_weights,
        # lambda m, d, t: model_extend_data_cache(m, 128, 1.0e-6),
        # lambda m, d, t: model_reinit_weights_same_distribution(m, True),
        # lambda m, d, t: model_fill_data_cache_latents(m, 1.0e-6),
        # lambda m, d, t: model_reinit_weights_same_distribution(m),
        # lambda m, d, t: model_reinit_weights_same_distribution(m, target="siglog_params"),
        # lambda m, d, t: model_reinit_weights_same_distribution(m, target="block_out"),
        # lambda m, d, t: model_reinit_weights_same_distribution(m, target="block_01"),
        # lambda m, d, t: model_reinit_weights_same_distribution(m, target="block_02"),
        # lambda m, d, t: model_reinit_weights_same_distribution(m, target="block_03"),
        # lambda m, d, t: model_reinit_weights_same_distribution(m, target="block_04"),
        # lambda m, d, t: model_reinit_weights_same_distribution(m, target="block_05"),
        # lambda m, d, t: model_reinit_weights_same_distribution(m, target="data_cache"),
        lambda m, d, t: model_fill_data_cache_context(m, 1.0e-5),
        lambda m, d, t: model_fill_data_cache_latents(m, 1.0e-5),
        # model_freeze_all,
        # model_unfreeze_ctx,
        # model_unfreeze_latents,
        # model_unfreeze_all,
        # model_freeze_model,
        # model_freeze_latents,
        # model_freeze_ctx,
        # lambda m, d, t: model_unfreeze_block(m, d, t, "siglog_params"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_out"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_01"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_02"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_03"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_04"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_05"),
        # lambda m, d, t: model_freeze_block(m, d, t, "siglog_params"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_out"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_01"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_02"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_03"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_04"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_05"),
    ]
    onload_optim_fn = [
        # lambda o: optim_change_momentum(o, 0.9),
    ]

    path_prefix_load = "/mnt/f/git_AIResearch/dyna/data/models"
    path_prefix_save = "/mnt/f/git_AIResearch/dyna/data/models"
    load_path_model = f"{path_prefix_load}/"
    load_path_optim = f"{path_prefix_load}/"
    save_path_model = f"{path_prefix_save}/model.Type-08.G00.AdamW"
    save_path_optim = f"{path_prefix_save}/optim.Type-08.G00.AdamW"
    save_model = True
    save_optim = False
    save_nth_iteration = 10_000
    log_nth_update_step = 1

    # optimizer type
    optimizer_type = torch.optim.AdamW
    # optimizer: torch.optim.SGD
    sgd_learning_rate = 1.0e-5
    sgd_momentum = 0.0
    sgd_dampening = 0.0
    sgd_weight_decay = 0.0
    sgd_nesterov = False
    # optimizer: torch.optim.Adam
    adam_learning_rate = 2.5e-4
    adam_amsgrad = True
    adam_weight_decay = 0.0
    adam_eps = 1.0e-8
    # optimizer: torch.optim.AdamW
    adamw_learning_rate = 1.0e-5
    adamw_amsgrad = True
    adamw_weight_decay = 1.0e-3
    adamw_eps = 1.0e-8
    # optimizer: torch.optim.NAdam
    nadam_learning_rate = 1.0e-5
    nadam_weight_decay = 1.0e-6
    nadam_momentum_decay = 5.0e-3
    nadam_decoupled_weight_decay = True
    # optimizer: torch.optim.RAdam
    radam_learning_rate = 1.0e-5
    radam_weight_decay = 1.0e-4
    # optimizer: MADGRAD
    madgrad_learning_rate = 1.0e-4
    madgrad_momentum = 0.9
    madgrad_weight_decay = 0.0
    madgrad_eps = 1.0e-6
    # various for optimizers
    optim_update_lr = False
    optim_target_lr = 1.0e-4
    optim_update_wd = False
    optim_target_wd = 0.1

    data_cache_ctx_bound = 1.0e-4
    data_cache_latents_bound = 1.0e-4
    use_regularization_model = True
    use_regularization_ctx = False
    use_regularization_latents = False
    regularization_alpha_model = 1.0e-10
    regularization_alpha_ctx = 1.0e-12
    regularization_alpha_latents = 1.0e-12
    regularization_low_weights_model_bound = [
        1.0e-2,
        1.0e-4,
    ]
    regularization_low_weights_model_alpha = [
        2.5e-6,
        1.0e-3,
    ]
    # regularization_low_weights_fn = [
    #     get_regularization_term_low_weights_model_alpha,
    #     get_regularization_term_low_weights_model_beta,
    # ]
    regularization_low_weights_fn = None
    weights_hysteresis_loop = False
    weights_hysteresis_loop_zero_bound = 1.0e-3
    weights_hysteresis_loop_zero_jump = 2.0e-3
    loss_channels_weights = [1.0, 1.0, 1.0]
    loss_weights_main_reg = [1.0, 1.0]
    grad_min_clip_value = None
    grad_max_clip_value = None
    # clip_grad_by = "mean"  # ctx/latents/mean/None
    # clip_grad_by = "latents"
    # clip_grad_by = "ctx"
    clip_grad_by = None
    grad_clip_norm = None

    freeze_ctx_nth_epoch = 0
    freeze_ctx_epochs = 0
    freeze_latents_nth_epoch = 0
    freeze_latents_epochs = 0
    freeze_model_nth_epoch = 0
    freeze_model_epochs = 0

    nelements = 512
    data_cache_ctx_len = nelements
    data_cache_latents_len = nelements
    data_cache_latents_shape = [16, 8, 8]

    context_through = False

    dropout_rate_latents = 0.0
    dropout_rate_context = 0.0

    noisein_rate_latents = 0.0
    noisein_rate_context = 0.0
    noisein_rate_latents_input = 0.0
    noisein_rate_latents_output = 0.0
    noisein_rate_context_input = 0.0
    noisein_rate_context_output = 0.0

    noiseover_rate_latents = 0.0
    noiseover_rate_context = 0.0
    noiseover_rate_latents_input = 0.0
    noiseover_rate_latents_output = 0.0
    noiseover_rate_context_input = 0.0
    noiseover_rate_context_output = 0.0

    total_steps = 200_000
    batch_size = 32
    sliding_batch = False
    grad_accumulation_steps = nelements // batch_size

    images_sample_count = nelements
    starting_from = 1024 * 10
    images_path_src = "/mnt/f/Datasets/Images_512x512/dataset_01"
    images_path_dst = "/mnt/f/git_AIResearch/dyna/data/img_dst"
    output_shape = [161, 161]  # 16x16=[417, 417]; 8x8=[161, 161]
    dtype_weights = torch.float32
    device = torch.device("cuda")

    model = DecoderOnlyModel(
        data_cache_ctx_len=data_cache_ctx_len,
        data_cache_latents_len=data_cache_latents_len,
        data_cache_latents_shape=data_cache_latents_shape,
        dropout_rate_latents=dropout_rate_latents,
        dropout_rate_context=dropout_rate_context,
        noisein_rate_latents=noisein_rate_latents,
        noisein_rate_context=noisein_rate_context,
        noisein_rate_latents_input=noisein_rate_latents_input,
        noisein_rate_latents_output=noisein_rate_latents_output,
        noisein_rate_context_input=noisein_rate_context_input,
        noisein_rate_context_output=noisein_rate_context_output,
        noiseover_rate_latents=noiseover_rate_latents,
        noiseover_rate_context=noiseover_rate_context,
        noiseover_rate_latents_input=noiseover_rate_latents_input,
        noiseover_rate_latents_output=noiseover_rate_latents_output,
        noiseover_rate_context_input=noiseover_rate_context_input,
        noiseover_rate_context_output=noiseover_rate_context_output,
        data_cache_ctx_bound=data_cache_ctx_bound,
        data_cache_latents_bound=data_cache_latents_bound,
        context_through=context_through,
        dtype_weights=dtype_weights,
    ).to(device=device, dtype=dtype_weights)

    if load_model:
        inconsistent_keys = model.load_state_dict(
            torch.load(load_path_model),
            strict=False,
        )
        print(f"Model loaded from {load_path_model}")

    if onload_model_fn is not None:
        if callable(onload_model_fn):
            onload_model_fn(model, device, dtype_weights)
        if type(onload_model_fn) is list:
            for fn in onload_model_fn:
                if callable(fn):
                    fn(model, device, dtype_weights)

    if drop_ctx_cache:
        model.data_cache_ctx = nn.Parameter(
            torch.nn.init.uniform_(
                model.data_cache_ctx,
                a=-data_cache_ctx_bound,
                b=+data_cache_ctx_bound,
            ).to(
                dtype=dtype_weights,
            )
        )
        print("Dropped: data_cache_ctx")

    if drop_latents_cache:
        model.data_cache_latents = nn.Parameter(
            torch.nn.init.uniform_(
                model.data_cache_latents,
                a=-data_cache_latents_bound,
                b=+data_cache_latents_bound,
            ).to(
                dtype=dtype_weights,
            )
        )
        print("Dropped: data_cache_latents")

    print("Model specs:")
    for name, param in model.named_parameters():
        param.data = param.data.to(dtype=dtype_weights)
        print(f"{name}: {param.shape} -> {param.data.dtype}")

    if train_mode:
        model = model.train()
        print("Training mode")
    else:
        model = model.eval()
        print("Evaluation mode")

    if optimizer_type is torch.optim.Adam:
        print("Using Adam")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=adam_learning_rate,
            weight_decay=adam_weight_decay,
            amsgrad=adam_amsgrad,
            eps=adam_eps,
        )
    elif optimizer_type is torch.optim.NAdam:
        print("Using NAdam")
        optimizer = torch.optim.NAdam(
            model.parameters(),
            lr=nadam_learning_rate,
            weight_decay=nadam_weight_decay,
            momentum_decay=nadam_momentum_decay,
            decoupled_weight_decay=nadam_decoupled_weight_decay,
        )
    elif optimizer_type is torch.optim.RAdam:
        print("Using RAdam")
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=radam_learning_rate,
            weight_decay=radam_weight_decay,
        )
    elif optimizer_type is torch.optim.AdamW:
        print("Using AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=adamw_learning_rate,
            weight_decay=adamw_weight_decay,
            amsgrad=adamw_amsgrad,
            eps=adamw_eps,
        )
    elif optimizer_type is MADGRAD:
        print("Using MADGRAD")
        optimizer = MADGRAD(
            model.parameters(),
            lr=madgrad_learning_rate,
            weight_decay=madgrad_weight_decay,
            momentum=madgrad_momentum,
            eps=madgrad_eps,
        )
    elif optimizer_type is torch.optim.SGD:
        print("Using SGD")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=sgd_learning_rate,
            momentum=sgd_momentum,
            dampening=sgd_dampening,
            weight_decay=sgd_weight_decay,
            nesterov=sgd_nesterov,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    if load_optim:
        optimizer.load_state_dict(torch.load(load_path_optim))
        print(f"Optimizer loaded from {load_path_optim}")

    # onload_optim_fn
    if onload_optim_fn is not None:
        if callable(onload_optim_fn):
            onload_optim_fn(optimizer)
        if type(onload_optim_fn) is list:
            for fn in onload_optim_fn:
                if callable(fn):
                    fn(optimizer)

    with torch.no_grad():
        data = generate_data_from_images(
            shape=output_shape,
            images_path_src=images_path_src,
            images_sample_count=images_sample_count,
            starting_from=starting_from,
        ).to(device)

    print("Generated data specs:")
    print(f"{data.min()=}")
    print(f"{data.max()=}")

    savers = [
        lambda p: torch.save(model.state_dict(), f"{save_path_model}.{p}.pth"),
        lambda p: torch.save(optimizer.state_dict(), f"{save_path_optim}.{p}.pth"),
    ]

    if optim_update_lr:
        old_param = None
        for g in optimizer.param_groups:
            old_param = g["lr"] if old_param is None else old_param
            g["lr"] = optim_target_lr
        print(f"Updated learning rate from {old_param} to {optim_target_lr}")

    if optim_update_wd:
        old_param = None
        for g in optimizer.param_groups:
            old_param = g["weight_decay"] if old_param is None else old_param
            g["weight_decay"] = optim_target_wd
        print(f"Updated weight decay from {old_param} to {optim_target_wd}")

    # WARMUP
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer=optimizer,
    #     T_0=16,
    #     T_mult=1,
    # )
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer=optimizer,
        factor=1.00,
        total_iters=4096 * 16,
    )
    warmup_epochs = 128
    warmup_scheduler = warmup.LinearWarmup(
        optimizer=optimizer,
        warmup_period=warmup_epochs,
    )

    # lr_scheduler = None
    # warmup_epochs = None
    # warmup_scheduler = None

    start_training = lambda: train(
        data=data,
        total_steps=total_steps,
        batch_size=batch_size,
        grad_accumulation_steps=grad_accumulation_steps,
        sliding_batch=sliding_batch,
        loss_channels_weights=torch.tensor(loss_channels_weights),
        use_regularization_model=use_regularization_model,
        use_regularization_ctx=use_regularization_ctx,
        use_regularization_latents=use_regularization_latents,
        regularization_alpha_model=regularization_alpha_model,
        regularization_alpha_ctx=regularization_alpha_ctx,
        regularization_alpha_latents=regularization_alpha_latents,
        regularization_low_weights_model_bound=regularization_low_weights_model_bound,
        regularization_low_weights_model_alpha=regularization_low_weights_model_alpha,
        regularization_low_weights_fn=regularization_low_weights_fn,
        weights_hysteresis_loop=weights_hysteresis_loop,
        weights_hysteresis_loop_zero_bound=weights_hysteresis_loop_zero_bound,
        weights_hysteresis_loop_zero_jump=weights_hysteresis_loop_zero_jump,
        loss_weights_main_reg=loss_weights_main_reg,
        grad_min_clip_value=grad_min_clip_value,
        grad_max_clip_value=grad_max_clip_value,
        grad_clip_norm=grad_clip_norm,
        clip_grad_by=clip_grad_by,
        freeze_ctx_nth_epoch=freeze_ctx_nth_epoch,
        freeze_ctx_epochs=freeze_ctx_epochs,
        freeze_latents_nth_epoch=freeze_latents_nth_epoch,
        freeze_latents_epochs=freeze_latents_epochs,
        freeze_model_nth_epoch=freeze_model_nth_epoch,
        freeze_model_epochs=freeze_model_epochs,
        model=model,
        optimizer=optimizer,
        log_nth_update_step=log_nth_update_step,
        images_path_dst=images_path_dst,
        save_nth_iteration=save_nth_iteration,
        savers=savers,
        to_save=[save_model, save_optim],
        warmup_scheduler=warmup_scheduler,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
    )

    if train_mode:
        start_training()
