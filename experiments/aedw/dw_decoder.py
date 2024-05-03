import torch
import torch.nn as nn
import torch.nn.functional as F

from dyna import DynamicConv2D


class DWDecoder(nn.Module):
    def __init__(
        self,
        data_cache_ctx_len: int = None,
        data_cache_latents_len: int = None,
        data_cache_latents_shape: list[int] = None,
        use_bias: bool = True,
        bias_static: float = 0.0,
        context_length: int = 32,
        mod_rank: int = 16,
        kernel_size_t: list[int] = [4, 4],
        kernel_size_r: list[int] = [3, 3],
        stride_t: list[int] = [2, 2],
        stride_r: list[int] = [1, 1],
        padding_t: list[int] = [1, 1],
        padding_r: list[int] = [1, 1],
        dilation_t: list[int] = [1, 1],
        dilation_r: list[int] = [1, 1],
        output_padding_t: list[int] = [0, 0],
        eps: float = 1.0e-2,
        q_levels: int = 100,
        channels_io: int = 3,
        channels_dynamic_levels: list[int] = [8, 8, 8, 8, 8],
        dtype_weights: torch.dtype = torch.float32,
    ):
        super().__init__()

        # ================================================================================= #
        # ____________________________> Parameters.
        # ================================================================================= #
        self.data_cache_ctx_len = data_cache_ctx_len
        self.data_cache_latents_len = data_cache_latents_len
        self.data_cache_latents_shape = data_cache_latents_shape
        self.use_bias = use_bias
        self.bias_static = bias_static
        self.context_length = context_length
        self.mod_rank = mod_rank
        self.kernel_size_t = kernel_size_t
        self.kernel_size_r = kernel_size_r
        self.stride_t = stride_t
        self.stride_r = stride_r
        self.padding_t = padding_t
        self.padding_r = padding_r
        self.dilation_t = dilation_t
        self.dilation_r = dilation_r
        self.output_padding_t = output_padding_t
        self.eps = eps
        self.q_levels = q_levels
        self.channels_io = channels_io
        self.channels_dynamic_levels = channels_dynamic_levels
        self.dtype_weights = dtype_weights

        # ================================================================================= #
        # ____________________________> Submodules.
        # ================================================================================= #
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv_up_04_t = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[4],
            out_channels=self.channels_dynamic_levels[3],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_t,
            stride=self.stride_t,
            padding=self.padding_t,
            dilation=self.dilation_t,
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=self.output_padding_t,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_04_r = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[4],
            out_channels=self.channels_dynamic_levels[3],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_r,
            stride=self.stride_r,
            padding=self.padding_r,
            dilation=self.dilation_r,
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_04_sum = nn.Conv2d(
            in_channels=self.channels_dynamic_levels[3] * 2,
            out_channels=self.channels_dynamic_levels[3],
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            bias=False,
        )
        self.conv_up_03_t = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[3],
            out_channels=self.channels_dynamic_levels[2],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_t,
            stride=self.stride_t,
            padding=self.padding_t,
            dilation=self.dilation_t,
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=self.output_padding_t,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_03_r = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[3],
            out_channels=self.channels_dynamic_levels[2],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_r,
            stride=self.stride_r,
            padding=self.padding_r,
            dilation=self.dilation_r,
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_03_sum = nn.Conv2d(
            in_channels=self.channels_dynamic_levels[2] * 2,
            out_channels=self.channels_dynamic_levels[2],
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            bias=False,
        )
        self.conv_up_02_t = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[2],
            out_channels=self.channels_dynamic_levels[1],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_t,
            stride=self.stride_t,
            padding=self.padding_t,
            dilation=self.dilation_t,
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=self.output_padding_t,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_02_r = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[2],
            out_channels=self.channels_dynamic_levels[1],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_r,
            stride=self.stride_r,
            padding=self.padding_r,
            dilation=self.dilation_r,
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_02_sum = nn.Conv2d(
            in_channels=self.channels_dynamic_levels[1] * 2,
            out_channels=self.channels_dynamic_levels[1],
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            bias=False,
        )
        self.conv_up_01_t = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[1],
            out_channels=self.channels_dynamic_levels[0],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_t,
            stride=self.stride_t,
            padding=self.padding_t,
            dilation=self.dilation_t,
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=self.output_padding_t,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_01_r = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[1],
            out_channels=self.channels_dynamic_levels[0],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_r,
            stride=self.stride_r,
            padding=self.padding_r,
            dilation=self.dilation_r,
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_01_sum = nn.Conv2d(
            in_channels=self.channels_dynamic_levels[0] * 2,
            out_channels=self.channels_dynamic_levels[0],
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            bias=False,
        )
        self.conv_out = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[0],
            out_channels=self.channels_io,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )

        # ================================================================================= #
        # ____________________________> Additional parameters.
        # ================================================================================= #
        self.data_cache_ctx = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [self.data_cache_ctx_len, self.context_length],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=1.0e-2,
            )
        )
        self.data_cache_latents = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [self.data_cache_latents_len, *self.data_cache_latents_shape],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=1.0e-2,
            ),
        )

        pass

    def discretize(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            l = self.q_levels
            shift = 1.0 / (l * 2)
            x_q = (x * l).clamp(-l, +l)
            x_q = (x_q // 1.0).to(dtype=x.dtype, device=x.device)
            x_q = (x_q / l) + shift
        x = x + (x_q - x).detach()
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

    def forward(
        self,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        ids = ids.to(device=self.data_cache_latents.device, dtype=torch.int32)
        context = self.data_cache_ctx[ids]
        x = self.data_cache_latents[ids]

        # ########################################################## #
        # NOTE:
        #       Examine the behavior of the following code (the
        #   results look interesting):
        #
        # x_p = DynamicConv_A(+x, context)
        # x_n = DynamicConv_A(-x, context)
        # x = self.doubleLogNorm(x_p - x_n)
        # x = self.quantizer(x)

        # x_p = DynamicConv_B(+x, context)
        # x_n = DynamicConv_B(-x, context)
        # x = self.doubleLogNorm(x_p - x_n)
        # x = self.quantizer(x)

        # x_p = DynamicConv_C(+x, context)
        # x_n = DynamicConv_C(-x, context)
        # x = self.doubleLogNorm(x_p - x_n)
        # x_p = DynamicConv_D(+x, context)
        # x_n = DynamicConv_D(-x, context)
        # x = self.doubleLogNorm((x_p - x_n).abs())
        # ########################################################## #

        x_pos, x_neg = x, -x
        x_pos = self.conv_up_04_r(x_pos, context)
        x_pos = self.doubleLogNorm(x_pos)
        x_pos = self.upsample(x_pos)
        x_neg = self.conv_up_04_t(x_neg, context)
        x_neg = self.doubleLogNorm(x_neg)
        x_sum = torch.cat([x_pos, x_neg], dim=1)
        x_sum = self.conv_up_04_sum(x_sum)
        x = F.leaky_relu(x_sum, negative_slope=0.1)

        x_pos, x_neg = x, -x
        x_pos = self.conv_up_03_r(x_pos, context)
        x_pos = self.doubleLogNorm(x_pos)
        x_pos = self.upsample(x_pos)
        x_neg = self.conv_up_03_t(x_neg, context)
        x_neg = self.doubleLogNorm(x_neg)
        x_sum = torch.cat([x_pos, x_neg], dim=1)
        x_sum = self.conv_up_03_sum(x_sum)
        x = F.leaky_relu(x_sum, negative_slope=0.1)

        x_pos, x_neg = x, -x
        x_pos = self.conv_up_02_r(x_pos, context)
        x_pos = self.doubleLogNorm(x_pos)
        x_pos = self.upsample(x_pos)
        x_neg = self.conv_up_02_t(x_neg, context)
        x_neg = self.doubleLogNorm(x_neg)
        x_sum = torch.cat([x_pos, x_neg], dim=1)
        x_sum = self.conv_up_02_sum(x_sum)
        x = F.leaky_relu(x_sum, negative_slope=0.1)

        x_pos, x_neg = x, -x
        x_pos = self.conv_up_01_r(x_pos, context)
        x_pos = self.doubleLogNorm(x_pos)
        x_pos = self.upsample(x_pos)
        x_neg = self.conv_up_01_t(x_neg, context)
        x_neg = self.doubleLogNorm(x_neg)
        x_sum = torch.cat([x_pos, x_neg], dim=1)
        x_sum = self.conv_up_01_sum(x_sum)
        x = F.leaky_relu(x_sum, negative_slope=0.1)

        x = self.conv_out(x, context)
        x = F.sigmoid(x)

        return x
