import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dyna.module import DynamicConv2D


class DWDecoder(nn.Module):
    def __init__(
        self,
        data_cache_ctx_len: int,
        data_cache_latents_len: int,
        data_cache_latents_shape: list[int],
        channels_levels: list[int],
        channels_io: int = 3,
        use_bias: bool = True,
        bias_static: float = 0.0,
        context_length: int = 32,
        mod_rank: int = 16,
        kernel_size_t: list[int] = [4, 4],
        kernel_size_r: list[int] = [3, 3],
        kernel_size_m: list[int] = [5, 5],
        stride_t: list[int] = [2, 2],
        stride_r: list[int] = [1, 1],
        stride_m: list[int] = [1, 1],
        padding_t: list[int] = [1, 1],
        padding_r: list[int] = [1, 1],
        padding_m: list[int] = [2, 2],
        dilation_t: list[int] = [1, 1],
        dilation_r: list[int] = [1, 1],
        dilation_m: list[int] = [1, 1],
        output_padding_t: list[int] = [0, 0],
        eps: float = 1.0e-2,
        discretization_levels: int = 32,
        discretization_scale: float = math.pi / 2,
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
        self.kernel_size_m = kernel_size_m
        self.stride_t = stride_t
        self.stride_r = stride_r
        self.stride_m = stride_m
        self.padding_t = padding_t
        self.padding_r = padding_r
        self.padding_m = padding_m
        self.dilation_t = dilation_t
        self.dilation_r = dilation_r
        self.dilation_m = dilation_m
        self.output_padding_t = output_padding_t
        self.eps = eps
        self.discretization_levels = discretization_levels
        self.discretization_scale = discretization_scale
        self.channels_io = channels_io
        self.channels_levels = channels_levels
        self.dtype_weights = dtype_weights

        # ================================================================================= #
        # ____________________________> Common submodules.
        # ================================================================================= #
        self.upsample_nearest = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample_bilinear = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv_out = DynamicConv2D(
            in_channels=self.channels_levels[0],
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
        # ____________________________> Block submodules.
        # ================================================================================= #
        # TODO: create decoder blocks.
        self._create_decoder()

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

    def _create_decoder(self) -> None:
        # self.conv_up_04_t = DynamicConv2D(
        #     in_channels=self.channels_dynamic_levels[4],
        #     out_channels=self.channels_dynamic_levels[3],
        #     context_length=self.context_length,
        #     mod_rank=self.mod_rank,
        #     kernel_size=self.kernel_size_t,
        #     stride=[2, 2],
        #     padding=[1, 1],
        #     dilation=[1, 1],
        #     bias_dynamic=self.use_bias,
        #     bias_static=self.bias_static,
        #     transpose=True,
        #     output_padding=[0, 0],
        #     asymmetry=1.0e-3,
        #     dtype_weights=self.dtype_weights,
        # )
        # self.conv_up_04_r = DynamicConv2D(
        #     in_channels=self.channels_dynamic_levels[4],
        #     out_channels=self.channels_dynamic_levels[3],
        #     context_length=self.context_length,
        #     mod_rank=self.mod_rank,
        #     kernel_size=self.kernel_size_r,
        #     stride=[1, 1],
        #     padding=[1, 1],
        #     dilation=[1, 1],
        #     bias_dynamic=self.use_bias,
        #     bias_static=self.bias_static,
        #     transpose=False,
        #     output_padding=None,
        #     asymmetry=1.0e-3,
        #     dtype_weights=self.dtype_weights,
        # )
        # self.conv_up_04_m = DynamicConv2D(
        #     in_channels=self.channels_dynamic_levels[4],
        #     out_channels=self.channels_dynamic_levels[3],
        #     context_length=self.context_length,
        #     mod_rank=self.mod_rank,
        #     kernel_size=self.kernel_size_m,
        #     stride=[1, 1],
        #     padding=[2, 2],
        #     dilation=[1, 1],
        #     bias_dynamic=self.use_bias,
        #     bias_static=self.bias_static,
        #     transpose=False,
        #     output_padding=None,
        #     asymmetry=1.0e-3,
        #     dtype_weights=self.dtype_weights,
        # )
        raise NotImplementedError("TODO: transfer implementation from the test code.")
    
    def _block_process(self) -> torch.Tensor:
        # x_pos, x_neg, x_mul = x, -x, x.abs().add(1.0).log()
        # x_pos = self.conv_up_04_r(x_pos, context)
        # x_pos = self.doubleLogNorm(x_pos)
        # x_pos = self.upsample_nearest(x_pos)
        # x_neg = self.conv_up_04_t(x_neg, context)
        # x_neg = self.doubleLogNorm(x_neg)
        # x_mul = self.upsample_bilinear(x_mul)
        # x_mul = self.conv_up_04_m(x_mul, context)
        # x_mul = self.doubleLogNorm(x_mul)
        # x = (x_pos + x_neg) * x_mul
        # x = torch.arctan(x)
        # x = self.quantizer(x)
        raise NotImplementedError("TODO: transfer implementation from the test code.")


    def get_regterm_model(
        self,
        alpha: float = 2.5e-7,
    ) -> torch.Tensor:
        sum = 0.0
        for name, param in self.named_parameters():
            if "data_cache" in name:
                continue
            else:
                sum = sum + (param**2).sum()
        return alpha * sum

    def get_regterm_ctx(
        self,
        alpha: float = 2.5e-4,
    ) -> torch.Tensor:
        sum = 0.0
        for name, param in self.named_parameters():
            if "data_cache_ctx" in name:
                sum = sum + (param**2).sum()
        return alpha * (sum / self.data_cache_ctx.shape[0])

    def get_regterm_latents(
        self,
        alpha: float = 2.0e-6,
    ) -> torch.Tensor:
        sum = 0.0
        for name, param in self.named_parameters():
            if "data_cache_latents" in name:
                sum = sum + (param**2).sum()
        return alpha * (sum / self.data_cache_latents.shape[0])

    def get_regterm_underweight_model(
        self,
        bound: float = 1.0e-3,
        alpha: float = 1.0e-3,
    ) -> torch.Tensor:
        sum = 0.0
        for name, param in self.named_parameters():
            if "data_cache" in name:
                continue
            vars = param.abs().clamp(0.0, bound).sub(bound).abs()
            sum = sum + vars.sum()
        return alpha * sum

    def weights_hysteresis_loop(
        self,
        bound: float = 1.0e-3,
        jump: float = 2.5e-3,
    ) -> None:
        with torch.no_grad():
            for name, param in self.named_parameters():
                if "data_cache" in name:
                    continue
                cond_to_neg = (param > -0.0) & (param < +bound)
                cond_to_pos = (param < +0.0) & (param > -bound)
                if cond_to_neg.any():
                    param[cond_to_neg] = param[cond_to_neg] - jump
                if cond_to_pos.any():
                    param[cond_to_pos] = param[cond_to_pos] + jump

    def _util_count_vals(
        self,
        target: float,
        excl: str = None,
        incl: str = None,
        abs: bool = True,
        mode: str = "gt",
    ) -> int:
        assert mode in ["gt", "lt", "eq"], f"mode must be 'gt' or 'lt', got {mode}"
        cnt_fn = (
            lambda x, acc: acc
            + x[
                torch.where(
                    ((x.abs() if abs else x) < target)
                    if mode == "lt"
                    else (
                        ((x.abs() if abs else x) > target)
                        if mode == "gt"
                        else ((x.abs() if abs else x) == target)
                    )
                )
            ].numel()
        )
        cnt = 0
        for name, param in self.named_parameters():
            if excl is not None and excl in name:
                continue
            if incl is not None:
                cnt = cnt_fn(param, cnt) if incl in name else cnt
            else:
                cnt = cnt_fn(param, cnt)
        return cnt

    def _util_count_params(
        self,
    ) -> int:
        sum = 0
        for name, param in self.named_parameters():
            if "data_cache" in name:
                continue
            sum = sum + param.numel()
        return sum

    def discretize(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            l = self.discretization_levels
            s = self.discretization_scale
            shift = 1.0 / (l * 2)
            x_q = x / s
            x_q = (x_q * l).clamp(-l, +l)
            x_q = (x_q // 1.0).to(dtype=x.dtype, device=x.device)
            x_q = (x_q / l) + shift
            x_q = x_q * s
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
        #
        # ########################################################## #

        raise NotImplementedError("TODO: transfer implementation from the test code.")

        x = self.conv_out(x, context)
        x = F.sigmoid(x)

        return x
