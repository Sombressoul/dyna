import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Union, List

from dyna.weights_lib_2d_lite import WeightsLib2DLite


class DynamicConv2D(nn.Module):
    dtypes: List[torch.dtype] = [
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
    ]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_length: int,
        mod_rank: int,
        kernel_size: Union[int, List[int]] = [3, 3],
        stride: Union[int, List[int]] = [1, 1],
        padding: Union[int, List[int]] = [0, 0],
        dilation: Union[int, List[int]] = [1, 1],
        bias: bool = True,
        asymmetry: float = 1.0e-3,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        if type(kernel_size) == int:
            kernel_size = [kernel_size, kernel_size]
        if type(stride) == int:
            stride = [stride, stride]
        if type(padding) == int:
            padding = [padding, padding]
        if type(dilation) == int:
            dilation = [dilation, dilation]

        assert context_length > 0, "context_length must be greater than 0"
        assert mod_rank > 0, "mod_rank must be greater than 0"
        assert len(kernel_size) == 2, "kernel_size must be an int or a 2-element tuple"
        assert len(stride) == 2, "stride must be an int or a 2-element tuple"
        assert len(padding) == 2, "padding must be an int or a 2-element tuple"
        assert len(dilation) == 2, "dilation must be an int or a 2-element tuple"
        assert (
            in_channels > 0
            and out_channels > 0
            and kernel_size[0] > 0
            and kernel_size[1] > 0
        ), "in_channels, out_channels, and kernel_size must be greater than 0"
        assert stride[0] > 0 and stride[1] > 0, "stride must be greater than 0"
        assert (
            padding[0] >= 0 and padding[1] >= 0
        ), "padding must be greater than or equal to 0"
        assert dilation[0] > 0 and dilation[1] > 0, "dilation must be greater than 0"
        assert dtype_weights in self.dtypes, f"dtype must be one of {self.dtypes}"

        # ================================================================================= #
        # ____________________________> Parameters.
        # ================================================================================= #
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_length = context_length
        self.mod_rank = mod_rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias
        self.asymmetry = asymmetry
        self.dtype_weights = dtype_weights

        # ================================================================================= #
        # ____________________________> Calculate weights shapes and indexing params.
        # ================================================================================= #
        self.conv_weights_shape = [
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        ]
        self.dynamic_weights_shape = self._fit_shape(
            [
                self.conv_weights_shape[0] * self.conv_weights_shape[2],
                self.conv_weights_shape[1] * self.conv_weights_shape[3],
            ]
        )
        self.dynamic_weights_inversion = (
            self.dynamic_weights_shape[0] < self.dynamic_weights_shape[1]
        )
        self.dynamic_weights_shape = (
            self.dynamic_weights_shape[-1::-1]
            if self.dynamic_weights_inversion
            else self.dynamic_weights_shape
        )
        self.dynamic_weights_index = self._create_index()
        self.dynamic_weights_index.requires_grad = False

        # ================================================================================= #
        # ____________________________> Init submodules and additional weights.
        # ================================================================================= #
        self.weights_lib = WeightsLib2DLite(
            output_shape=self.dynamic_weights_shape,
            components_count=self.context_length,
            mod_rank=self.mod_rank,
            asymmetry=self.asymmetry,
            dtype_weights=self.dtype_weights,
        )
        self.bias_dynamic = (
            nn.Parameter(
                data=nn.init.uniform_(
                    tensor=torch.empty(
                        [1, *self.dynamic_weights_shape],
                        dtype=self.dtype_weights,
                    ),
                    a=-self.asymmetry,
                    b=+self.asymmetry,
                )
            )
            if self.use_bias
            else None
        )
        self.bias_conv = (
            nn.Parameter(
                data=nn.init.uniform_(
                    tensor=torch.empty(
                        [self.out_channels],
                        dtype=self.dtype_weights,
                    ),
                    a=-math.sqrt(1 / (self.in_channels * math.prod(self.kernel_size))),
                    b=+math.sqrt(1 / (self.in_channels * math.prod(self.kernel_size))),
                )
            )
            if self.use_bias
            else None
        )

        pass

    def _fit_shape(
        self,
        shape: List[int],
    ) -> List[int]:
        assert len(shape) == 2, "Shape must be a 2-element tuple"
        assert shape[0] > 0 and shape[1] > 0, " ".join(
            [
                "Shape elements must be greater than 0.",
                f"Got: {shape}",
            ]
        )

        new_shape = []

        if (
            max(shape) % 2 == 0
            and max(shape) / 2 > min(shape)
            and (
                (
                    (max(shape) // 2) % self.kernel_size[0] == 0
                    and (min(shape) ** 2) % self.kernel_size[1] == 0
                )
                or (
                    (max(shape) // 2) % self.kernel_size[1] == 0
                    and (min(shape) ** 2) % self.kernel_size[0] == 0
                )
            )
        ):
            new_shape.append(int(max(shape) // 2))
            new_shape.append(int(min(shape) * 2))
            return self._fit_shape(new_shape)
        else:
            return shape

    def _create_index(
        self,
    ) -> torch.Tensor:
        base_ids = torch.arange(
            start=0,
            end=math.prod(self.conv_weights_shape),
            step=1,
            dtype=torch.int64,
        ).reshape(self.dynamic_weights_shape)

        axis_i = 1 if self.dynamic_weights_inversion else 0
        axis_j = 0 if self.dynamic_weights_inversion else 1
        num_blocks_i = self.dynamic_weights_shape[axis_i] // self.kernel_size[0]
        num_blocks_j = self.dynamic_weights_shape[axis_j] // self.kernel_size[1]
        block_div = num_blocks_i if self.dynamic_weights_inversion else num_blocks_j
        num_blocks = num_blocks_i * num_blocks_j

        index = torch.empty(self.conv_weights_shape, dtype=torch.int64)

        for block_id in range(num_blocks):
            row_from = block_id // block_div * self.kernel_size[axis_i]
            row_to = row_from + self.kernel_size[axis_i]
            col_from = block_id % block_div * self.kernel_size[axis_j]
            col_to = col_from + self.kernel_size[axis_j]
            block = base_ids[row_from:row_to, col_from:col_to].unsqueeze(0).unsqueeze(0)
            block = block.transpose(-1, -2) if self.dynamic_weights_inversion else block
            out_channel = block_id % self.out_channels
            in_channel = block_id // self.out_channels
            index[out_channel, in_channel, :, :] = block

        index = index.reshape(self.dynamic_weights_shape).contiguous()

        return index

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if context.shape[0] != 1 and context.shape[0] != x.shape[0]:
            raise ValueError(
                " ".join(
                    [
                        "context.shape[0] must be equal to 1 or x.shape[0].",
                        f"Got: {context.shape[0]=} and {x.shape[0]=}",
                    ]
                )
            )

        with torch.no_grad():
            dynamic_weights_index = self.dynamic_weights_index
            dynamic_weights_index = dynamic_weights_index.unsqueeze(0).repeat(
                [
                    x.shape[0],
                    *[1 for _ in range(len(dynamic_weights_index.shape))],
                ]
            )
            index_shifts = torch.arange(
                0,
                self.dynamic_weights_index.numel() * x.shape[0],
                self.dynamic_weights_index.numel(),
            ).reshape(
                [x.shape[0], *[1 for _ in range(len(dynamic_weights_index.shape) - 1)]]
            )
            dynamic_weights_index = dynamic_weights_index + index_shifts

        dynamic_weights = self.weights_lib(context)
        dynamic_weights = (
            dynamic_weights + self.bias_dynamic.unsqueeze(0)
            if self.use_bias
            else dynamic_weights
        )
        dynamic_weights = (
            dynamic_weights.repeat(
                [
                    x.shape[0],
                    *[1 for _ in range(len(dynamic_weights.shape) - 1)],
                ]
            )
            if dynamic_weights.shape[0] != x.shape[0]
            else dynamic_weights
        )
        dynamic_weights = torch.take(
            input=dynamic_weights,
            index=dynamic_weights_index,
        )
        conv_weights = dynamic_weights.reshape([x.shape[0], *self.conv_weights_shape])

        wrapped_conv = lambda x, w: F.conv2d(
            input=x,
            weight=w,
            bias=self.bias_conv,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        batched_conv = torch.vmap(wrapped_conv)
        x = batched_conv(x.unsqueeze(1), conv_weights)
        x = x.squeeze(1)

        return x
