import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Union, List, Optional

from dyna.lib.weights_lib_2d_alpha import WeightsLib2DAlpha


class DynamicConv2DAlpha(nn.Module):
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
        context_rank: int,
        context_use_bias: bool = False,
        context_conv_use_bias: bool = False,
        context_dropout_rate: float = 0.0,
        kernel_size: Union[int, List[int]] = [3, 3],
        stride: Union[int, List[int]] = [1, 1],
        padding: Union[int, List[int]] = [0, 0, 0, 0],
        padding_dynamic: bool = True,
        dilation: Union[int, List[int]] = [1, 1],
        transpose: bool = False,
        output_padding: Optional[Union[int, List[int]]] = None,
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
            padding = [
                padding,
                padding,
                padding,
                padding,
            ]
        if type(output_padding) == int:
            output_padding = [
                output_padding,
                output_padding,
            ]
        if type(dilation) == int:
            dilation = [dilation, dilation]

        assert context_length > 0, "context_length must be greater than 0."
        assert context_rank > 0, "mod_rank must be greater than 0."
        assert len(kernel_size) == 2, "kernel_size must be an int or a 2-element tuple."
        assert len(stride) == 2, "stride must be an int or a 2-element tuple."
        assert len(dilation) == 2, "dilation must be an int or a 2-element tuple."
        assert (
            in_channels > 0
            and out_channels > 0
            and kernel_size[0] > 0
            and kernel_size[1] > 0
        ), "in_channels, out_channels, and kernel_size must be greater than 0."
        assert stride[0] > 0 and stride[1] > 0, "stride must be greater than 0."
        assert len(padding) == 4, "padding must an int or a 4-element tuple."
        assert (
            padding[0] >= 0
            and padding[1] >= 0
            and padding[2] >= 0
            and padding[3] >= 0
        ), "padding must be greater than or equal to 0."
        assert dilation[0] > 0 and dilation[1] > 0, "dilation must be greater than 0."
        assert (
            dtype_weights in self.dtypes
        ), f"dtype_weights must be one of {self.dtypes}."

        if transpose:
            assert (
                output_padding is not None
            ), "output_padding must be specified if transposed."
            assert (
                len(output_padding) == 2
            ), "output_padding must be an int or a 2-element tuple."
            assert (
                output_padding[0] >= 0 and output_padding[1] >= 0
            ), "output_padding must be greater than or equal to 0."
        else:
            assert (
                output_padding is None
            ), "output_padding must be None if not transposed."

        # ================================================================================= #
        # ____________________________> Parameters.
        # ================================================================================= #
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_length = context_length
        self.context_rank = context_rank
        self.context_use_bias = context_use_bias
        self.context_conv_use_bias = context_conv_use_bias
        self.context_dropout_rate = context_dropout_rate
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_dynamic = padding_dynamic
        self.output_padding = output_padding
        self.dilation = dilation
        self.transpose = transpose
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
        self.weights_lib = WeightsLib2DAlpha(
            output_shape=self.dynamic_weights_shape,
            context_length=self.context_length,
            context_rank=self.context_rank,
            context_use_bias=self.context_use_bias,
            context_conv_use_bias=self.context_conv_use_bias,
            context_dropout_rate=self.context_dropout_rate,
            dtype_weights=self.dtype_weights,
        )
        self.padding_dynamic_value = (
            nn.Parameter(
                data=nn.init.normal_(
                    tensor=torch.empty(
                        [1],
                        dtype=self.dtype_weights,
                    ),
                    mean=0.0,
                    std=1.0e-3,
                )
            )
            if self.padding_dynamic
            else None
        )

        pass

    def _fit_shape(
        self,
        shape: List[int],
    ) -> List[int]:
        assert len(shape) == 2, "Shape must be a 2-element tuple."
        assert shape[0] > 0 and shape[1] > 0, " ".join(
            [
                "Shape elements must be greater than 0.",
                f"Got: {shape}.",
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

    def get_weights(
        self,
        context: torch.Tensor,
        batch_dim: Optional[int] = None,
    ) -> torch.Tensor:
        batch_dim = batch_dim if batch_dim is not None else context.shape[0]

        if batch_dim < 1:
            raise ValueError(
                " ".join(
                    [
                        "batch_dim must be greater than or equal to 1.",
                        f"Got: {batch_dim}.",
                    ]
                )
            )

        with torch.no_grad():
            if self.dynamic_weights_index.device != context.device:
                self.dynamic_weights_index = self.dynamic_weights_index.to(
                    context.device
                )

            dynamic_weights_index = self.dynamic_weights_index
            dynamic_weights_index = dynamic_weights_index.unsqueeze(0).repeat(
                [
                    batch_dim,
                    *[1 for _ in range(len(dynamic_weights_index.shape))],
                ]
            )
            index_shifts = torch.arange(
                0,
                self.dynamic_weights_index.numel() * batch_dim,
                self.dynamic_weights_index.numel(),
                device=context.device,
            ).reshape(
                [batch_dim, *[1 for _ in range(len(dynamic_weights_index.shape) - 1)]]
            )
            dynamic_weights_index = dynamic_weights_index + index_shifts

        dynamic_weights = self.weights_lib(context)
        dynamic_weights = (
            dynamic_weights.repeat(
                [
                    batch_dim,
                    *[1 for _ in range(len(dynamic_weights.shape) - 1)],
                ]
            )
            if dynamic_weights.shape[0] != batch_dim
            else dynamic_weights
        )
        dynamic_weights = torch.take(
            input=dynamic_weights,
            index=dynamic_weights_index,
        )
        conv_weights = dynamic_weights.reshape([batch_dim, *self.conv_weights_shape])
        conv_weights = conv_weights.transpose(1, 2) if self.transpose else conv_weights

        return conv_weights

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
                        f"Got: {context.shape[0]=} and {x.shape[0]=}.",
                    ]
                )
            )

        if self.transpose:
            wrapped_fn = lambda x, w: F.conv_transpose2d(
                input=x,
                weight=w,
                bias=None,
                stride=self.stride,
                padding=[0, 0],
                output_padding=self.output_padding,
                dilation=self.dilation,
            )
        else:
            wrapped_fn = lambda x, w: F.conv2d(
                input=x,
                weight=w,
                bias=None,
                stride=self.stride,
                padding=[0, 0],
                dilation=self.dilation,
            )

        if self.padding_dynamic_value:
            x_padded_ones = F.pad(
                input=x,
                pad=(
                    self.padding
                    if len(self.padding) == 4
                    else [
                        self.padding[0],
                        self.padding[0],
                        self.padding[1],
                        self.padding[1],
                    ]
                ),
                mode="constant",
                value=1.0,
            )
            x_padded_zeros = F.pad(
                input=x,
                pad=(
                    self.padding
                    if len(self.padding) == 4
                    else [
                        self.padding[0],
                        self.padding[0],
                        self.padding[1],
                        self.padding[1],
                    ]
                ),
                mode="constant",
                value=1.0,
            )
            x_padding = (x_padded_ones - x_padded_zeros) * self.padding_dynamic_value
            x = x_padded_zeros + x_padding
        else:
            x = F.pad(
                input=x,
                pad=self.padding,
                mode="constant",
                value=self.padding_dynamic_value if self.padding_dynamic else None,
            )

        batched_fn = torch.vmap(wrapped_fn)
        x = batched_fn(
            x.unsqueeze(1),
            self.get_weights(
                context=context,
                batch_dim=x.shape[0],
            ),
        )
        x = x.squeeze(1)
        
        return x
