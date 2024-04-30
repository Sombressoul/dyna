import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Union, Optional, Callable, List, Dict

script_dir = os.path.dirname(os.path.abspath(__file__))
evals_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(evals_dir)
sys.path.append(project_dir)

from dyna import DynamicConv2D


class AEDW(nn.Module):
    def __init__(
        self,
        shape_base: Union[List[int], torch.Size] = torch.Size([256, 256]),
        channels_io: int = 3,
        channels_dynamic: int = 32,
        depth: int = 4,
        channels_base: int = 16,
        channels_multiplier: int = 2,
        context_length: int = 16,
        mod_rank: int = 16,
        asymmetry: float = 1.0e-3,
        dropout_rate: float = 0.1,
        eps: float = 1.0e-3,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        assert len(shape_base) == 2, " ".join(
            [
                f"target_shape must be a tuple of 2 elements.",
                f"Got: {shape_base}.",
            ]
        )
        assert shape_base[0] % 2 == 0 and shape_base[1] % 2 == 0, " ".join(
            [
                f"target_shape must be divisible by 2.",
                f"Got: {shape_base}.",
            ]
        )
        assert shape_base[0] == shape_base[1], " ".join(
            [
                f"target_shape must be a square.",
                f"Got: {shape_base}.",
            ]
        )
        assert channels_io > 0, " ".join(
            [
                f"io_channels must be greater than 0.",
                f"Got: {channels_io}.",
            ]
        )
        assert depth > 1, " ".join(
            [
                f"n_depth must be greater than 1.",
                f"Got: {depth}.",
            ]
        )
        assert channels_base > 0, " ".join(
            [
                f"initial_channels must be greater than 0.",
                f"Got: {channels_base}.",
            ]
        )
        assert channels_multiplier > 1, " ".join(
            [
                f"channels_multiplier must be greater than 1.",
                f"Got: {channels_multiplier}.",
            ]
        )
        assert type(channels_multiplier) == int, " ".join(
            [
                f"channels_multiplier must be an integer.",
                f"Got: {channels_multiplier}.",
            ]
        )
        assert channels_dynamic > 0, " ".join(
            [
                f"channels_dynamic_conv must be greater than 0.",
                f"Got: {channels_dynamic}.",
            ]
        )
        assert context_length > 0, " ".join(
            [
                f"context_length must be greater than 0.",
                f"Got: {context_length}.",
            ]
        )
        assert mod_rank > 0, " ".join(
            [
                f"mod_rank must be greater than 0.",
                f"Got: {mod_rank}.",
            ]
        )
        assert asymmetry > 0.0, " ".join(
            [
                f"asymmetry must be greater than 0.0.",
                f"Got: {asymmetry}.",
            ]
        )
        assert eps > 0.0, " ".join(
            [
                f"eps must be greater than 0.0.",
                f"Got: {eps}.",
            ]
        )

        # ================================================================================= #
        # ____________________________> Parameters.
        # ================================================================================= #
        self.shape_base = torch.Size(shape_base)
        self.channels_io = channels_io
        self.depth = depth
        self.channels_base = channels_base
        self.channels_multiplier = channels_multiplier
        self.channels_dynamic = channels_dynamic
        self.context_length = context_length
        self.mod_rank = mod_rank
        self.asymmetry = asymmetry
        self.dropout_rate = dropout_rate
        self.eps = eps
        self.dtype_weights = dtype_weights

        # ================================================================================= #
        # ____________________________> Internal parameters.
        # ================================================================================= #
        self.extractor_channels = [
            self.channels_base * (self.channels_multiplier**i)
            for i in range(self.depth + 1)
        ]
        self.shape_io = [
            self.channels_io,
            self.shape_base[0],
            self.shape_base[1],
        ]
        self.shape_bottleneck = [
            self.channels_base * (self.channels_multiplier**self.depth),
            self.shape_base[0] // (2**self.depth),
            self.shape_base[1] // (2**self.depth),
        ]

        # ================================================================================= #
        # ____________________________> Init submodules.
        # ================================================================================= #
        # self._create_context_extractor()
        self._create_encoder()
        self._create_decoder()

        pass

    def _create_context_extractor(
        self,
    ) -> None:
        self.extractor_head = nn.Sequential(
            *nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=self.channels_io,
                        out_channels=self.channels_base,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dtype=self.dtype_weights,
                    ),
                    nn.PReLU(
                        num_parameters=self.channels_base,
                        init=2.5e-1,
                        dtype=self.dtype_weights,
                    ),
                    nn.BatchNorm2d(
                        num_features=self.channels_base,
                        momentum=0.1,
                        affine=True,
                        eps=self.eps,
                        dtype=self.dtype_weights,
                    ),
                    *[
                        self._create_context_extractor_head_block(
                            channels_in=self.extractor_channels[i],
                            channels_out=self.extractor_channels[i + 1],
                        )
                        for i in range(self.depth)
                    ],
                ]
            )
        )

        self.extractor_body_01_linear = nn.Linear(
            in_features=self.shape_bottleneck[0],
            out_features=self.context_length,
            dtype=self.dtype_weights,
        )
        self.extractor_body_01_activation = nn.PReLU(
            num_parameters=self.context_length,
            init=2.5e-1,
            dtype=self.dtype_weights,
        )
        self.extractor_body_02_linear = nn.Linear(
            in_features=self.shape_bottleneck[0] + self.context_length,
            out_features=self.context_length,
            dtype=self.dtype_weights,
        )
        self.extractor_body_02_activation = nn.PReLU(
            num_parameters=self.context_length,
            init=2.5e-1,
            dtype=self.dtype_weights,
        )
        self.extractor_body_03_linear = nn.Linear(
            in_features=self.shape_bottleneck[0],
            out_features=self.context_length,
            dtype=self.dtype_weights,
        )
        self.extractor_body_03_activation = nn.Tanh()
        self.extractor_body_03_norm = nn.LayerNorm(
            normalized_shape=math.prod(self.shape_bottleneck[-2:]),
            dtype=self.dtype_weights,
        )

        context_numel = self.context_length * self.depth * 2 + (self.context_length * 2)
        self.extractor_tail = nn.Sequential(
            *nn.ModuleList(
                [
                    nn.Linear(
                        in_features=math.prod(self.shape_bottleneck[-2:]),
                        out_features=context_numel,
                        dtype=self.dtype_weights,
                    ),
                    nn.PReLU(
                        num_parameters=context_numel,
                        init=2.5e-1,
                        dtype=self.dtype_weights,
                    ),
                    nn.Dropout1d(
                        p=self.dropout_rate,
                    ),
                    nn.Linear(
                        in_features=context_numel,
                        out_features=context_numel,
                        dtype=self.dtype_weights,
                    ),
                    nn.Tanh(),
                    nn.LayerNorm(
                        normalized_shape=context_numel,
                        dtype=self.dtype_weights,
                    ),
                ]
            )
        )

        pass

    def _create_context_extractor_head_block(
        self,
        channels_in: int,
        channels_out: int,
    ) -> nn.Module:
        block = nn.Sequential(
            *nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=channels_in,
                        out_channels=channels_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        dtype=self.dtype_weights,
                    ),
                    nn.PReLU(
                        num_parameters=channels_out,
                        init=2.5e-1,
                        dtype=self.dtype_weights,
                    ),
                    nn.BatchNorm2d(
                        num_features=channels_out,
                        momentum=0.1,
                        affine=True,
                        eps=self.eps,
                        dtype=self.dtype_weights,
                    ),
                    nn.Conv2d(
                        in_channels=channels_out,
                        out_channels=channels_out,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                        dtype=self.dtype_weights,
                    ),
                    nn.PReLU(
                        num_parameters=channels_out,
                        init=2.5e-1,
                        dtype=self.dtype_weights,
                    ),
                    nn.BatchNorm2d(
                        num_features=channels_out,
                        momentum=0.1,
                        affine=True,
                        eps=self.eps,
                        dtype=self.dtype_weights,
                    ),
                ]
            ),
        )
        return block

    def _create_encoder(
        self,
    ) -> None:
        self.encoder_head_conv = DynamicConv2D(
            in_channels=self.channels_io,
            out_channels=self.channels_dynamic,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            bias_dynamic=True,
            transpose=False,
            output_padding=None,
            asymmetry=self.asymmetry,
            dtype_weights=self.dtype_weights,
        )
        self.encoder_head_norm = nn.BatchNorm2d(
            num_features=self.channels_dynamic,
            momentum=0.1,
            affine=True,
            eps=self.eps,
            dtype=self.dtype_weights,
        )

        for layer_index in range(self.depth):
            name = f"encoder_level_{layer_index:02d}"

            self.register_module(
                name=f"{name}_conv",
                module=DynamicConv2D(
                    in_channels=self.channels_dynamic,
                    out_channels=self.channels_dynamic,
                    context_length=self.context_length,
                    mod_rank=self.mod_rank,
                    kernel_size=[3, 3],
                    stride=[2, 2],
                    padding=[1, 1],
                    dilation=[1, 1],
                    bias_dynamic=True,
                    transpose=False,
                    output_padding=None,
                    asymmetry=self.asymmetry,
                    dtype_weights=self.dtype_weights,
                ),
            )
            self.register_module(
                name=f"{name}_norm",
                module=nn.BatchNorm2d(
                    num_features=self.channels_dynamic,
                    momentum=0.1,
                    affine=True,
                    eps=self.eps,
                    dtype=self.dtype_weights,
                ),
            )
        pass

    def _create_decoder(
        self,
    ) -> None:
        for layer_index in range(self.depth):
            name = f"decoder_level_{layer_index:02d}"

            self.register_module(
                name=f"{name}_conv",
                module=DynamicConv2D(
                    in_channels=self.channels_dynamic,
                    out_channels=self.channels_dynamic,
                    context_length=self.context_length,
                    mod_rank=self.mod_rank,
                    kernel_size=[3, 3],
                    stride=[2, 2],
                    padding=[1, 1],
                    dilation=[1, 1],
                    bias_dynamic=True,
                    transpose=True,
                    output_padding=[1, 1],
                    asymmetry=self.asymmetry,
                    dtype_weights=self.dtype_weights,
                ),
            )
            self.register_module(
                name=f"{name}_norm",
                module=nn.BatchNorm2d(
                    num_features=self.channels_dynamic,
                    momentum=0.1,
                    affine=True,
                    eps=self.eps,
                    dtype=self.dtype_weights,
                ),
            )

        self.decoder_tail_conv = nn.Conv2d(
            in_channels=self.channels_dynamic,
            out_channels=self.channels_io,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=self.dtype_weights,
        )

        pass

    def extract_context(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.extractor_head(x)
        x = x.permute([0, 2, 3, 1])

        x_a = self.extractor_body_01_linear(x)
        x_a = x_a.permute([0, 3, 1, 2])
        x_a = self.extractor_body_01_activation(x_a)
        x_a = x_a.permute([0, 2, 3, 1])
        x_a = torch.cat([x, x_a], dim=-1)
        x_a = self.extractor_body_02_linear(x_a)
        x_a = x_a.permute([0, 3, 1, 2])
        x_a = self.extractor_body_02_activation(x_a)
        x_a = x_a.flatten(-2)
        x_a = F.softmax(x_a, dim=-1, dtype=self.dtype_weights)
        x_a = x_a.permute([0, 2, 1])

        x_b = self.extractor_body_03_linear(x)
        x_b = x_b.permute([0, 3, 1, 2])
        x_b = self.extractor_body_03_activation(x_b)
        x_b = x_b.flatten(-2)
        x_b = self.extractor_body_03_norm(x_b)
        x_b = x_b.permute([0, 2, 1])

        x_c = x_a * x_b
        x_c = x_c.permute([0, 2, 1])
        x_c = F.softmax(x_c, dim=-1, dtype=self.dtype_weights)
        x_c = x_c.permute([0, 2, 1])
        x_c = x_c.sum(-1)

        x = self.extractor_tail(x_c)
        x = x.reshape(x.shape[0], 2, self.depth + 1, self.context_length)
        return x

    def encode(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        x = self.encoder_head_conv(x, context[:, 0, -1, :])
        x = self.encoder_head_norm(x)

        for layer_index in range(self.depth):
            name = f"encoder_level_{layer_index:02d}"
            x = getattr(self, name + "_conv")(x, context[:, 0, layer_index, :])
            x = getattr(self, name + "_norm")(x)
        return x

    def decode(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        for layer_index in range(self.depth):
            name = f"decoder_level_{layer_index:02d}"
            x = getattr(self, name + "_conv")(x, context[:, 1, layer_index, :])
            x = getattr(self, name + "_norm")(x)

        x = self.decoder_tail_conv(x)
        return x

    def _log_var(
        self,
        x: torch.Tensor,
        name: str,
        is_breakpoint: bool = False,
    ) -> None:
        mem = (x.element_size() * x.nelement()) / (1024 * 1024)
        unk = "<UNKNOWN>"

        print(f"\n")
        print(f"# =====> Name: {name if name is not None else unk}")
        print(f"# =====> Memory: {mem:.2f} MB")
        print(f"# =====> Elements: {x.numel():_}")
        print(f"{x.shape=}")
        print(f"{x.min()=}")
        print(f"{x.max()=}")
        print(f"{x.mean()=}")
        print(f"{x.std()=}")
        print(f"\n")

        if is_breakpoint:
            exit()

        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        assert x.shape[2::] == self.shape_base, " ".join(
            [
                f"Wrong input shape.",
                f"Expected: {torch.Size([1, self.channels_io, *self.shape_base])}.",
                f"Got: {x.shape}.",
            ]
        )

        # ================================================================================= #
        # ____________________________> Casting.
        # ================================================================================= #
        x = x.to(dtype=self.dtype_weights)

        # ================================================================================= #
        # ____________________________> Processing.
        # ================================================================================= #
        # context = self.extract_context(x)
        try:
            context = self.context
        except AttributeError:
            self.context = nn.Parameter(
                data=torch.randn(
                    [x.shape[0], 2, self.depth + 1, self.context_length],
                    dtype=self.dtype_weights,
                    device=x.device,
                ),
            )
            context = self.context

        if torch.isnan(context).any() or torch.isinf(context).any():
            self._log_var(context, "context")
            raise ValueError("context has NaN or Inf elements.")

        encoded = self.encode(x, context)

        if torch.isnan(encoded).any() or torch.isinf(encoded).any():
            self._log_var(encoded, "encoded")
            raise ValueError("encoded has NaN or Inf elements.")

        decoded = self.decode(encoded, context)

        if torch.isnan(decoded).any() or torch.isinf(decoded).any():
            self._log_var(decoded, "decoded")
            raise ValueError("decoded has NaN or Inf elements.")

        return decoded, encoded, context
