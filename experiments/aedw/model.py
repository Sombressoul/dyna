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

from dyna.module import DynamicConv2D


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
        # TODO: ...

        pass

    def extract_context(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: ...
        return x

    def encode(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: ...
        return x

    def decode(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: ...
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
        context = ...
        encoded = ...
        decoded = ...
        
        return decoded, encoded, context
