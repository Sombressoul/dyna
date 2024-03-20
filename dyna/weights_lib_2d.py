import torch
import torch.nn as nn
import math

from typing import Union


class WeightsLib2D(nn.Module):
    def __init__(
        self,
        shape: Union[torch.Size, list[int]],
        rank: int = 8,
        dtype: torch.dtype = torch.float32,
        complex_result: bool = False,
    ) -> None:
        super().__init__()

        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        shape = torch.Size(shape) if type(shape) == list else shape
        assert len(shape) == 2, "Shape must be 2D."
        assert rank > 0, "Rank must be greater than 0."
        assert dtype in [
            torch.float32,
            torch.float64,
        ], "dtype must be float32 or float64."

        # ================================================================================= #
        # ____________________________> Parameters.
        # ================================================================================= #
        self.shape = shape
        self.rank = rank
        self.dtype = dtype
        self.complex_result = complex_result

        # ================================================================================= #
        # ____________________________> Weights.
        # ================================================================================= #
        self.weights_i = nn.Parameter(
            data=self._create_weights_base([self.rank, self.shape[0]]),
        )
        self.weights_j = nn.Parameter(
            data=self._create_weights_base([self.rank, self.shape[1]]),
        )

        pass

    def _create_weights_base(
        self,
        shape: Union[torch.Size, list[int]],
    ) -> torch.Tensor:
        base_r = torch.empty(shape, dtype=self.dtype)
        base_r = nn.init.uniform_(
            tensor=base_r,
            a=-math.sqrt(2.0 / nn.init._calculate_correct_fan(base_r, "fan_in")),
            b=+math.sqrt(2.0 / nn.init._calculate_correct_fan(base_r, "fan_in")),
        )
        base_i = torch.empty_like(base_r)
        base_i = nn.init.uniform_(
            tensor=base_i,
            a=-math.sqrt((math.pi * 2) / math.prod(self.shape)),
            b=+math.sqrt((math.pi * 2) / math.prod(self.shape)),
        )
        base = torch.complex(
            real=base_r,
            imag=base_i,
            dtype=torch.complex64 if self.dtype == torch.float32 else torch.complex128,
        )

        return base

    def _create_weights_controls(
        self,
    ) -> torch.Tensor:
        bias = nn.init.normal_(
            tensor=torch.empty([2, self.rank, 1], dtype=self.dtype),
            mean=0.0,
            std=1.0 / self.rank,
        )
        scale = nn.init.normal_(
            tensor=torch.empty([2, self.rank, 1], dtype=self.dtype),
            mean=1.0,
            std=1.0 / self.rank,
        )

        controls_r = torch.cat([bias, scale], dim=-1)
        controls_i = torch.empty_like(controls_r)
        controls_i = nn.init.uniform_(
            tensor=controls_i,
            a=-math.sqrt((math.pi * 2) / math.prod(self.shape)),
            b=+math.sqrt((math.pi * 2) / math.prod(self.shape)),
        )
        controls = torch.complex(
            real=controls_r,
            imag=controls_i,
            dtype=torch.complex64 if self.dtype == torch.float32 else torch.complex128,
        )

        return controls

    def _get_weights(
        self,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        weights_i = (self.weights_i + controls[0, ..., 0].unsqueeze(-1)) * controls[
            0, ..., 1
        ].unsqueeze(-1)
        weights_j = (self.weights_j + controls[1, ..., 0].unsqueeze(-1)) * controls[
            1, ..., 1
        ].unsqueeze(-1)

        return torch.einsum("ki,kj->ij", weights_i, weights_j)

    def _get_controls(
        self,
        name: str,
        force_create: bool = False,
    ) -> torch.Tensor:
        try:
            controls = self.get_parameter(f"weight_controls_{name}")
        except AttributeError:
            controls = self._create_weights_controls()
        # TODO: implement.
        ...

    def get_weights(
        self,
        name: str,
    ) -> torch.Tensor:
        # TODO: implement.
        ...