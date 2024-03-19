import torch
import torch.nn as nn

from typing import Union


class WeightsLib2D(nn.Module):
    def __init__(
        self,
        shape: Union[torch.Size, list[int]],
        rank: int = 8,
        dtype: torch.dtype = torch.float32,
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
        return nn.init.normal_(
            tensor=torch.empty(shape, dtype=self.dtype),
            mean=0.0,
            std=1.0,
        )

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

        return torch.cat([bias, scale], dim=-1)

    def get_weights(
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
