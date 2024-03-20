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
        return_as_complex: bool = False,
    ) -> None:
        super().__init__()

        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        shape = torch.Size(shape) if type(shape) == list else shape
        dtype_r = dtype
        dtype_c = torch.complex64 if dtype_r == torch.float32 else torch.complex128

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
        self.dtype_r = dtype_r
        self.dtype_c = dtype_c
        self.return_as_complex = return_as_complex

        # ================================================================================= #
        # ____________________________> Weights.
        # ================================================================================= #
        self.weights_base = nn.Parameter(
            data=self._create_weights_base(),
        )
        self.weights_mod_i = nn.Parameter(
            data=self._create_weights_mod([self.rank, self.shape[0]]),
        )
        self.weights_mod_j = nn.Parameter(
            data=self._create_weights_mod([self.rank, self.shape[1]]),
        )

        pass

    def _create_weights_base(
        self,
    ) -> torch.Tensor:
        std = 1.0 / math.log(math.prod(self.shape), math.e)

        base_r = torch.empty(self.shape, dtype=self.dtype_r)
        base_r = nn.init.normal_(
            tensor=base_r,
            mean=0.0,
            std=std,
        )
        base_i = torch.empty_like(base_r)
        base_i = nn.init.normal_(
            tensor=base_i,
            mean=0.0,
            std=std,
        )
        base = torch.complex(
            real=base_r,
            imag=base_i,
        ).to(self.dtype_c)

        return base

    def _create_weights_mod(
        self,
        shape: Union[torch.Size, list[int]],
    ) -> torch.Tensor:
        bound_r = bound_i = 1.0 / math.log(math.prod(self.shape), math.e)

        mod_r = torch.empty(shape, dtype=self.dtype_r)
        mod_r = nn.init.uniform_(
            tensor=mod_r,
            a=-bound_r,
            b=+bound_r,
        )
        mod_i = torch.empty_like(mod_r)
        mod_i = nn.init.uniform_(
            tensor=mod_i,
            a=-bound_i,
            b=+bound_i,
        )
        mod = torch.complex(
            real=mod_r,
            imag=mod_i,
        ).to(self.dtype_c)

        return mod

    def _create_weights_base_controls(
        self,
    ) -> torch.Tensor:
        bias = nn.init.normal_(
            tensor=torch.empty([1], dtype=self.dtype_r),
            mean=0.0,
            std=math.sqrt(1.0 / math.sqrt(math.prod(self.shape))),
        )
        scale = nn.init.normal_(
            tensor=torch.empty([1], dtype=self.dtype_r),
            mean=1.0,
            std=math.sqrt(1.0 / math.sqrt(math.prod(self.shape))),
        )

        base_controls_r = torch.cat([bias, scale], dim=0)
        base_controls_i = torch.empty_like(base_controls_r)
        base_controls_i = nn.init.uniform_(
            tensor=base_controls_i,
            a=-math.sqrt((math.pi * 2) / math.sqrt(math.prod(self.shape))),
            b=+math.sqrt((math.pi * 2) / math.sqrt(math.prod(self.shape))),
        )
        base_controls = torch.complex(
            real=base_controls_r,
            imag=base_controls_i,
        ).to(self.dtype_c)

        return base_controls

    def _create_weights_mod_controls(
        self,
    ) -> torch.Tensor:
        bias = nn.init.normal_(
            tensor=torch.empty([2, self.rank, 1], dtype=self.dtype_r),
            mean=0.0,
            std=math.sqrt(1.0 / self.rank),
        )
        scale = nn.init.normal_(
            tensor=torch.empty([2, self.rank, 1], dtype=self.dtype_r),
            mean=1.0,
            std=math.sqrt(1.0 / self.rank),
        )

        mod_controls_r = torch.cat([bias, scale], dim=-1)
        mod_controls_i = torch.empty_like(mod_controls_r)
        mod_controls_i = nn.init.normal_(
            tensor=mod_controls_i,
            mean=0.0,
            std=math.sqrt(1.0 / self.rank),
        )
        mod_controls = torch.complex(
            real=mod_controls_r,
            imag=mod_controls_i,
        ).to(self.dtype_c)

        return mod_controls

    def _get_weights(
        self,
        base_controls: torch.Tensor,
        mod_controls: torch.Tensor,
    ) -> torch.Tensor:
        weights_base = self.weights_base + base_controls[0]
        weights_base = (weights_base**2) * base_controls[1]

        weights_mod_i = self.weights_mod_i + mod_controls[0, ..., 0].unsqueeze(-1)
        weights_mod_i = (weights_mod_i**2) * mod_controls[0, ..., 1].unsqueeze(-1)
        weights_mod_j = self.weights_mod_j + mod_controls[1, ..., 0].unsqueeze(-1)
        weights_mod_j = (weights_mod_j**2) * mod_controls[1, ..., 1].unsqueeze(-1)
        weights_mod = weights_mod_i.permute([-1, -2]) @ weights_mod_j

        return weights_base * weights_mod

    def _get_base_controls(
        self,
        name: str,
    ) -> torch.Tensor:
        weights_name = f"weight_base_controls_{name}"

        try:
            base_controls = self.get_parameter(weights_name)
        except AttributeError:
            base_controls = self._create_weights_base_controls()

            self.register_parameter(
                name=weights_name,
                param=nn.Parameter(
                    data=base_controls,
                ),
            )

        return base_controls

    def _get_mod_controls(
        self,
        name: str,
    ) -> torch.Tensor:
        weights_name = f"weight_mod_controls_{name}"

        try:
            mod_controls = self.get_parameter(weights_name)
        except AttributeError:
            mod_controls = self._create_weights_mod_controls()

            self.register_parameter(
                name=weights_name,
                param=nn.Parameter(
                    data=mod_controls,
                ),
            )

        return mod_controls

    def get_weights(
        self,
        name: str,
    ) -> torch.Tensor:
        weights = self._get_weights(
            base_controls=self._get_base_controls(
                name=name,
            ),
            mod_controls=self._get_mod_controls(
                name=name,
            ),
        )

        return weights if self.return_as_complex else weights.real
