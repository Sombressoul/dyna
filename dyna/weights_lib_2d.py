import torch
import torch.nn as nn
import math

from typing import Union, Optional


class WeightsLib2D(nn.Module):
    def __init__(
        self,
        shape: Union[torch.Size, list[int]],
        rank_mod: Optional[int] = None,
        rank_deltas: int = 1,
        use_deltas: bool = True,
        complex: bool = True,
        complex_output: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        shape = torch.Size(shape) if type(shape) == list else shape
        dtype_r = dtype

        if not complex:
            dtype_c = None
        elif dtype_r == torch.float32:
            dtype_c = torch.complex64
        elif dtype_r == torch.float64:
            dtype_c = torch.complex128
        else:
            raise ValueError(f"Unsupported dtype for complex mode: {dtype_r}.")

        assert len(shape) == 2, "Shape must be 2D."
        assert rank_deltas > 0, "Rank must be greater than 0."

        if complex_output:
            assert complex, "Complex output is only supported in complex mode."

        if complex:
            assert dtype in [
                torch.float32,
                torch.float64,
            ], "dtype must be float32 or float64 in complex mode."
        else:
            assert dtype in [
                torch.bfloat16,
                torch.float16,
                torch.float32,
                torch.float64,
            ], "dtype must be bfloat16, float16, float32 or float64 in real mode."

        # ================================================================================= #
        # ____________________________> Parameters.
        # ================================================================================= #
        self.shape = shape
        self.rank_mod = (
            int(max(rank_mod, 1))
            if rank_mod is not None
            else int(math.sqrt(math.prod([*shape])))
        )
        self.rank_deltas = rank_deltas
        self.use_deltas = use_deltas
        self.complex = complex
        self.complex_output = complex_output
        self.dtype_real = dtype_r
        self.dtype_complex = dtype_c

        # ================================================================================= #
        # ____________________________> Weights.
        # ================================================================================= #
        self.weights_base = nn.Parameter(
            data=self._create_weights_base(),
        )
        self.weights_mod_i = nn.Parameter(
            data=self._create_weights_mod([self.rank_mod, self.shape[0]]),
        )
        self.weights_mod_j = nn.Parameter(
            data=self._create_weights_mod([self.rank_mod, self.shape[1]]),
        )

        pass

    def _create_weights_base(
        self,
    ) -> torch.Tensor:
        std = 1.0 / math.log(math.prod(self.shape), math.e)

        base_r = torch.empty(self.shape, dtype=self.dtype_real)
        base_r = nn.init.normal_(
            tensor=base_r,
            mean=0.0,
            std=std,
        )

        if self.complex:
            base_i = torch.empty_like(base_r)
            base_i = nn.init.normal_(
                tensor=base_i,
                mean=0.0,
                std=std,
            )
            base = torch.complex(
                real=base_r,
                imag=base_i,
            ).to(self.dtype_complex)
        else:
            base = base_r.to(self.dtype_real)

        return base

    def _create_weights_mod(
        self,
        shape: Union[torch.Size, list[int]],
    ) -> torch.Tensor:
        bound_r = bound_i = 1.0 / math.log(math.prod(self.shape), math.e)

        mod_r = torch.empty(shape, dtype=self.dtype_real)
        mod_r = nn.init.uniform_(
            tensor=mod_r,
            a=-bound_r,
            b=+bound_r,
        )

        if self.complex:
            mod_i = torch.empty_like(mod_r)
            mod_i = nn.init.uniform_(
                tensor=mod_i,
                a=-bound_i,
                b=+bound_i,
            )
            mod = torch.complex(
                real=mod_r,
                imag=mod_i,
            ).to(dtype=self.dtype_complex, device=self.weights_base.device)
        else:
            mod = mod_r.to(dtype=self.dtype_real, device=self.weights_base.device)

        return mod

    def _create_weights_base_controls(
        self,
    ) -> torch.Tensor:
        bias = nn.init.normal_(
            tensor=torch.empty([1], dtype=self.dtype_real),
            mean=0.0,
            std=math.sqrt(1.0 / math.sqrt(math.prod(self.shape))),
        )
        scale = nn.init.normal_(
            tensor=torch.empty([1], dtype=self.dtype_real),
            mean=1.0,
            std=math.sqrt(1.0 / math.sqrt(math.prod(self.shape))),
        )

        base_controls_r = torch.cat([bias, scale], dim=0)

        if self.complex:
            base_controls_i = torch.empty_like(base_controls_r)
            base_controls_i = nn.init.uniform_(
                tensor=base_controls_i,
                a=-math.sqrt((math.pi * 2) / math.sqrt(math.prod(self.shape))),
                b=+math.sqrt((math.pi * 2) / math.sqrt(math.prod(self.shape))),
            )
            base_controls = torch.complex(
                real=base_controls_r,
                imag=base_controls_i,
            ).to(dtype=self.dtype_complex, device=self.weights_base.device)
        else:
            base_controls = base_controls_r.to(
                dtype=self.dtype_real, device=self.weights_base.device
            )

        return base_controls

    def _create_weights_mod_controls(
        self,
    ) -> torch.Tensor:
        bias = nn.init.normal_(
            tensor=torch.empty([2, self.rank_mod, 1], dtype=self.dtype_real),
            mean=0.0,
            std=math.sqrt(1.0 / self.rank_mod),
        )
        scale = nn.init.normal_(
            tensor=torch.empty([2, self.rank_mod, 1], dtype=self.dtype_real),
            mean=1.0,
            std=math.sqrt(1.0 / self.rank_mod),
        )

        mod_controls_r = torch.cat([bias, scale], dim=-1)

        if self.complex:
            mod_controls_i = torch.empty_like(mod_controls_r)
            mod_controls_i = nn.init.normal_(
                tensor=mod_controls_i,
                mean=0.0,
                std=math.sqrt(1.0 / self.rank_mod),
            )
            mod_controls = torch.complex(
                real=mod_controls_r,
                imag=mod_controls_i,
            ).to(dtype=self.dtype_complex, device=self.weights_base.device)
        else:
            mod_controls = mod_controls_r.to(
                dtype=self.dtype_real, device=self.weights_base.device
            )

        return mod_controls

    def _create_deltas(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bound = math.sqrt((math.pi * 2) / (self.rank_deltas**2))

        delta_a_r = torch.empty(
            [self.shape[-1], self.rank_deltas], dtype=self.dtype_real
        )
        delta_a_r = nn.init.uniform_(
            tensor=delta_a_r,
            a=-bound,
            b=+bound,
        )

        if self.complex:
            delta_a_i = torch.empty_like(delta_a_r)
            delta_a_i = nn.init.uniform_(
                tensor=delta_a_i,
                a=-bound,
                b=+bound,
            )
            delta_a = torch.complex(
                real=delta_a_r,
                imag=delta_a_i,
            ).to(dtype=self.dtype_complex, device=self.weights_base.device)
        else:
            delta_a = delta_a_r.to(
                dtype=self.dtype_real, device=self.weights_base.device
            )

        delta_b_r = torch.empty(
            [self.shape[-2], self.rank_deltas], dtype=self.dtype_real
        )
        delta_b_r = nn.init.uniform_(
            tensor=delta_b_r,
            a=-bound,
            b=+bound,
        )

        if self.complex:
            delta_b_i = torch.empty_like(delta_b_r)
            delta_b_i = nn.init.uniform_(
                tensor=delta_b_i,
                a=-bound,
                b=+bound,
            )
            delta_b = torch.complex(
                real=delta_b_r,
                imag=delta_b_i,
            ).to(dtype=self.dtype_complex, device=self.weights_base.device)
        else:
            delta_b = delta_b_r.to(
                dtype=self.dtype_real, device=self.weights_base.device
            )

        return delta_a, delta_b

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

    def _get_base_controls_list(
        self,
        names: Union[str, list[str]],
    ) -> torch.Tensor:
        names = names if isinstance(names, list) else [names]

        controls_list = [self._get_base_controls(name).unsqueeze(0) for name in names]

        return torch.cat(controls_list, dim=0)

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

    def _get_mod_controls_list(
        self,
        names: Union[str, list[str]],
    ) -> torch.Tensor:
        names = names if isinstance(names, list) else [names]

        controls_list = [self._get_mod_controls(name).unsqueeze(0) for name in names]

        return torch.cat(controls_list, dim=0)

    def _get_deltas(
        self,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights_name = f"weight_deltas_{name}"

        try:
            delta_a = self.get_parameter(f"{weights_name}_a")
            delta_b = self.get_parameter(f"{weights_name}_b")
        except AttributeError:
            delta_a, delta_b = self._create_deltas()

            self.register_parameter(
                name=f"{weights_name}_a",
                param=nn.Parameter(
                    data=delta_a,
                ),
            )
            self.register_parameter(
                name=f"{weights_name}_b",
                param=nn.Parameter(
                    data=delta_b,
                ),
            )

        return delta_a, delta_b

    def _get_deltas_list(
        self,
        names: Union[str, list[str]],
    ) -> torch.Tensor:
        names = names if isinstance(names, list) else [names]

        deltas_list = [self._get_deltas(name) for name in names]
        deltas_a_list = [delta[0].unsqueeze(0) for delta in deltas_list]
        deltas_b_list = [delta[1].unsqueeze(0) for delta in deltas_list]

        deltas_a = torch.cat(deltas_a_list, dim=0)
        deltas_b = torch.cat(deltas_b_list, dim=0)

        return deltas_a, deltas_b

    def _get_weights(
        self,
        base_controls: torch.Tensor,
        mod_controls: torch.Tensor,
        deltas: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Cast base and base controls to match weights base.
        weights_base = self.weights_base.unsqueeze(0).repeat(
            [base_controls.shape[0], 1, 1]
        )
        base_controls_bias = (
            base_controls[:, 0]
            .reshape(
                [
                    base_controls.shape[0],
                    *[1 for _ in range(len(self.weights_base.shape))],
                ]
            )
            .expand_as(weights_base)
        )
        base_controls_scale = (
            base_controls[:, 1]
            .reshape(
                [
                    base_controls.shape[0],
                    *[1 for _ in range(len(self.weights_base.shape))],
                ]
            )
            .expand_as(weights_base)
        )

        # Apply base controls.
        weights_base = weights_base + base_controls_bias
        weights_base = (weights_base**2) * base_controls_scale

        # i-dim: cast mod base and mod controls to match weights mod.
        weights_mod_i = self.weights_mod_i.unsqueeze(0).repeat(
            [mod_controls.shape[0], 1, 1]
        )
        mod_controls_i_bias = (
            mod_controls[:, 0, ..., 0].unsqueeze(-1).expand_as(weights_mod_i)
        )
        mod_controls_i_scale = (
            mod_controls[:, 0, ..., 1].unsqueeze(-1).expand_as(weights_mod_i)
        )

        # i-dim: apply mod controls.
        weights_mod_i = weights_mod_i + mod_controls_i_bias
        weights_mod_i = (weights_mod_i**2) * mod_controls_i_scale

        # i-dim: cast mod base and mod controls to match weights mod.
        weights_mod_j = self.weights_mod_j.unsqueeze(0).repeat(
            [mod_controls.shape[0], 1, 1]
        )
        mod_controls_j_bias = (
            mod_controls[:, 1, ..., 0].unsqueeze(-1).expand_as(weights_mod_j)
        )
        mod_controls_j_scale = (
            mod_controls[:, 1, ..., 1].unsqueeze(-1).expand_as(weights_mod_j)
        )

        # j-dim: apply mod controls.
        weights_mod_j = weights_mod_j + mod_controls_j_bias
        weights_mod_j = (weights_mod_j**2) * mod_controls_j_scale

        # Apply mod controls.
        weights_mod = weights_mod_i.permute([0, -1, -2]) @ weights_mod_j
        base_mod = weights_base * weights_mod

        # Apply deltas.
        if self.use_deltas:
            delta_a = base_mod @ deltas[0]
            delta_a = delta_a**2
            delta_b = (base_mod.permute([0, -1, -2]) @ deltas[1]).permute([0, -1, -2])
            delta_b = delta_b**2
            deltas = delta_a @ delta_b

            base_deltas = weights_base * deltas

            weights = base_mod + base_deltas
        else:
            weights = base_mod

        return weights

    def get_weights(
        self,
        names: Union[str, list[str]],
    ) -> torch.Tensor:
        names = names if isinstance(names, list) else [names]

        weights = self._get_weights(
            base_controls=self._get_base_controls_list(
                names=names,
            ),
            mod_controls=self._get_mod_controls_list(
                names=names,
            ),
            deltas=self._get_deltas_list(
                names=names,
            ),
        )

        return weights.real if not self.complex_output else weights
