import torch
import torch.nn as nn
import math

from typing import Union, Optional, Callable

# Best with:
#   - no deltas, no exponentiation, complex
#   - deltas with trainable exponents, complex
class WeightsLib2D(nn.Module):
    def __init__(
        self,
        shape: Union[torch.Size, list[int]],
        rank_mod: Optional[int] = None,
        rank_deltas: int = 1,
        use_deltas: bool = False,
        complex: bool = True,
        complex_output: bool = True,
        use_exponentiation: bool = False,
        trainable_exponents_base: bool = True,
        trainable_exponents_mod: bool = True,
        trainable_exponents_deltas: bool = True,
        exponents_initial_value_real: float = 1.0,
        exponents_initial_value_imag: float = 0.0,
        asymmetry: float = 1e-3,
        activation: Optional[Union[str, Callable]] = "cardioid",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        shape = torch.Size(shape) if type(shape) == list else shape
        dtype_r = dtype

        if use_exponentiation and trainable_exponents_deltas:
            assert (
                use_deltas
            ), "Trainable deltas exponents are only supported in use_deltas mode."

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

        if use_exponentiation:
            assert complex, "Exponentiation is only supported in complex mode."

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
        self.use_exponentiation = use_exponentiation
        self.trainable_exponents_base = trainable_exponents_base
        self.trainable_exponents_mod = trainable_exponents_mod
        self.trainable_exponents_deltas = trainable_exponents_deltas
        self.exponents_initial_value_real = exponents_initial_value_real
        self.exponents_initial_value_imag = exponents_initial_value_imag
        self.asymmetry = asymmetry
        self.activation = self._get_activation(activation)
        self.dtype_real = dtype_r
        self.dtype_complex = dtype_c

        # ================================================================================= #
        # ____________________________> Weights.
        # ================================================================================= #
        self.weights_main_i = nn.Parameter(
            data=self._create_weights_base(
                shape=torch.Size([self.shape[0], self.shape[0]]),
            ),
        )
        self.weights_main_j = nn.Parameter(
            data=self._create_weights_base(
                shape=torch.Size([self.shape[1], self.shape[1]]),
            ),
        )
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

    def _get_activation(
        self,
        activation: Optional[Union[str, Callable]],
    ) -> Callable:
        if type(activation) == str:
            if activation == "cardioid":
                activation = self._activation_cardioid
            else:
                raise ValueError(f"Unsupported activation: {activation}.")
        elif activation is None:
            activation = lambda x: x
        else:
            assert callable(activation), "Activation must be callable."

        return activation

    def _activation_cardioid(
        self,
        x: torch.Tensor,
        operand_name: str,
    ) -> torch.Tensor:
        alpha_weight_name = f"activation_alpha_{operand_name}"

        try:
            alpha = getattr(self, alpha_weight_name)
        except AttributeError:
            alpha = torch.complex(
                real=torch.nn.init.normal_(
                    tensor=torch.empty([1], dtype=self.dtype_real),
                    mean=1.0,
                    std=self.asymmetry,
                ),
                imag=torch.nn.init.normal_(
                    tensor=torch.empty([1], dtype=self.dtype_real),
                    mean=0.0,
                    std=self.asymmetry,
                ),
            ).to(dtype=self.dtype_complex, device=self.weights_base.device)

            self.register_parameter(
                name=alpha_weight_name,
                param=nn.Parameter(
                    data=alpha,
                ),
            )

        cos_arg = torch.angle(x) + torch.angle(alpha)
        fx = 0.5 * (1.0 + torch.cos(cos_arg)) * x

        return fx

    def _create_weights_base(
        self,
        shape: torch.Size = None,
    ) -> torch.Tensor:
        shape = shape if shape is not None else self.shape

        std = 1.0 / math.log(math.prod(shape), math.e)

        base_r = torch.empty(shape, dtype=self.dtype_real)
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

    def _create_exponents_base(
        self,
    ) -> torch.Tensor:
        exponents_r = nn.init.normal_(
            tensor=torch.empty([1], dtype=self.dtype_real),
            mean=self.exponents_initial_value_real,
            std=self.asymmetry,
        )

        if self.complex:
            exponents_i = nn.init.normal_(
                tensor=torch.empty([1], dtype=self.dtype_real),
                mean=self.exponents_initial_value_imag,
                std=self.asymmetry,
            )
            exponents = torch.complex(
                real=exponents_r,
                imag=exponents_i,
            ).to(dtype=self.dtype_complex, device=self.weights_base.device)
        else:
            exponents = exponents_r.to(
                dtype=self.dtype_real, device=self.weights_base.device
            )

        return exponents

    def _get_exponents_base(
        self,
        name: str,
    ) -> torch.Tensor:
        exponents_name = f"exponents_base_{name}"

        if self.trainable_exponents_base:
            try:
                exponents_base = self.get_parameter(exponents_name)
            except AttributeError:
                exponents_base = self._create_exponents_base()

                self.register_parameter(
                    name=exponents_name,
                    param=nn.Parameter(
                        data=exponents_base,
                    ),
                )
        else:
            exponents_base = (
                torch.complex(
                    real=torch.tensor(self.exponents_initial_value_real),
                    imag=torch.tensor(self.exponents_initial_value_imag),
                ).to(
                    dtype=self.dtype_complex,
                    device=self.weights_base.device,
                )
                if self.complex
                else torch.tensor(self.exponents_initial_value_real).to(
                    dtype=self.dtype_real,
                    device=self.weights_base.device,
                )
            )
            exponents_base = exponents_base.requires_grad_(False)

        return exponents_base

    def _get_exponents_base_list(
        self,
        names: Union[str, list[str]],
    ) -> torch.Tensor:
        names = names if isinstance(names, list) else [names]

        exponents_list = [self._get_exponents_base(name).unsqueeze(0) for name in names]
        exponents = torch.cat(exponents_list, dim=0)

        return exponents

    def _create_exponents_mod(
        self,
    ) -> torch.Tensor:
        exponents_r = nn.init.normal_(
            tensor=torch.empty([2, self.rank_mod, 1], dtype=self.dtype_real),
            mean=self.exponents_initial_value_real,
            std=self.asymmetry,
        )

        if self.complex:
            exponents_i = nn.init.normal_(
                tensor=torch.empty([2, self.rank_mod, 1], dtype=self.dtype_real),
                mean=self.exponents_initial_value_imag,
                std=self.asymmetry,
            )
            exponents = torch.complex(
                real=exponents_r,
                imag=exponents_i,
            ).to(dtype=self.dtype_complex, device=self.weights_base.device)
        else:
            exponents = exponents_r.to(
                dtype=self.dtype_real, device=self.weights_base.device
            )

        return exponents

    def _get_exponents_mod(
        self,
        name: str,
    ) -> torch.Tensor:
        exponents_name = f"exponents_mod_{name}"

        if self.trainable_exponents_mod:
            try:
                exponents_mod = self.get_parameter(exponents_name)
            except AttributeError:
                exponents_mod = self._create_exponents_mod()

                self.register_parameter(
                    name=exponents_name,
                    param=nn.Parameter(
                        data=exponents_mod,
                    ),
                )
        else:
            exponents_mod = (
                torch.complex(
                    real=torch.tensor(self.exponents_initial_value_real),
                    imag=torch.tensor(self.exponents_initial_value_imag),
                ).to(
                    dtype=self.dtype_complex,
                    device=self.weights_base.device,
                )
                if self.complex
                else torch.tensor(self.exponents_initial_value_real).to(
                    dtype=self.dtype_real,
                    device=self.weights_base.device,
                )
            )
            exponents_mod = exponents_mod.reshape([1, 1, 1])
            exponents_mod = exponents_mod.repeat([2, self.rank_mod, 1])
            exponents_mod = exponents_mod.requires_grad_(False)

        return exponents_mod

    def _get_exponents_mod_list(
        self,
        names: Union[str, list[str]],
    ) -> torch.Tensor:
        names = names if isinstance(names, list) else [names]

        exponents_list = [self._get_exponents_mod(name).unsqueeze(0) for name in names]
        exponents = torch.cat(exponents_list, dim=0)

        return exponents

    def _create_exponents_deltas(
        self,
    ) -> torch.Tensor:
        exponents = (
            torch.complex(
                real=nn.init.normal_(
                    tensor=torch.empty([2, 1], dtype=self.dtype_real),
                    mean=self.exponents_initial_value_real,
                    std=self.asymmetry,
                ),
                imag=nn.init.normal_(
                    tensor=torch.empty([2, 1], dtype=self.dtype_real),
                    mean=self.exponents_initial_value_imag,
                    std=self.asymmetry,
                ),
            ).to(
                dtype=self.dtype_complex,
                device=self.weights_base.device,
            )
            if self.complex
            else nn.init.normal_(
                tensor=torch.empty([2, 1], dtype=self.dtype_real),
                mean=self.exponents_initial_value_real,
                std=self.asymmetry,
            ).to(
                dtype=self.dtype_real,
                device=self.weights_base.device,
            )
        )

        return exponents

    def _get_exponents_deltas(
        self,
        name: str,
    ) -> torch.Tensor:
        exponents_name = f"exponents_deltas_{name}"

        if self.trainable_exponents_deltas:
            try:
                exponents_deltas = self.get_parameter(exponents_name)
            except AttributeError:
                exponents_deltas = self._create_exponents_deltas()

                self.register_parameter(
                    name=exponents_name,
                    param=nn.Parameter(
                        data=exponents_deltas,
                    ),
                )
        else:
            exponents_deltas = (
                torch.complex(
                    real=nn.init.constant_(
                        tensor=torch.empty([2, 1], dtype=self.dtype_real),
                        val=self.exponents_initial_value_real,
                    ),
                    imag=nn.init.constant_(
                        tensor=torch.empty([2, 1], dtype=self.dtype_real),
                        val=self.exponents_initial_value_imag,
                    ),
                ).to(
                    dtype=self.dtype_complex,
                    device=self.weights_base.device,
                )
                if self.complex
                else nn.init.constant_(
                    tensor=torch.empty([2, 1], dtype=self.dtype_real),
                    val=self.exponents_initial_value_real,
                ).to(
                    dtype=self.dtype_real,
                    device=self.weights_base.device,
                )
            ).requires_grad_(False)

        return exponents_deltas

    def _get_exponents_deltas_list(
        self,
        names: Union[str, list[str]],
    ) -> torch.Tensor:
        names = names if isinstance(names, list) else [names]

        exponents_list = [
            self._get_exponents_deltas(name).unsqueeze(0) for name in names
        ]
        exponents = torch.cat(exponents_list, dim=0)

        return exponents

    def _get_weights(
        self,
        base_controls: torch.Tensor,
        mod_controls: torch.Tensor,
        deltas: tuple[torch.Tensor, torch.Tensor],
        exponents_base: Optional[torch.Tensor] = None,  # [n, 1]
        exponents_mod: Optional[torch.Tensor] = None,  # [n, 2, mod_rank, 1]
        exponents_deltas: Optional[torch.Tensor] = None,  # [n, 2, 1]
    ) -> torch.Tensor:
        # Various functions.
        normalize = lambda x: (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=0, keepdim=True)

        # Cast main weights.
        weights_main_i = self.weights_main_i.unsqueeze(0).repeat(
            [base_controls.shape[0], 1, 1]
        )
        weights_main_j = self.weights_main_j.unsqueeze(0).repeat(
            [base_controls.shape[0], 1, 1]
        )
        weights_base = self.weights_base.unsqueeze(0).repeat(
            [base_controls.shape[0], 1, 1]
        )

        # Cast base controls and base exponents to match target weights.
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
        weights_base = (
            weights_base.pow(
                exponents_base.reshape(
                    [
                        base_controls.shape[0],
                        *[1 for _ in range(len(weights_base.shape) - 1)],
                    ]
                )
            )
            if self.use_exponentiation
            else weights_base
        )
        weights_base = weights_base.mul(base_controls_scale)

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
        weights_mod_i = (
            weights_mod_i.pow(exponents_mod[:, 0, ...]).mul(mod_controls_i_scale)
            if self.use_exponentiation
            else weights_mod_i
        )
        weights_mod_i = self.activation(weights_mod_i, "weights_mod_i")
        weights_mod_i = weights_mod_i.mul(mod_controls_i_scale)

        # j-dim: cast mod base and mod controls to match weights mod.
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
        weights_mod_j = (
            weights_mod_j.pow(exponents_mod[:, 1, ...]).mul(mod_controls_j_scale)
            if self.use_exponentiation
            else weights_mod_j
        )
        weights_mod_j = self.activation(weights_mod_j, "weights_mod_j")
        weights_mod_j = weights_mod_j.mul(mod_controls_j_scale)

        # Apply mod controls.
        weights_mod = weights_mod_i.permute([0, -1, -2]) @ weights_mod_j
        weights_mod = self.activation(weights_mod, "weights_mod")
        weights_mod = normalize(weights_mod) + 1.0
        base_mod = weights_base + (weights_base * weights_mod)

        # Apply deltas.
        if self.use_deltas:
            delta_a = base_mod @ deltas[0]
            delta_a = (
                delta_a.pow(
                    exponents_deltas[:, 0, ...].reshape(
                        [
                            exponents_deltas.shape[0],
                            *[1 for _ in range(len(delta_a.shape) - 1)],
                        ]
                    )
                )
                if self.use_exponentiation
                else delta_a
            )
            delta_a = self.activation(delta_a, "delta_a")

            delta_b = (base_mod.permute([0, -1, -2]) @ deltas[1]).permute([0, -1, -2])
            delta_b = (
                delta_b.pow(
                    exponents_deltas[:, 1, ...].reshape(
                        [
                            exponents_deltas.shape[0],
                            *[1 for _ in range(len(delta_b.shape) - 1)],
                        ]
                    )
                )
                if self.use_exponentiation
                else delta_b
            )
            delta_b = self.activation(delta_b, "delta_b")

            deltas_combined = delta_a @ delta_b
            deltas_combined = self.activation(deltas_combined, "deltas_combined")
            deltas_weighted = base_mod * deltas_combined
            deltas_weighted = self.activation(deltas_weighted, "deltas_weighted")

            weights_dynamic = base_mod + deltas_weighted
        else:
            weights_dynamic = base_mod

        weights = weights_main_i @ weights_dynamic
        weights = self.activation(weights, "weights")
        weights = weights @ weights_main_j

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
            exponents_base=(
                self._get_exponents_base_list(
                    names=names,
                )
                if self.use_exponentiation
                else None
            ),
            exponents_mod=(
                self._get_exponents_mod_list(
                    names=names,
                )
                if self.use_exponentiation
                else None
            ),
            exponents_deltas=(
                self._get_exponents_deltas_list(
                    names=names,
                )
                if self.use_exponentiation
                else None
            ),
        )

        return weights.real if not self.complex_output else weights
