import torch
import torch.nn as nn
import math

from enum import Enum
from dataclasses import dataclass

from typing import Union, Optional, Callable, List, Dict


class ActivationType(Enum):
    CUSTOM = "custom"
    IDENTITY = "identity"
    CARDIOID = "cardioid"
    MOBIUS = "mobius"


class TransformationType(Enum):
    TRANSLATION = "translation"
    INVERSION = "inversion"


@dataclass
class ActivationParams:
    operand_name: str
    params: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]


# Notes:
#   Combo #1: no deltas, no exponentiation, no activation, inversive transformation, complex
#       - fast, great performance
#   Combo #2: deltas with trainable exponents, cardiod activation, translative transformation, complex
#       - more diverse on long-range training
class WeightsLib2D(nn.Module):
    def __init__(
        self,
        shape: Union[torch.Size, List[int]],
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
        use_bias: bool = True,
        use_scale: bool = True,
        asymmetry: float = 1e-3,
        activation_type: Union[str, ActivationType] = ActivationType.IDENTITY,
        activation_fn: Optional[Callable] = None,
        transformation_type: Union[
            str, TransformationType
        ] = TransformationType.INVERSION,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        shape = torch.Size(shape) if type(shape) == list else shape
        dtype_r = dtype
        activation_type = (
            activation_type
            if type(activation_type) == ActivationType
            else ActivationType(activation_type)
        )
        transformation_type = (
            transformation_type
            if type(transformation_type) == TransformationType
            else TransformationType(transformation_type)
        )

        if activation_fn is not None and activation_type != ActivationType.CUSTOM:
            raise ValueError(
                "activation_fn is only supported in custom activation mode."
            )
        if activation_type == ActivationType.CUSTOM and activation_fn is None:
            raise ValueError("activation_fn is required in custom activation mode.")

        if use_exponentiation and trainable_exponents_deltas:
            assert (
                use_deltas
            ), "Trainable deltas exponents are only supported in use_deltas mode."

        if transformation_type == TransformationType.INVERSION:
            if activation_type == ActivationType.CARDIOID:
                raise ValueError(
                    " ".join(
                        [
                            "Cardioid activation is incompatible with inversive",
                            "transformation mode due to the gradient explosions.",
                        ]
                    )
                )

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
        self.activation_type = activation_type
        self.activation_fn = (
            activation_fn
            if activation_type == ActivationType.CUSTOM
            else self._get_activation()
        )
        self.transformation_type = transformation_type
        self.use_bias = use_bias
        self.use_scale = use_scale
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
    ) -> Callable:
        if self.activation_type == ActivationType.IDENTITY:
            return lambda x, _: x
        elif self.activation_type == ActivationType.CARDIOID:
            return self._activation_cardioid
        elif self.activation_type == ActivationType.MOBIUS:
            return self._activation_mobius

        raise ValueError(f"Unsupported activation type: {self.activation_type}.")

    def _activation_mobius(
        self,
        x: torch.Tensor,
        operand_name: str,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _activation_cardioid(
        self,
        x: torch.Tensor,
        params: ActivationParams,
    ) -> torch.Tensor:
        alpha_weight_name = f"activation_alpha_{params.operand_name}"

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
        shape: Union[torch.Size, List[int]],
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
        names: Union[str, List[str]],
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
        names: Union[str, List[str]],
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
        names: Union[str, List[str]],
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
        names: Union[str, List[str]],
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
        names: Union[str, List[str]],
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
        names: Union[str, List[str]],
    ) -> torch.Tensor:
        names = names if isinstance(names, list) else [names]

        exponents_list = [
            self._get_exponents_deltas(name).unsqueeze(0) for name in names
        ]
        exponents = torch.cat(exponents_list, dim=0)

        return exponents

    def _create_bias(
        self,
    ) -> torch.Tensor:
        bias = (
            torch.complex(
                real=nn.init.normal_(
                    tensor=torch.empty([1], dtype=self.dtype_real),
                    mean=0.0,
                    std=self.asymmetry,
                ),
                imag=nn.init.normal_(
                    tensor=torch.empty([1], dtype=self.dtype_real),
                    mean=0.0,
                    std=self.asymmetry,
                ),
            ).to(
                dtype=self.dtype_complex,
                device=self.weights_base.device,
            )
            if self.complex
            else nn.init.normal_(
                tensor=torch.empty([1], dtype=self.dtype_real),
                mean=0.0,
                std=self.asymmetry,
            ).to(
                dtype=self.dtype_real,
                device=self.weights_base.device,
            )
        )

        return bias

    def _get_bias(
        self,
        name: str,
    ) -> torch.Tensor:
        bias_name = f"main_bias_{name}"

        try:
            bias = self.get_parameter(bias_name)
        except AttributeError:
            bias = self._create_bias()

            self.register_parameter(
                name=bias_name,
                param=nn.Parameter(
                    data=bias,
                ),
            )

        return bias

    def _get_bias_list(
        self,
        names: Union[str, List[str]],
    ) -> torch.Tensor:
        names = names if isinstance(names, list) else [names]

        bias_list = [self._get_bias(name).unsqueeze(0) for name in names]
        bias = torch.cat(bias_list, dim=0)

        return bias

    def _create_scale(
        self,
    ) -> torch.Tensor:
        scale = (
            torch.complex(
                real=nn.init.normal_(
                    tensor=torch.empty([1], dtype=self.dtype_real),
                    mean=0.0,
                    std=self.asymmetry,
                ),
                imag=nn.init.normal_(
                    tensor=torch.empty([1], dtype=self.dtype_real),
                    mean=0.0,
                    std=self.asymmetry,
                ),
            ).to(
                dtype=self.dtype_complex,
                device=self.weights_base.device,
            )
            if self.complex
            else nn.init.normal_(
                tensor=torch.empty([1], dtype=self.dtype_real),
                mean=1.0,
                std=self.asymmetry,
            ).to(
                dtype=self.dtype_real,
                device=self.weights_base.device,
            )
        )

        return scale

    def _get_scale(
        self,
        name: str,
    ) -> torch.Tensor:
        scale_name = f"main_scale_{name}"

        try:
            scale = self.get_parameter(scale_name)
        except AttributeError:
            scale = self._create_scale()

            self.register_parameter(
                name=scale_name,
                param=nn.Parameter(
                    data=scale,
                ),
            )

        return scale

    def _get_scale_list(
        self,
        names: Union[str, List[str]],
    ) -> torch.Tensor:
        names = names if isinstance(names, list) else [names]

        scale_list = [self._get_scale(name).unsqueeze(0) for name in names]
        scale = torch.cat(scale_list, dim=0)

        return scale

    def _create_inversions(
        self,
    ) -> torch.Tensor:
        inversions = (
            torch.complex(
                real=nn.init.constant_(
                    tensor=torch.empty([1, 1], dtype=self.dtype_real),
                    val=-1.0,
                ),
                imag=nn.init.constant_(
                    tensor=torch.empty([1, 1], dtype=self.dtype_real),
                    val=+1.0,
                ),
            ).to(
                dtype=self.dtype_complex,
                device=self.weights_base.device,
            )
            if self.complex
            else nn.init.constant_(
                tensor=torch.empty([1, 1], dtype=self.dtype_real),
                val=1.0,
            ).to(
                dtype=self.dtype_real,
                device=self.weights_base.device,
            )
        )

        return inversions

    def _get_inversions(
        self,
        name: str,
    ) -> torch.Tensor:
        inversions_name = f"inversions_{name}"

        try:
            inversions = self.get_parameter(inversions_name)
        except AttributeError:
            inversions = self._create_inversions()

            self.register_parameter(
                name=inversions_name,
                param=nn.Parameter(
                    data=inversions,
                ),
            )

        return inversions

    def _get_inversions_list(
        self,
        names: Union[str, List[str]],
    ) -> torch.Tensor:
        names = names if isinstance(names, list) else [names]

        inversions_list = [self._get_inversions(name).unsqueeze(0) for name in names]
        inversions = torch.cat(inversions_list, dim=0)

        return inversions

    def _normalize_real(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        num = x - x.mean(dim=[-1, -2], keepdim=True)
        denom = x.std(dim=[-1, -2], keepdim=True)
        return num / denom

    def _normalize_polar(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if torch.is_complex(x):
            real = (x.abs() / x.abs().max()) * x.angle().cos()
            imag = (x.abs() / x.abs().max()) * x.angle().sin()
            x = torch.complex(
                real=real,
                imag=imag,
            ).to(dtype=self.dtype_complex, device=x.device)
        else:
            x = self._normalize_real(x)

        return x

    def _normalize_partial(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if torch.is_complex(x):
            real = self._normalize_real(x.real)
            imag = x.imag
            x = torch.complex(
                real=real,
                imag=imag,
            ).to(dtype=self.dtype_complex, device=x.device)
        else:
            x = self._normalize_real(x)

        return x

    def _get_weights(
        self,
        base_controls: torch.Tensor,
        mod_controls: torch.Tensor,
        deltas: tuple[torch.Tensor, torch.Tensor],
        exponents_base: Optional[torch.Tensor] = None,  # [n, 1]
        exponents_mod: Optional[torch.Tensor] = None,  # [n, 2, mod_rank, 1]
        exponents_deltas: Optional[torch.Tensor] = None,  # [n, 2, 1]
        bias: Optional[torch.Tensor] = None,  # [n, 1]
        scale: Optional[torch.Tensor] = None,  # [n, 1]
        inversions: Optional[torch.Tensor] = None,  # [n, 1, 1]
    ) -> torch.Tensor:
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
        weights_mod_i = self.activation_fn(
            weights_mod_i,
            ActivationParams(
                operand_name="weights_mod_i",
                params=None,  # NOTE: temporary placeholder.
            ),
        )
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
        weights_mod_j = self.activation_fn(
            weights_mod_j,
            ActivationParams(
                operand_name="weights_mod_j",
                params=None,  # NOTE: temporary placeholder.
            ),
        )
        weights_mod_j = weights_mod_j.mul(mod_controls_j_scale)

        # Apply mod controls.
        if self.transformation_type == TransformationType.TRANSLATION:
            weights_mod = weights_mod_i.permute([0, -1, -2]) @ weights_mod_j
        elif self.transformation_type == TransformationType.INVERSION:
            weights_mod = weights_mod_i.permute([0, -1, -2]) @ (
                inversions / weights_mod_j
            )
            weights_mod = self._normalize_partial(weights_mod)
        else:
            raise ValueError(
                f"Unknown transformation type: {self.transformation_type}."
            )
        
        if torch.isnan(weights_mod).any() or torch.isinf(weights_mod).any():
            raise ValueError("weights_mod has NaN or Inf elements.")

        weights_mod = self.activation_fn(
            weights_mod,
            ActivationParams(
                operand_name="weights_mod",
                params=None,  # NOTE: temporary placeholder.
            ),
        )
        weights_mod = self._normalize_partial(weights_mod) + 1.0
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
            delta_a = self.activation_fn(
                delta_a,
                ActivationParams(
                    operand_name="delta_a",
                    params=None,  # NOTE: temporary placeholder.
                ),
            )

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
            delta_b = self.activation_fn(
                delta_b,
                ActivationParams(
                    operand_name="delta_b",
                    params=None,  # NOTE: temporary placeholder.
                ),
            )

            deltas_combined = delta_a @ delta_b
            deltas_combined = self.activation_fn(
                deltas_combined,
                ActivationParams(
                    operand_name="deltas_combined",
                    params=None,  # NOTE: temporary placeholder.
                ),
            )
            deltas_weighted = base_mod * deltas_combined
            deltas_weighted = self.activation_fn(
                deltas_weighted,
                ActivationParams(
                    operand_name="deltas_weighted",
                    params=None,  # NOTE: temporary placeholder.
                ),
            )

            weights_dynamic = base_mod + deltas_weighted
        else:
            weights_dynamic = base_mod

        weights = weights_main_i @ weights_dynamic
        weights = self.activation_fn(
            weights,
            ActivationParams(
                operand_name="weights",
                params=None,  # NOTE: temporary placeholder.
            ),
        )
        weights = weights @ weights_main_j

        weights = (
            weights.add(
                bias.reshape(
                    [
                        bias.shape[0],
                        *[1 for _ in range(len(weights.shape) - 1)],
                    ]
                )
            )
            if self.use_bias
            else weights
        )
        weights = (
            weights.mul(
                scale.reshape(
                    [
                        scale.shape[0],
                        *[1 for _ in range(len(weights.shape) - 1)],
                    ]
                )
            )
            if self.use_scale
            else weights
        )

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
            bias=(
                self._get_bias_list(
                    names=names,
                )
                if self.use_bias
                else None
            ),
            scale=(
                self._get_scale_list(
                    names=names,
                )
                if self.use_scale
                else None
            ),
            inversions=(
                self._get_inversions_list(
                    names=names,
                )
                if self.transformation_type == TransformationType.INVERSION
                else None
            ),
        )

        return weights.real if not self.complex_output else weights
