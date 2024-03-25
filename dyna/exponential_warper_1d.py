import torch
import torch.nn as nn
import math

from typing import Optional


class ExponentialWarper1D(nn.Module):
    _DEBUG: bool = False

    def __init__(
        self,
        features_in: int,
        features_hidden: Optional[int] = None,
        features_out: Optional[int] = None,
        **kwargs,
    ) -> None:
        super(ExponentialWarper1D, self).__init__(**kwargs)

        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        features_hidden = features_hidden if features_hidden is not None else features_in * 2
        features_out = features_out if features_out is not None else features_in

        assert type(features_in) == int and features_in > 0, "Features in must be a positive integer."
        assert type(features_hidden) == int and features_hidden > 0, "Features hidden must be a positive integer."
        assert type(features_out) == int and features_out > 0, "Features out must be a positive integer."

        # ================================================================================= #
        # ____________________________> Arguments.
        # ================================================================================= #
        self.features_in = features_in
        self.features_hidden = features_hidden
        self.features_out = features_out

        # ================================================================================= #
        # ____________________________> Weights.
        # ================================================================================= #
        # Upscale matrix.
        mat_up_r = torch.empty(
            [
                self.features_in,
                self.features_hidden,
            ],
            dtype=torch.float32,
        )
        mat_up_r = nn.init.uniform_(
            tensor=mat_up_r,
            a=-math.sqrt(math.pi / mat_up_r.shape[-2]),
            b=+math.sqrt(math.pi / mat_up_r.shape[-2]),
        )
        mat_up_i = torch.empty(
            [
                self.features_in,
                self.features_hidden,
            ],
            dtype=torch.float32,
        )
        mat_up_i = nn.init.normal_(
            tensor=mat_up_i,
            mean=0.0,
            std=1.0 / mat_up_i.numel(),
        )
        self.mat_up = nn.Parameter(
            data=torch.complex(
                real=mat_up_r,
                imag=mat_up_i,
            ),
        )

        # Downscale matrix.
        mat_down_r = torch.empty(
            [
                self.features_hidden,
                self.features_out,
            ],
            dtype=torch.float32,
        )
        mat_down_r = nn.init.uniform_(
            tensor=mat_down_r,
            a=-math.sqrt(math.pi / mat_down_r.shape[-2]),
            b=+math.sqrt(math.pi / mat_down_r.shape[-2]),
        )
        mat_down_i = torch.empty(
            [
                self.features_hidden,
                self.features_out,
            ],
            dtype=torch.float32,
        )
        mat_down_i = nn.init.normal_(
            tensor=mat_down_i,
            mean=0.0,
            std=1.0 / mat_down_i.numel(),
        )
        self.mat_down = nn.Parameter(
            data=torch.complex(
                real=mat_down_r,
                imag=mat_down_i,
            ),
        )

        # Exponentiation matrix.
        mat_exp_r = torch.empty(
            [
                self.features_hidden,
            ],
            dtype=torch.float32,
        )
        mat_exp_r = nn.init.uniform_(
            tensor=mat_exp_r,
            a=1.0 - math.sqrt(math.pi / mat_exp_r.numel()),
            b=1.0 + math.sqrt(math.pi / mat_exp_r.numel()),
        )
        mat_exp_i = torch.empty(
            [
                self.features_hidden,
            ],
            dtype=torch.float32,
        )
        mat_exp_i = nn.init.normal_(
            tensor=mat_exp_i,
            mean=0.0,
            std=1.0 / mat_exp_i.numel(),
        )
        self.mat_exp = nn.Parameter(
            data=torch.complex(
                real=mat_exp_r,
                imag=mat_exp_i,
            ),
        )

        pass

    def _log_x(
        self,
        x: torch.Tensor,
        label: Optional[str] = None,
    ) -> None:
        if not self._DEBUG:
            return None

        if label is not None:
            print(f"\n# =====> {label}:")

        print(f"{x.shape=}")
        print(f"{x.min()=}")
        print(f"{x.max()=}")
        print(f"{x.mean()=}")
        print(f"{x.std()=}")

        pass

    def forward(
        self,
        x: torch.Tensor,
        complex_output: bool = True,
    ) -> torch.Tensor:
        self._log_x(x, "Input")
        if x.dtype not in [torch.complex32, torch.complex64, torch.complex128]:
            x_r = x
            x_i = torch.zeros_like(x_r)
            x = torch.complex(
                real=x_r,
                imag=x_i,
            )
        self._log_x(x.real, "Input (complex), real")
        self._log_x(x.imag, "Input (complex), imag")

        x = torch.einsum(
            "b...i,ik->bk",
            x,
            self.mat_up,
        )
        self._log_x(x.real, "Input (complex, upscaled), real")
        self._log_x(x.imag, "Input (complex, upscaled), imag")

        x = torch.reshape(
            self.mat_exp,
            [
                *[1 for _ in range(x.ndim - 1)],
                self.mat_exp.shape[-1],
            ],
        ).expand_as(x) ** x
        self._log_x(x.real, "Input (complex, upscaled, exponentiated), real")
        self._log_x(x.imag, "Input (complex, upscaled, exponentiated), imag")

        # TODO: normalize in polar coordinates.

        x = torch.einsum(
            "bi,ik->bk",
            x,
            self.mat_down,
        )
        self._log_x(x.real, "Input (complex, downscaled), real")
        self._log_x(x.imag, "Input (complex, downscaled), imag")

        return x if complex_output else x.real
