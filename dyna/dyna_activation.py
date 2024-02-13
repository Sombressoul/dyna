import torch
import torch.nn as nn
import math

from typing import Optional


class DyNAActivation(nn.Module):
    def __init__(
        self,
        passive: Optional[bool] = True,
        count_modes: Optional[int] = 5,
        features: Optional[int] = 1,
        expected_input_min: Optional[float] = -5.0,
        expected_input_max: Optional[float] = +5.0,
    ):
        super(DyNAActivation, self).__init__()

        self.passive = passive
        self.count_modes = count_modes
        self.features = features
        self.expected_input_min = expected_input_min
        self.expected_input_max = expected_input_max

        # Init alphas.
        alphas = torch.empty([self.count_modes, 1, self.features])
        alphas = torch.nn.init.normal_(
            alphas,
            mean=0.0,
            std=1.0 / self.count_modes,
        )

        # Init betas.
        betas = torch.empty([self.count_modes, 1, self.features])
        betas = torch.nn.init.uniform_(
            betas,
            a=1.0 / math.sqrt(self.count_modes),
            b=math.log(
                self.count_modes,
                math.sqrt(2),
            ),
        )

        # Init gammas.
        gammas = torch.empty([self.count_modes, 1, self.features])
        gammas = torch.nn.init.uniform_(
            gammas,
            a=1.0 / self.count_modes,
            b=math.fabs(self.expected_input_max - self.expected_input_min) / 2.0,
        )

        # Init deltas.
        deltas = torch.linspace(
            start=self.expected_input_min,
            end=self.expected_input_max,
            steps=self.count_modes,
        )
        deltas = deltas.reshape([-1, 1, 1]).repeat([1, 1, self.features])
        deltas_bias = torch.empty_like(deltas)
        deltas_bias = torch.nn.init.normal_(
            deltas_bias,
            mean=0.0,
            std=(
                math.fabs(self.expected_input_max - self.expected_input_min)
                / (self.count_modes * 2.0)
            ),
        )
        deltas = deltas + deltas_bias

        self.modes = nn.Parameter(torch.cat([alphas, betas, gammas, deltas], dim=1))

        pass

    def _dyna(
        self,
        x: torch.Tensor,
        modes: torch.Tensor,
    ) -> torch.Tensor:
        x_expanded = x
        x_expanded = x_expanded.reshape(
            [*x_expanded.shape[0:-1], 1, x_expanded.shape[-1]]
        )
        modes_extra_dims = len(modes.shape[1:-3])
        modes_expanded = modes.permute([0, -2, *range(1, 1 + modes_extra_dims), -3, -1])
        modes_expanded = modes_expanded.reshape(
            [
                *modes_expanded.shape[0:-2],
                *modes_expanded.shape[-2:],
            ]
        )

        alphas = modes_expanded[:, 0, :]
        betas = modes_expanded[:, 1, :]
        gammas = modes_expanded[:, 2, :]
        deltas = modes_expanded[:, 3, :]

        transformed = alphas * (
            # (
            #     1.0
            #     / (1 + torch.e ** (torch.abs(betas) * (x - deltas - torch.abs(gammas))))
            # )
            # - (
            #     1.0
            #     / (1 + torch.e ** (torch.abs(betas) * (x - deltas + torch.abs(gammas))))
            # )
            # NOTE: The same sigmoid, but numerically stable. Thanks to PyTorch team!
            -torch.sigmoid(betas * (x_expanded - deltas - torch.abs(gammas)))
            + torch.sigmoid(betas * (x_expanded - deltas + torch.abs(gammas)))
        )

        return transformed

    def forward(
        self,
        x: torch.Tensor,
        modes: Optional[torch.Tensor] = None,
        return_components: Optional[bool] = False,
        return_nonlinearity: Optional[bool] = False,
    ) -> torch.Tensor:
        if self.passive:
            assert modes is None, "modes must be None in passive mode"

            extra_dims = len(x.shape[1:-1])
            modes = self.modes
            modes = modes.repeat([x.shape[0], *[1 for _ in range(len(modes.shape))]])
            modes = modes.reshape(
                [
                    modes.shape[0],
                    *[1 for _ in range(extra_dims)],
                    *modes.shape[1:],
                ]
            )
        else:
            assert modes is not None, "modes must be provided in active mode"

        components = self._dyna(x, modes)
        nonlinearity = components.sum(dim=-2) + 1.0
        x_transformed = x * nonlinearity

        if return_nonlinearity and return_components:
            return x_transformed, nonlinearity, components
        elif return_nonlinearity:
            return x_transformed, nonlinearity
        elif return_components:
            return x_transformed, components
        else:
            return x_transformed
