import torch
import torch.nn as nn
import math

from typing import Optional


class DyNAFActivation(nn.Module):
    def __init__(
        self,
        passive: Optional[bool] = True,
        count_modes: Optional[int] = 5,
        features: Optional[int] = 1,
        expected_input_min: Optional[float] = -1.0,
        expected_input_max: Optional[float] = 1.0,
        eps: Optional[float] = 1e-3,
    ):
        super(DyNAFActivation, self).__init__()

        self.passive = passive
        self.count_modes = count_modes
        self.features = features
        self.expected_input_min = expected_input_min
        self.expected_input_max = expected_input_max
        self.eps = eps

        # Init alphas.
        alphas = torch.empty([self.count_modes, 1, self.features])
        alphas = torch.nn.init.normal_(
            alphas,
            mean=0.0,
            std=math.sqrt(2.0),
        )

        # Init betas.
        betas = torch.empty([self.count_modes, 1, self.features])
        betas = torch.nn.init.uniform_(
            betas,
            a=1.0 / math.sqrt(self.count_modes),
            b=math.sqrt(self.count_modes),
        )

        # Init gammas.
        gammas = torch.empty([self.count_modes, 1, self.features])
        gammas = torch.nn.init.uniform_(
            gammas,
            a=self.eps,
            b=math.log(
                math.fabs(self.expected_input_max - self.expected_input_min),
                math.sqrt(2),
            ),
        )

        # Init deltas.
        deltas = torch.arange(
            start=self.expected_input_min,
            end=self.expected_input_max,
            step=(
                math.fabs(self.expected_input_max - self.expected_input_min)
                / self.count_modes
            ),
        ) + (
            math.fabs(self.expected_input_max - self.expected_input_min)
            / self.count_modes
        )
        deltas = deltas.reshape([-1, 1, 1]).repeat([1, 1, self.features])
        deltas_bias = torch.empty_like(deltas)
        deltas_bias = torch.nn.init.normal_(
            deltas_bias,
            mean=0.0,
            std=(
                math.fabs(self.expected_input_max - self.expected_input_min)
                / self.count_modes
            ),
        )
        deltas = deltas + deltas_bias

        self.modes = nn.Parameter(torch.cat([alphas, betas, gammas, deltas], dim=1))

        pass

    def _dynaf(
        self,
        x: torch.Tensor,
        modes: torch.Tensor,
    ) -> torch.Tensor:
        x_expanded = x.unsqueeze(0).expand([modes.shape[0], *x.shape])
        modes_expanded = modes.reshape(
            [
                *modes.shape[0:2],
                *[1 for _ in range(len(x_expanded.shape) - 2)],
                x.shape[-1],
            ]
        )

        alphas = modes_expanded[:, 0, :]
        betas = modes_expanded[:, 1, :]
        gammas = modes_expanded[:, 2, :]
        deltas = modes_expanded[:, 3, :]

        transformed = alphas * (
            (
                1.0
                / (1 + torch.e ** (torch.abs(betas) * (x - deltas - torch.abs(gammas))))
            )
            - (
                1.0
                / (1 + torch.e ** (torch.abs(betas) * (x - deltas + torch.abs(gammas))))
            )
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
            modes = self.modes
        else:
            assert modes is not None, "modes must be provided in active mode"

        components = self._dynaf(x, modes)
        nonlinearity = components.sum(dim=0) + 1.0
        x_transformed = x * nonlinearity

        if return_nonlinearity and return_components:
            return x_transformed, nonlinearity, components
        elif return_nonlinearity:
            return x_transformed, nonlinearity
        elif return_components:
            return x_transformed, components
        else:
            return x_transformed
