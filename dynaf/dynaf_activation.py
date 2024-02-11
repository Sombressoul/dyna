import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class DyNAFActivation(nn.Module):
    def __init__(
        self,
        passive: Optional[bool] = True,
        count_modes: Optional[int] = 5,
        features: Optional[int] = 1,
    ):
        super(DyNAFActivation, self).__init__()

        self.passive = passive
        self.count_modes = count_modes
        self.features = features

        modes = torch.empty([self.count_modes, 4, self.features])
        modes = torch.nn.init.normal_(
            modes,
            mean=0.0,
            std=1.0 / self.count_modes,
        )
        ranges = torch.arange(
            start=-1.0,
            end=1.0,
            step=2.0 / self.count_modes,
        ) + (1.0 / self.count_modes)
        ranges = ranges.reshape([-1, 1, 1]).repeat([1, 1, self.features])
        modes[:, 2] = ranges[:, 0]
        self.modes = nn.Parameter(modes)

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
