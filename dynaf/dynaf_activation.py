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
        mode: torch.Tensor,
    ) -> torch.Tensor:
        return mode[0] * (
            F.sigmoid(mode[1].abs() * (x - mode[2] - mode[3].abs()))
            - F.sigmoid(mode[1].abs() * (x - mode[2] + mode[3].abs()))
        )

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

        nonlinearity = torch.zeros_like(x)
        components = []
        for mode in modes:
            component = self._dynaf(x, mode)
            nonlinearity += component

            if return_components:
                components.append(component)

        nonlinearity = nonlinearity + 1.0

        if return_nonlinearity and return_components:
            return x, nonlinearity, components
        elif return_nonlinearity:
            return x, nonlinearity
        elif return_components:
            return x, components
        else:
            return x
