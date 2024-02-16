import torch
import torch.nn as nn
import math

from typing import Optional, Union

from dyna.signal import SignalModular, SignalComponential


class ModulatedActivation(nn.Module):
    def __init__(
        self,
        passive: Optional[bool] = True,
        count_modes: Optional[int] = 7,
        features: Optional[int] = 1,
        theta_dynamic_range: Optional[float] = 7.5,
    ):
        super(ModulatedActivation, self).__init__()

        self.passive = passive
        self.count_modes = count_modes
        self.features = features
        self.theta_dynamic_range = theta_dynamic_range

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
            b=self.theta_dynamic_range / 2.0,
        )

        # Init deltas.
        deltas = torch.linspace(
            start=-self.theta_dynamic_range,
            end=+self.theta_dynamic_range,
            steps=self.count_modes,
        )
        deltas = deltas.reshape([-1, 1, 1]).repeat([1, 1, self.features])
        deltas_bias = torch.empty_like(deltas)
        deltas_bias = torch.nn.init.normal_(
            deltas_bias,
            mean=0.0,
            std=(self.theta_dynamic_range / (self.count_modes * 2.0)),
        )
        deltas = deltas + deltas_bias

        self.modes = nn.Parameter(torch.cat([alphas, betas, gammas, deltas], dim=1))

        pass

    def _dyna(
        self,
        signal: SignalModular,
    ) -> torch.Tensor:
        x_expanded = signal.x
        x_expanded = x_expanded.reshape(
            [*x_expanded.shape[0:-1], 1, x_expanded.shape[-1]]
        )
        modes_extra_dims = len(signal.modes.shape[1:-3])
        modes_expanded = signal.modes.permute(
            [0, -2, *range(1, 1 + modes_extra_dims), -3, -1]
        )
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
        x: Union[torch.Tensor, SignalModular],
        modes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.passive:
            assert modes is None, "modes must be None in active mode"
            assert not isinstance(
                x, SignalModular
            ), "x must be a tensor in active mode"

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
            signal = SignalModular(
                x=x,
                modes=modes,
            )
        else:
            assert isinstance(
                x, SignalModular
            ), "x must be a SignalModular instance in passive mode"
            signal = x

        components = self._dyna(signal)
        nonlinearity = components.sum(dim=-2) + 1.0
        x_transformed = signal.x * nonlinearity

        return SignalComponential(
            x=x_transformed,
            components=components,
            nonlinearity=nonlinearity,
        )
