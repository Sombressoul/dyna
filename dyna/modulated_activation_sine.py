import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Union

from dyna.signal import SignalModular, SignalComponential


class ModulatedActivationSine(nn.Module):
    def __init__(
        self,
        passive: Optional[bool] = True,
        count_modes: Optional[int] = 7,
        features: Optional[int] = 1,
        std: Optional[float] = 0.01,
    ):
        super(ModulatedActivationSine, self).__init__()

        self.passive = passive
        self.count_modes = count_modes
        self.features = features
        self.std = std

        # Init modes.
        modes = []

        for mode_idx in range(self.count_modes):
            freq = math.pi * (1.0 / (self.count_modes - mode_idx))
            noise = torch.empty([1, 4, self.features])
            noise = torch.nn.init.normal_(noise, mean=1.0, std=self.std)

            a = torch.ones([1, 1, self.features]) + noise[:, 0, :]
            b = torch.ones([1, 1, self.features]) * freq * noise[:, 1, :]
            g = torch.zeros([1, 1, self.features]) + (noise[:, 2, :] - 1.0)
            d = torch.zeros([1, 1, self.features]) + (noise[:, 3, :] - 1.0)

            mode = torch.cat([a, b, g, d], dim=-2)
            modes.append(mode)

        modes = torch.cat(modes, dim=0)
        self.modes = nn.Parameter(modes)

        pass

    def _wave_fn(
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

        transformed = alphas * torch.sin(x_expanded * betas + gammas) + deltas

        return transformed

    def forward(
        self,
        x: Union[torch.Tensor, SignalModular],
        modes: Optional[torch.Tensor] = None,
    ) -> SignalComponential:
        if not self.passive:
            assert modes is None, "modes must be None in active mode"
            assert not isinstance(x, SignalModular), "x must be a tensor in active mode"

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

        components = self._wave_fn(signal)
        nonlinearity = 1.0 + (components.sum(dim=-2) / self.count_modes)
        x_transformed = signal.x * nonlinearity

        return SignalComponential(
            x=x_transformed,
            components=components,
            nonlinearity=nonlinearity,
        )
