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
        std: Optional[float] = 0.5,
    ):
        super(ModulatedActivationSine, self).__init__()

        self.passive = passive
        self.count_modes = count_modes
        self.features = features
        self.std = std

        # Init modes.
        modes = torch.empty([self.count_modes, 4, self.features])
        modes = nn.init.normal_(modes, mean=0.0, std=self.std)
        modes[:, 0, :] = modes[:, 0, :] + 1.0  # Alphas
        modes[:, 1, :] = modes[:, 1, :] + 1.0  # Betas

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

        count_modes = modes_expanded.shape[-2]
        freq_mul = 1.0 / self.count_modes
        freq_modes = torch.arange(0, count_modes, 1).to(x_expanded.device)
        freq = math.pi * (freq_mul * (count_modes - freq_modes))
        freq = freq.reshape(
            [
                *[1 for _ in range(len(modes_expanded.shape[:-2]) - 1)],
                count_modes,
                1,
            ]
        )

        alphas = modes_expanded[:, 0, :]
        betas = modes_expanded[:, 1, :] * freq
        gammas = modes_expanded[:, 2, :] * freq
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
