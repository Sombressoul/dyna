import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Union

from dyna.signal import SignalModular, SignalComponential


class ModulatedActivationAD(nn.Module):
    def __init__(
        self,
        passive: Optional[bool] = True,
        features: Optional[int] = 1,
        count_modes: Optional[int] = 5,
        eps: Optional[float] = 1.0e-5,
    ) -> None:
        super(ModulatedActivationAD, self).__init__()

        self.passive = passive
        self.features = features
        self.count_modes = count_modes
        self.eps = eps

        # ================================================================================= #
        # ____________________________> Modulation matrices.
        # ================================================================================= #
        # Only for active mode.
        if not self.passive:
            modes = torch.empty([self.count_modes, 2, self.features])
            std_base = 1 / math.sqrt(self.count_modes) + self.eps
            # Alphas.
            modes[:, 0, :] = nn.init.normal_(
                tensor=modes[:, 0, :],
                mean=0.0,
                std=std_base,
            )
            # Deltas.
            modes[:, 1, :] = nn.init.normal_(
                tensor=modes[:, 1, :],
                mean=0.0,
                std=1 - std_base,
            )
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
        deltas = modes_expanded[:, 1, :]

        count_modes = modes_expanded.shape[-2] if self.passive else self.count_modes
        steepness = torch.log(torch.tensor(torch.e + count_modes).to(x_expanded.device))
        spread = torch.e / 2

        transformed = (
            torch.e
            * F.tanh(alphas)
            * (
                -torch.sigmoid(steepness * (x_expanded - deltas) - spread)
                + torch.sigmoid(steepness * (x_expanded - deltas) + spread)
            )
        )

        return transformed

    def forward(
        self,
        x: Union[torch.Tensor, SignalModular],
        modes: Optional[torch.Tensor] = None,
    ) -> SignalComponential:
        modes = None

        if not self.passive:
            assert modes is None, "modes must be None in active mode"
            assert not isinstance(x, SignalModular), "x must be a tensor in active mode"

            modes = self.modes
            modes = modes.repeat([x.shape[0], *[1 for _ in range(len(modes.shape))]])
            modes = modes.reshape(
                [
                    modes.shape[0],
                    *[1 for _ in range(len(x.shape[1:-1]))],
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
        nonlinearity = components.sum(dim=-2) + 1.0
        x_transformed = signal.x * nonlinearity

        return SignalComponential(
            x=x_transformed,
            components=components,
            nonlinearity=nonlinearity,
        )
