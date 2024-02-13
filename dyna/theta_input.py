import torch
import torch.nn as nn

from typing import Optional

from dyna.signal import SignalComponential
from dyna.modulated_activation import ModulatedActivation


class ThetaInput(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta_modes: Optional[int] = 5,
        theta_full_features: Optional[bool] = True,
        theta_expected_input_min: Optional[float] = -5.0,
        theta_expected_input_max: Optional[float] = +5.0,
        **kwargs,
    ) -> None:
        super(ThetaInput, self).__init__(in_features, out_features, **kwargs)

        self.activation = ModulatedActivation(
            passive=True,
            count_modes=theta_modes,
            features=out_features if theta_full_features else 1,
            expected_input_min=theta_expected_input_min,
            expected_input_max=theta_expected_input_max,
        )

        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> SignalComponential:
        return self.activation(super(ThetaInput, self).forward(x))
