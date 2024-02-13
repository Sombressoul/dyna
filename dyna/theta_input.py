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
        theta_modes: Optional[int] = 7,
        theta_full_features: Optional[bool] = True,
        theta_dynamic_range: Optional[float] = 7.5,
        **kwargs,
    ) -> None:
        super(ThetaInput, self).__init__(in_features, out_features, **kwargs)

        self.activation = ModulatedActivation(
            passive=True,
            count_modes=theta_modes,
            features=out_features if theta_full_features else 1,
            theta_dynamic_range=theta_dynamic_range,
        )

        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> SignalComponential:
        return self.activation(super(ThetaInput, self).forward(x))
