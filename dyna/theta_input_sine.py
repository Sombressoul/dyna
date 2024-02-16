import torch
import torch.nn as nn

from typing import Optional

from dyna.signal import SignalComponential
from dyna.modulated_activation_sine import ModulatedActivationSine


class ThetaInputSine(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta_modes_out: Optional[int] = 7,
        theta_full_features: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super(ThetaInputSine, self).__init__(in_features, out_features, **kwargs)

        self.activation = ModulatedActivationSine(
            passive=False,
            count_modes=theta_modes_out,
            features=out_features if theta_full_features else 1,
        )

        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> SignalComponential:
        return self.activation(super(ThetaInputSine, self).forward(x))
