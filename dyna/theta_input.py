import torch
import torch.nn as nn

from typing import Optional

from dyna.signal import SignalComponential
from dyna.modulated_activation import ModulatedActivation


class ThetaInput(nn.Module):
    def __init__(
        self,
        features: int,
        theta_activation: ModulatedActivation,
        theta_modes_out: Optional[int] = 7,
        theta_full_features: Optional[bool] = True,
    ) -> None:
        super(ThetaInput, self).__init__()

        self.linear = nn.Linear(
            in_features=features,
            out_features=features,
            bias=False,
        )

        self.activation = theta_activation(
            passive=False,
            count_modes=theta_modes_out,
            features=features if theta_full_features else 1,
        )

        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> SignalComponential:
        return self.activation(self.linear(x))
