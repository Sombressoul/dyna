import torch
import torch.nn as nn
import math

from typing import Optional

from dyna.signal import SignalComponential


class ThetaOutput(nn.Module):
    def __init__(
        self,
        in_features: int,
        theta_components_in: Optional[int] = 7,
    ) -> None:
        super(ThetaOutput, self).__init__()

        self.in_features = in_features
        self.theta_components_in = theta_components_in

        # Define matrices for output transformations.
        output_transform = torch.empty(
            [self.in_features, self.theta_components_in, self.theta_components_in]
        )
        output_transform = torch.nn.init.uniform_(
            tensor=output_transform,
            a=-math.sqrt(3 / self.theta_components_in),
            b=+math.sqrt(3 / self.theta_components_in),
        )
        self.output_transform = nn.Parameter(output_transform)

        output_extract = torch.empty([self.in_features, self.theta_components_in, 1])
        output_extract = torch.nn.init.uniform_(
            tensor=output_extract,
            a=-math.sqrt(3),
            b=+math.sqrt(3),
        )
        self.output_extract = nn.Parameter(output_extract)

        pass

    def forward(
        self,
        x: SignalComponential,
    ) -> torch.Tensor:
        x_signal = x.x
        x_signal = x_signal.unsqueeze(-1)
        x_componential = x.components
        x_componential = x_componential.permute(
            [
                *[i for i in range(len(x_componential.shape[:-2]))],
                -1,
                -2,
            ]
        )
        x_componential = torch.einsum(
            "b...ij,ijk -> b...ik", x_componential, self.output_transform
        )
        x_composite = x_signal * x_componential
        x_composite = torch.einsum(
            "b...ij,ijk -> b...ik", x_composite, self.output_extract
        )
        x_composite = x_composite.squeeze(-1)

        return x_composite
