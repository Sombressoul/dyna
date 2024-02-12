import torch
import torch.nn as nn

from typing import Tuple


class DyNAFThetaLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta_in: int,
        theta_out: int,
        **kwargs,
    ) -> None:
        super(DyNAFThetaLinear, self).__init__(in_features, out_features, **kwargs)

        self.theta_in = theta_in
        self.theta_out = theta_out
        self.theta_out_quads = self.theta_out * 4

        # Define an additional linear layer for theta transformation.
        self.theta_transformation = nn.Linear(
            in_features=out_features + self.theta_in,
            out_features=self.theta_out_quads,
            bias=True,
        )

        pass

    def forward(
        self,
        x: torch.Tensor,
        components: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply the parent class's linear transformation to x.
        x_transformed = super(DyNAFThetaLinear, self).forward(x)

        # Flatten and expand components to match the shape of x_transformed.
        components_expanded = ...

        # Concat x_transformed & components_expanded to form the input (activation_state) to theta_transformation.
        activation_state = ...

        # Apply linear transformation to activation_state to transform it into param_quads.
        param_quads = self.theta_transformation(activation_state)

        # Reshape param_quads to match the input modes shape of DyNAFActivation.
        param_quads = ...

        return x_transformed, param_quads
