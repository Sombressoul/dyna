import torch
import torch.nn as nn

from typing import Tuple


class DyNAFThetaLinear(nn.Linear):
    def __init__(
        self,
        *args,
        theta_in: int,
        theta_out: int,
        **kwargs,
    ) -> None:
        super(DyNAFThetaLinear, self).__init__(*args, **kwargs)

        self.theta_in = theta_in
        self.theta_out = theta_out

        # Init sumodules.
        ...

        pass

    def forward(
        self,
        x: torch.Tensor,
        components: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply linear transformation to x to transform it into x_transformed.
        ...

        # Concat x_transformed & components (previous neuromodulatory profile) into activation_state.
        ...

        # Apply linear transformation to activation_state to transform it into param_quads.
        ...

        # return x_transformed, param_quads.
        ...

        pass
