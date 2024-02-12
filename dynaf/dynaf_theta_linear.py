import torch
import torch.nn as nn

from typing import Tuple, Optional

# IMPORTANT!
# THIS IMPLEMENTATION IS INCORRECT!
#
# In the future, here will be a proper matrix operations.
#
# NOTE:
# 1. ThetaLinear represents a group of neurons, where each neuron receives an `x` (a signal value from 
#   the previous layer of neurons) and the components (the NM profile for each particular value of `x`).
# 2. Thus, each neuron in the ThetaLinear layer shares the same input profile, but the reaction to that 
#   profile should be individual. So, we could sum the whole input NM profile across neuromodulator 
#   dimensions to obtain a cumulative value for each type of incoming neuromodulator.
# 3. To simulate the variable sensitivities of each particular neuron in ThetaLinear to each particular 
#   neuromodulator, we could introduce for each neuron a weight matrix. That weight matrix will contain 
#   the multiplicative term of the neuron for the particular neuromodulator.
# 4. Along with weight matrices, we also need to introduce bias matrices, which represent the neuron's 
#   own contribution to each type of neuromodulator.
# 5. Thus, by obtaining a cumulative neuromodulation environment for the group of neurons (summation 
#   over neuromodulator dimention) and by applying individual (per neuron) weights and bias matrices 
#   to that cumulative environment, we will obtain an internal "influential matrices" for each neuron in a group.
#
class DyNAFThetaLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta_modes_in: int,
        theta_modes_out: int,
        theta_full_features: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super(DyNAFThetaLinear, self).__init__(in_features, out_features, **kwargs)

        self.theta_modes_in = theta_modes_in
        self.theta_modes_out = theta_modes_out
        self.theta_feautures = out_features if theta_full_features else 1

        # Calculate theta io size.
        self.theta_in = self.theta_modes_in * in_features + out_features
        self.theta_out = self.theta_modes_out * self.theta_feautures * 4

        # Define an additional linear layer for theta transformation.
        self.theta_transformation = nn.Linear(
            in_features=self.theta_in,
            out_features=self.theta_out,
            bias=True,
        )

        pass

    def forward(
        self,
        x: torch.Tensor,  # [batch, <unknown_dims>, in_features]
        components: torch.Tensor,  # [batch, <unknown_dims>, components_in, in_features]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply the parent class's linear transformation to x.
        x_transformed = super(DyNAFThetaLinear, self).forward(x)

        # Flatten: [batch, <unknown_dims>, components_in, in_features] -> [batch, <unknown_dims>, components_in*in_features]
        components_flat = components.flatten(-2)

        # Concat x_transformed with input components to form the activation state for Theta-transformation.
        activation_state = torch.cat([x_transformed, components_flat], dim=-1)

        # Apply linear transformation to activation_state to transform it into param_quads.
        param_quads = self.theta_transformation(activation_state)

        # Reshape param_quads to match the shape of input modes for DyNAFActivation.
        param_quads_shape = [
            *x_transformed.shape[:-1],
            self.theta_modes_out,
            4,
            self.theta_feautures,
        ]
        param_quads = param_quads.reshape(param_quads_shape)

        return x_transformed, param_quads
