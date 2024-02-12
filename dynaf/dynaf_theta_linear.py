import torch
import torch.nn as nn
import math

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

        self.in_features = in_features
        self.out_features = out_features
        self.theta_modes_in = theta_modes_in
        self.theta_modes_out = theta_modes_out
        self.theta_feautures = out_features if theta_full_features else 1

        # Define per-neuron sensetivity (to neuromodulatory environment) matrices.
        env_sensetivity = torch.empty([self.in_features, self.theta_modes_in])
        env_sensetivity = self._initializer_env(env_sensetivity)
        self.env_sensetivity = nn.Parameter(env_sensetivity)

        # Define per-neuron bias for NM environment (intrinsic contributions).
        env_bias = torch.empty([self.in_features, self.theta_modes_in])
        env_bias = self._initializer_env(env_bias)
        self.env_bias = nn.Parameter(env_bias)

        # Define per-neuron env sensitivity normalization. It can be loosely
        # analogous to the regulatory mechanisms in biological systems.
        self.env_norm = nn.LayerNorm([self.in_features, self.theta_modes_in])

        # Define perceptual matrices, which are necessary to calculate a
        # resulting perceptual_x for each neuron.
        perception = torch.empty([self.in_features, self.theta_modes_in, 1])
        perception = self._initializer_perception(perception)
        self.perception = nn.Parameter(perception)

        pass

    def _initializer_env(
        self,
        x,
    ) -> torch.Tensor:
        bound = self.theta_modes_in / self.in_features
        with torch.no_grad():
            return nn.init.uniform_(x, a=-bound, b=+bound)

    def _initializer_perception(
        self,
        x,
    ) -> torch.Tensor:
        bound = math.sqrt(3 / self.theta_modes_in)
        with torch.no_grad():
            return nn.init.uniform_(x, a=-bound, b=+bound)

    def forward(
        self,
        x: torch.Tensor,  # [batch, <unknown_dims>, in_features]
        components: torch.Tensor,  # [batch, <unknown_dims>, components_in, in_features]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        extra_dims = [1 for _ in range(len(components.shape[0:-2]))]
        cumulative_env = torch.sum(components, dim=-1).unsqueeze(-2)

        env_sensed = cumulative_env * self.env_sensetivity.reshape(
            [*extra_dims, *self.env_sensetivity.shape]
        )
        env_biased = env_sensed + self.env_bias.reshape(
            [*extra_dims, *self.env_bias.shape]
        )
        env_normalized = self.env_norm(env_biased)

        # We have to modulate incoming signals by calculated per-neuron environmental
        # influence and then multiply it with per-neuron perceptual matrices to obtain 
        # the actual perceptual x, which will mirror the environmental contribution of 
        # each neuromodulator to the perception of incoming x.
        # Thus, we obtain a "perceptual x" per each neuron.
        perceptual_x = x.unsqueeze(-1) * env_normalized
        perceptual_x = torch.einsum("...ij,ijk -> ...ik", perceptual_x, self.perception)
        perceptual_x = perceptual_x.squeeze(-1)

        print(f"x.shape: {x.shape}")
        print(f"env_normalized.shape: {env_normalized.shape}")
        print(f"perceptual_x.shape: {perceptual_x.shape}")
        exit()

        return ...
