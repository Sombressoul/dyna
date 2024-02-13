import torch
import torch.nn as nn
import math

from typing import Tuple, Optional


# NOTE: work-in-progress.
class DyNAFThetaLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta_modes_in: int,
        theta_modes_out: int,
        theta_full_features: Optional[bool] = True,
        theta_normalization: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super(DyNAFThetaLinear, self).__init__(in_features, out_features, **kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.theta_modes_in = theta_modes_in
        self.theta_modes_out = theta_modes_out
        self.theta_feautures = out_features if theta_full_features else 1
        self.theta_normalization = theta_normalization

        # Define per-neuron sesitivity to particular inputs.
        individual_sensetivity = torch.empty([self.out_features, self.in_features, 1])
        individual_sensetivity = self._initializer_individual(individual_sensetivity)
        self.individual_sensetivity = nn.Parameter(individual_sensetivity)

        # Define per-neuron sensetivity to cumulative neuromodulatory environment.
        env_sensetivity = torch.empty([self.in_features, self.theta_modes_in])
        env_sensetivity = self._initializer_env(env_sensetivity)
        self.env_sensetivity = nn.Parameter(env_sensetivity)

        # Define per-neuron bias for NM environment (intrinsic contributions).
        env_bias = torch.empty([self.in_features, self.theta_modes_in])
        env_bias = self._initializer_env(env_bias)
        self.env_bias = nn.Parameter(env_bias)

        # Define per-neuron env sensitivity normalization. It can be loosely
        # analogous to the regulatory mechanisms in biological systems.
        self.env_norm = (
            nn.LayerNorm([self.in_features, self.theta_modes_in])
            if self.theta_normalization
            else nn.Identity()
        )

        # Define perceptual matrices, which are necessary to calculate a
        # resulting perceptual_x for each neuron.
        perception = torch.empty([self.in_features, self.theta_modes_in, 1])
        perception = self._initializer_perception(perception)
        self.perception = nn.Parameter(perception)

        # Define perceptual_x bias.
        perceptual_bias = torch.empty([self.in_features])
        perceptual_bias = self._initializer_perception(perceptual_bias)
        self.perceptual_bias = nn.Parameter(perceptual_bias)

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
        std = math.sqrt(1 / self.theta_modes_in)
        with torch.no_grad():
            return nn.init.normal_(x, mean=0.0, std=std)

    def _initializer_individual(
        self,
        x,
    ) -> torch.Tensor:
        with torch.no_grad():
            return nn.init.normal_(x, mean=0.0, std=1.0)

    def forward(
        self,
        x: torch.Tensor,  # [batch, <unknown_dims>, in_features]
        components: torch.Tensor,  # [batch, <unknown_dims>, components_in, in_features]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #
        #       ThetaLinear represents a group of neurons, where each neuron receives an `x` (a signal value from
        #   the previous layer of neurons) and the components (the NM profile for each particular value of `x`).
        #       Thus, each neuron in the ThetaLinear layer shares the same input profile, but the reaction to that
        #   profile should be individual. So, we could weight each input profile for each neuron in the ThetaLinear,
        #   and then sum the personal input NM profile across neuromodulator dimensions to obtain a cumulative value
        #   for each type of incoming neuromodulator for each neuron.
        #       To simulate the variable sensitivities of each particular neuron in ThetaLinear to each particular
        #   neuromodulator, we could introduce for each neuron a weight matrix. That weight matrix will contain
        #   the multiplicative term of the neuron for the particular neuromodulator.
        #       Along with weight matrices, we also need to introduce bias matrices, which represent the neuron's
        #   own contribution to each type of neuromodulator.
        #       Thus, by obtaining a cumulative neuromodulation environment for the group of neurons (summation
        #   over neuromodulator dimention) and by applying individual (per neuron) weights and bias matrices
        #   to that cumulative environment, we will obtain an internal "influential matrices" for each neuron in a group.
        #
        extra_dims = [1 for _ in range(len(components.shape[0:-2]))]
        components_per_input = components.permute(
            [*range(len(components.shape[:-2])), -1, -2]
        )
        components_per_input = components_per_input.unsqueeze(-3)
        #
        #       Individual sensetivity here is the product of individual input NM profiles weighted by their contribution
        #   per each output neuron.
        #
        individual_sensetivity = (
            self.individual_sensetivity.reshape(
                [
                    *[1 for _ in range(len(components.shape[:-2]))],
                    *self.individual_sensetivity.shape,
                ]
            )
            * components_per_input
        )
        #
        #       Cumulative environment here is the sum of the weighted contributions of each input neuron
        #   over dimention of output neurons. Thus, we obtain a "perceptual environmental impact" of each 
        #   input neuron (a perceived environmental variables, that contributes to the outputs).
        #
        cumulative_env = torch.sum(individual_sensetivity, dim=-3)
        env_sensed = cumulative_env * self.env_sensetivity.reshape(
            [*extra_dims, *self.env_sensetivity.shape]
        )
        env_biased = env_sensed + self.env_bias.reshape(
            [*extra_dims, *self.env_bias.shape]
        )
        env_normalized = self.env_norm(env_biased)

        #
        #       We have to modulate incoming signals by calculated per-neuron environmental
        #   influence and then multiply it with per-neuron perceptual matrices to obtain
        #   the actual perceptual x, which will mirror the environmental contribution of
        #   each neuromodulator to the perception of incoming x.
        #       Thus, we obtain a "perceptual x" per each neuron.
        #       Then, we just transform the perceptual x to the output x by fully connected layer.
        #
        perceptual_x = x.unsqueeze(-1) * env_normalized
        perceptual_x = torch.einsum("...ij,ijk -> ...ik", perceptual_x, self.perception)
        perceptual_x = perceptual_x.squeeze(-1)
        perceptual_x = perceptual_x + self.perceptual_bias
        transformed_x = super(DyNAFThetaLinear, self).forward(perceptual_x)

        exit()
        #
        # TODO:
        #       Using the individual_sensetivity, but summed across dimention of output neurons,
        #   we can obtain a basis for deriving new neuromodulatory param-quads.
        # NOTE:
        #       Consider modulation of new param-quads by newly derived transformed_x in some way.

        return transformed_x
