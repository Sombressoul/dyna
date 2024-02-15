import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Union

from dyna.signal import SignalModular, SignalComponential


#
# TODO:
#   Emission matices (alphas, betas, gammas, deltas) proper initialization.
#
class ThetaLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta_components_in: Optional[int] = 7,
        theta_modes_out: Optional[int] = 7,
        theta_full_features: Optional[bool] = True,
        theta_normalize_env_input: Optional[bool] = True,
        theta_normalize_env_output: Optional[bool] = True,
        theta_dynamic_range: Optional[float] = 7.5,
        **kwargs,
    ) -> None:
        super(ThetaLinear, self).__init__(in_features, out_features, **kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.theta_components_in = theta_components_in
        self.theta_modes_out = theta_modes_out
        self.theta_quad_out = self.theta_modes_out * 4
        self.theta_feautures = out_features if theta_full_features else 1
        self.theta_normalize_env_input = theta_normalize_env_input
        self.theta_normalize_env_output = theta_normalize_env_output
        self.theta_dynamic_range = theta_dynamic_range

        # Magic constants.
        init_root_base = 3

        # Define per-neuron sesitivity to particular inputs.
        individual_sensetivity = torch.empty([self.out_features, self.in_features, 1])
        individual_sensetivity = torch.nn.init.uniform_(
            tensor=individual_sensetivity,
            a=-1.0,
            b=+1.0,
        )
        # individual_sensetivity = individual_sensetivity + 1.0
        self.individual_sensetivity = nn.Parameter(individual_sensetivity)

        # Define per-neuron sensetivity to cumulative neuromodulatory environment.
        env_sensetivity = torch.empty([self.in_features, self.theta_components_in])
        env_sensetivity = torch.nn.init.uniform_(
            tensor=env_sensetivity,
            a=-self.theta_components_in / self.in_features,
            b=+self.theta_components_in / self.in_features,
        )
        self.env_sensetivity = nn.Parameter(env_sensetivity)

        # Define per-neuron bias for NM environment (intrinsic contributions).
        env_bias = torch.empty([self.in_features, self.theta_components_in])
        env_bias = torch.nn.init.uniform_(
            tensor=env_bias,
            a=-1 / self.theta_components_in,
            b=+1 / self.theta_components_in,
        )
        self.env_bias = nn.Parameter(env_bias)

        # Define per-neuron env sensitivity normalization. It can be loosely
        # analogous to the regulatory mechanisms in biological systems.
        self.norm_env_input = (
            nn.LayerNorm([self.in_features, self.theta_components_in])
            if self.theta_normalize_env_input
            else nn.Identity()
        )
        self.norm_env_output = (
            nn.LayerNorm([self.out_features, self.theta_components_in])
            if self.theta_normalize_env_output
            else nn.Identity()
        )

        # Define perceptual matrices, which are necessary to calculate a
        # resulting perceptual_x for each neuron.
        perception = torch.empty([self.in_features, self.theta_components_in, 1])
        perception = torch.nn.init.uniform_(
            tensor=perception,
            a=-math.sqrt(init_root_base / self.theta_components_in),
            b=+math.sqrt(init_root_base / self.theta_components_in),
        )
        self.perception = nn.Parameter(perception)

        # Define perceptual_x bias.
        perceptual_bias = torch.empty([self.in_features])
        perceptual_bias = torch.nn.init.uniform_(
            tensor=perceptual_bias,
            a=-1 / self.in_features,
            b=+1 / self.in_features,
        )
        self.perceptual_bias = nn.Parameter(perceptual_bias)

        # Define neuromodulatory emitting matrices.
        emission = torch.empty(
            [
                self.theta_feautures,
                self.theta_components_in,
                self.theta_quad_out,
            ]
        )
        alphas = emission[:, :, 0::4]
        betas = emission[:, :, 1::4]
        gammas = emission[:, :, 2::4]
        deltas = emission[:, :, 3::4]

        emission_std = math.sqrt(1 / self.theta_modes_out)
        alphas = torch.nn.init.normal_(alphas, mean=0.0, std=emission_std)
        betas = torch.nn.init.normal_(betas, mean=0.0, std=emission_std)
        gammas = torch.nn.init.normal_(gammas, mean=0.0, std=emission_std)
        deltas = torch.nn.init.normal_(deltas, mean=0.0, std=emission_std)

        emission[:, :, 0::4] = alphas
        emission[:, :, 1::4] = betas
        emission[:, :, 2::4] = gammas
        emission[:, :, 3::4] = deltas

        self.emission = nn.Parameter(emission)

        # Define neuromodulatory emission scale matrices.
        emission_scale = torch.empty(
            [
                self.out_features,
                self.theta_quad_out,
            ]
        )
        emission_scale = torch.nn.init.uniform_(
            tensor=emission_scale,
            a=-1 / self.theta_quad_out,
            b=+1 / self.theta_quad_out,
        )
        emission_scale = emission_scale + 1.0
        self.emission_scale = nn.Parameter(emission_scale)

        # Define neuromodulatory emission bias matrices (intrinsic contributions).
        emission_bias = torch.empty(
            [
                self.out_features,
                self.theta_quad_out,
            ]
        )
        emission_bias = torch.nn.init.uniform_(
            tensor=emission_bias,
            a=-1 / self.theta_quad_out,
            b=+1 / self.theta_quad_out,
        )
        self.emission_bias = nn.Parameter(emission_bias)

        pass

    def forward(
        self,
        x: Union[torch.Tensor, SignalComponential],
        components: Optional[torch.Tensor] = None,
    ) -> SignalModular:
        if isinstance(x, SignalComponential):
            assert (
                components is None
            ), "components must be None when x is SignalComponential"
            signal = x
        else:
            assert (
                components is not None
            ), "components must be provided when x is not SignalComponential"
            signal = SignalComponential(
                x=x,
                components=components,
            )

        #
        #       ThetaLinear represents a group of neurons, where each neuron receives an `x` (a
        #   signal value from the previous layer of neurons) and the components (the NM profile
        #   for each particular value of `x`).
        #       Thus, each neuron in the ThetaLinear layer shares the same input profile, but the
        #   reaction to that profile should be individual. So, we could weight each input profile
        #   for each neuron in the ThetaLinear, and then sum the personal input NM profile across
        #   neuromodulator dimensions to obtain a cumulative value for each type of incoming
        #   neuromodulator for each neuron.
        #       To simulate the variable sensitivities of each particular neuron in ThetaLinear
        #   to each particular neuromodulator, we could introduce for each neuron a weight matrix.
        #   That weight matrix will contain the multiplicative term of the neuron for the particular
        #   neuromodulator.
        #       Along with weight matrices, we also need to introduce bias matrices, which represent
        #   the neuron's own contribution to each type of neuromodulator.
        #       Thus, by obtaining a cumulative neuromodulation environment for the group of neurons
        #   (summation over neuromodulator dimention) and by applying individual (per neuron)
        #   weights and bias matrices to that cumulative environment, we will obtain an internal
        #   "influential matrices" for each neuron in a group.
        #
        extra_dims = [1 for _ in range(len(signal.components.shape[0:-2]))]
        components_per_input = signal.components.permute(
            [*range(len(signal.components.shape[:-2])), -1, -2]
        )
        components_per_input = components_per_input.unsqueeze(-3)
        #
        #       Individual sensetivity here is the product of individual input NM profiles weighted
        #   by their contribution per each output neuron.
        #
        individual_sensetivity = (
            self.individual_sensetivity.reshape(
                [
                    *[1 for _ in range(len(signal.components.shape[:-2]))],
                    *self.individual_sensetivity.shape,
                ]
            )
            * components_per_input
        )
        #
        #       Cumulative environment here is the sum of the weighted contributions of each input
        #   neuron over dimention of output neurons. Thus, we obtain a "perceptual environmental
        #   impact" of each input neuron (a perceived environmental variables, that contributes to
        #   the outputs).
        #       In other words, it is a per-neuron NM influence (weighted sum) with respect to
        #   the inputs.
        #
        cumulative_env_inputs = torch.sum(individual_sensetivity, dim=-3)
        env_sensed = cumulative_env_inputs * self.env_sensetivity.reshape(
            [*extra_dims, *self.env_sensetivity.shape]
        )
        env_biased = env_sensed + self.env_bias.reshape(
            [*extra_dims, *self.env_bias.shape]
        )
        env_normalized = self.norm_env_input(env_biased)

        #
        #       We have to modulate incoming signals by calculated per-neuron environmental
        #   influence and then multiply it with per-neuron perceptual matrices to obtain
        #   the actual perceptual x, which will mirror the environmental contribution of
        #   each neuromodulator to the perception of incoming x.
        #       Thus, we obtain a "perceptual x" per each neuron.
        #       Then, we just transform the perceptual x to the output x by fully connected layer.
        #
        perceptual_x = signal.x.unsqueeze(-1) * env_normalized
        perceptual_x = torch.einsum("...ij,ijk -> ...ik", perceptual_x, self.perception)
        perceptual_x = perceptual_x.squeeze(-1)
        perceptual_x = perceptual_x + self.perceptual_bias
        transformed_x = super(ThetaLinear, self).forward(perceptual_x)

        #
        #       Here we could look at the transformed_x as at the neurons action potential, since
        #   it was derived from the per-neuron perceptual environment. Thus, we could use it as
        #   a multiplicative term for cumulative sum of weighted neuromodulatory contributions of
        #   each input neuron.
        #       In other words, it is a per-neuron NM influence (weighted sum) with respect to
        #   the outputs.
        #
        cumulative_env_outputs = torch.sum(individual_sensetivity, dim=-2)
        modulated_env = cumulative_env_outputs * transformed_x.unsqueeze(-1)
        modulated_env = self.norm_env_output(modulated_env)
        emission_raw = torch.einsum(
            "...ij,ijk -> ...ik",
            modulated_env,
            self.emission,
        )
        # print(f"{modulated_env.shape=}")
        # print(f"{self.emission.shape=}")
        # print(f"{emission_raw.shape=}")
        # print(f"{modulated_env[0, 0]=}")
        # print(f"{self.emission[0, 0]=}")
        # print(f"{emission_raw[0, 0]=}")
        # exit()
        emission_scaled = torch.einsum(
            "...ij,ij -> ...ij",
            emission_raw,
            self.emission_scale,
        )
        emission_biased = emission_scaled + self.emission_bias.reshape(
            [
                *[1 for _ in range(len(emission_scaled.shape[:-2]))],
                *self.emission_bias.shape,
            ]
        )
        param_quads = emission_biased.reshape(
            [
                *emission_biased.shape[:-1],
                self.theta_modes_out,
                -1,
            ]
        )
        param_quads = param_quads.permute(
            [
                *[i for i in range(len(param_quads.shape[:-3]))],
                -2,
                -1,
                -3,  # Features last.
            ]
        )

        # # Prepare for STE trick.
        # param_quads_buffer = param_quads.clone()
        # # Move betas and gammas towards their std as mean.
        # param_quads_buffer[:, :, 1] = param_quads_buffer[:, :, 1] + param_quads_buffer[:, :, 1].std()
        # param_quads_buffer[:, :, 2] = param_quads_buffer[:, :, 2] + param_quads_buffer[:, :, 2].std()
        # # Scale deltas to dynamic range by std.
        # param_quads_buffer[:, :, 3] = (param_quads_buffer[:, :, 3] / param_quads_buffer[:, :, 3].std()) * self.theta_dynamic_range
        # # STE.
        # param_quads = param_quads + (param_quads_buffer - param_quads).detach()

        # print(f"{param_quads.shape=}")
        # print("-")
        # print(f"alphas: {param_quads[:, :, 0].mean().item()=}")
        # print(f"alphas: {param_quads[:, :, 0].std().item()=}")
        # print(f"alphas: {param_quads[:, :, 0].min().item()=}")
        # print(f"alphas: {param_quads[:, :, 0].max().item()=}")
        # print("-")
        # print(f"betas: {param_quads[:, :, 1].mean().item()=}")
        # print(f"betas: {param_quads[:, :, 1].std().item()=}")
        # print(f"betas: {param_quads[:, :, 1].min().item()=}")
        # print(f"betas: {param_quads[:, :, 1].max().item()=}")
        # print("-")
        # print(f"gammas: {param_quads[:, :, 2].mean().item()=}")
        # print(f"gammas: {param_quads[:, :, 2].std().item()=}")
        # print(f"gammas: {param_quads[:, :, 2].min().item()=}")
        # print(f"gammas: {param_quads[:, :, 2].max().item()=}")
        # print("-")
        # print(f"deltas: {param_quads[:, :, 3].mean().item()=}")
        # print(f"deltas: {param_quads[:, :, 3].std().item()=}")
        # print(f"deltas: {param_quads[:, :, 3].min().item()=}")
        # print(f"deltas: {param_quads[:, :, 3].max().item()=}")
        # exit()

        return SignalModular(
            x=transformed_x,
            modes=param_quads,
        )
