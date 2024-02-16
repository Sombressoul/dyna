import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional

from dyna.signal import SignalModular, SignalComponential


class ThetaLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta_components_in: Optional[int] = 7,
        theta_modes_out: Optional[int] = 7,
        theta_heterogeneity: Optional[float] = 1.0e-2,
        eps: Optional[float] = 1.0e-5,
        **kwargs,
    ) -> None:
        super(ThetaLinear, self).__init__(in_features, out_features, **kwargs)

        # ================================================================================= #
        # ____________________________> Arguments.
        # ================================================================================= #
        self.in_features = in_features
        self.out_features = out_features
        self.theta_components_in = theta_components_in
        self.theta_modes_out = theta_modes_out
        self.theta_heterogeneity = theta_heterogeneity
        self.eps = eps

        # ================================================================================= #
        # ____________________________> Weights.
        # ================================================================================= #
        # Define per-neuron sensetivity to input potential signals.
        sensitivity_input_potential = torch.empty(
            [
                self.out_features,
                self.in_features,
            ]
        )
        sensitivity_input_potential = (
            nn.init.normal_(
                tensor=sensitivity_input_potential,
                mean=1.0,
                std=self.theta_heterogeneity,
            )
            / self.in_features
        )
        self.sensitivity_input_potential = nn.Parameter(
            data=sensitivity_input_potential,
        )

        # Define per-neuron sensetivity to input modulation signals.
        sensitivity_input_componential = torch.empty(
            [
                self.out_features,
                self.in_features,
                self.theta_components_in,
            ]
        )
        sensitivity_input_componential = (
            nn.init.normal_(
                tensor=sensitivity_input_componential,
                mean=1.0,
                std=self.theta_heterogeneity,
            )
            / self.in_features
        )
        self.sensitivity_input_componential = nn.Parameter(
            data=sensitivity_input_componential,
        )

        # Define componential input interference matrices.
        input_componential_interference = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
                self.theta_components_in,
            ]
        )
        input_componential_interference = nn.init.uniform_(
            tensor=input_componential_interference,
            a=-math.sqrt(math.pi / self.theta_components_in),
            b=+math.sqrt(math.pi / self.theta_components_in),
        )
        self.input_componential_interference = nn.Parameter(
            data=input_componential_interference,
        )

        # Define potential input weights for modulation of interferred components.
        input_modulation_weights = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
            ]
        )
        input_modulation_weights = nn.init.normal_(
            tensor=input_modulation_weights,
            mean=1.0,
            std=self.theta_heterogeneity,
        )
        self.input_modulation_weights = nn.Parameter(
            data=input_modulation_weights,
        )

        # Internal state normalization W&B.
        state_normalization_weight = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
            ]
        )
        state_normalization_weight = nn.init.normal_(
            tensor=state_normalization_weight,
            mean=1.0,
            std=self.theta_heterogeneity,
        )
        self.state_normalization_weight = nn.Parameter(state_normalization_weight)

        state_normalization_bias = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
            ]
        )
        state_normalization_bias = nn.init.normal_(
            tensor=state_normalization_bias,
            mean=0.0,
            std=self.theta_heterogeneity,
        )
        self.state_normalization_bias = nn.Parameter(state_normalization_bias)

        # Define potential distortion matrices.
        potential_distortion = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
            ]
        )
        potential_distortion = nn.init.uniform_(
            tensor=potential_distortion,
            a=-math.sqrt(math.pi / self.theta_components_in),
            b=+math.sqrt(math.pi / self.theta_components_in),
        )
        self.potential_distortion = nn.Parameter(potential_distortion)

        # Define matrices for alphas.
        alphas = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
                self.theta_modes_out,
            ]
        )
        alphas = nn.init.uniform_(
            tensor=alphas,
            a=-math.sqrt(math.pi / self.theta_components_in),
            b=+math.sqrt(math.pi / self.theta_components_in),
        )
        self.alphas = nn.Parameter(alphas)

        # Define matrices for betas.
        betas = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
                self.theta_modes_out,
            ]
        )
        betas = nn.init.uniform_(
            tensor=betas,
            a=-math.sqrt(math.pi / self.theta_components_in),
            b=+math.sqrt(math.pi / self.theta_components_in),
        )
        self.betas = nn.Parameter(betas)

        # Define matrices for gammas.
        gammas = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
                self.theta_modes_out,
            ]
        )
        gammas = nn.init.uniform_(
            tensor=gammas,
            a=-math.sqrt(math.pi / self.theta_components_in),
            b=+math.sqrt(math.pi / self.theta_components_in),
        )
        self.gammas = nn.Parameter(gammas)

        # Define matrices for deltas.
        deltas = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
                self.theta_modes_out,
            ]
        )
        deltas = nn.init.uniform_(
            tensor=deltas,
            a=-math.sqrt(math.pi / self.theta_components_in),
            b=+math.sqrt(math.pi / self.theta_components_in),
        )
        self.deltas = nn.Parameter(deltas)

        pass

    def forward(
        self,
        x: SignalComponential,
    ) -> SignalModular:
        x_signal = x.x
        x_componential = x.components

        # ================================================================================= #
        # ____________________________> Input: potential.
        # ================================================================================= #
        sensitivity_input_potential = torch.einsum(
            "b...i,oi -> b...oi",
            x_signal,
            self.sensitivity_input_potential,
        )
        input_potential = sensitivity_input_potential.sum(-1)
        # input_potential: [b, *, features_out]

        # ================================================================================= #
        # ____________________________> Input: componential.
        # ================================================================================= #
        # [b, *, components, features_in] -> [b, *, features_in, components]
        x_componential = x_componential.permute(
            [
                *[i for i in range(len(x_componential.shape[:-2]))],
                -1,
                -2,
            ]
        )
        sensitivity_input_componential = torch.einsum(
            "b...ij,kij -> b...kij",
            x_componential,
            self.sensitivity_input_componential,
        )
        input_componential = sensitivity_input_componential.sum(-2)
        # input_componential: [b, *, features_out, components]

        # Apply componential input internal interference.
        input_componential_interference = torch.einsum(
            "b...ij,ijk -> b...ik",
            input_componential,
            self.input_componential_interference,
        )

        # ================================================================================= #
        # ____________________________> Intermodulation.
        # ================================================================================= #
        # Modulate componential input by weighted potential input.
        potential_modulation = input_potential.unsqueeze(-1)
        potential_modulation = (
            potential_modulation
            * self.input_modulation_weights.reshape(
                [
                    *[1 for _ in range(len(input_potential.shape[:-1]))],
                    *self.input_modulation_weights.shape,
                ]
            )
        )
        potential_modulation = potential_modulation + 1.0

        # ================================================================================= #
        # ____________________________> Internal state.
        # ================================================================================= #
        # Calculate and normalize internal state.
        internal_state = input_componential_interference * potential_modulation
        internal_state = F.layer_norm(
            input=internal_state,
            normalized_shape=internal_state.shape[-2:],
            weight=self.state_normalization_weight,
            bias=self.state_normalization_bias,
            eps=self.eps,
        )

        # Produce a potential distortion signal based on internal state.
        potential_distortion = torch.einsum(
            "b...ijk,ikl -> b...ijl",
            internal_state.unsqueeze(-2),
            self.potential_distortion.unsqueeze(-1),
        )
        potential_distortion = potential_distortion.reshape(input_potential.shape)

        # ================================================================================= #
        # ____________________________> Output: potential.
        # ================================================================================= #
        # Apply distortion to weighted signal to get resulting output.
        output_potential = input_potential + potential_distortion

        # ================================================================================= #
        # ____________________________> Output: modular.
        # ================================================================================= #
        # Derive alphas from internal state.
        alphas = internal_state.unsqueeze(-2)
        alphas = torch.einsum(
            "b...ijk,ikl -> b...ijl",
            alphas,
            self.alphas,
        )

        # Derive betas from internal state.
        betas = internal_state.unsqueeze(-2)
        betas = torch.einsum(
            "b...ijk,ikl -> b...ijl",
            betas,
            self.betas,
        )

        # Derive gammas from internal state.
        gammas = internal_state.unsqueeze(-2)
        gammas = torch.einsum(
            "b...ijk,ikl -> b...ijl",
            gammas,
            self.gammas,
        )

        # Derive deltas from internal state.
        deltas = internal_state.unsqueeze(-2)
        deltas = torch.einsum(
            "b...ijk,ikl -> b...ijl",
            deltas,
            self.deltas,
        )

        # Modify modulators to appropriate scales.
        # alphas = alphas # No changes for alphas.
        # betas = betas * math.log(self.theta_modes_out)
        # gammas = gammas # No changes for gammas.
        # deltas = deltas # No changes for deltas.

        # Combine modular output.
        output_modular = torch.cat([alphas, betas, gammas, deltas], dim=-2)
        output_modular = output_modular.permute(
            [
                *[i for i in range(len(output_modular.shape[:-3]))],
                -1,
                -2,
                -3,
            ]
        )

        # ================================================================================= #
        # ____________________________> End.
        # ================================================================================= #

        return SignalModular(
            x=output_potential,
            modes=output_modular,
        )
