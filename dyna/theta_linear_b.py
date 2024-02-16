import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Union

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

        self.in_features = in_features
        self.out_features = out_features
        self.theta_components_in = theta_components_in
        self.theta_modes_out = theta_modes_out
        self.theta_heterogeneity = theta_heterogeneity
        self.eps = eps

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

        # Define scales for percieved potential inputs.
        input_potential_scale = torch.empty(
            [
                self.out_features,
            ]
        )
        input_potential_scale = nn.init.normal_(
            tensor=input_potential_scale,
            mean=1.0,
            std=self.theta_heterogeneity,
        )
        self.input_potential_scale = nn.Parameter(
            data=input_potential_scale,
        )

        # Define bias for percieved potential inputs.
        input_potential_bias = torch.empty(
            [
                self.out_features,
            ]
        )
        input_potential_bias = nn.init.normal_(
            tensor=input_potential_bias,
            mean=0.0,
            std=self.theta_heterogeneity,
        )
        self.input_potential_bias = nn.Parameter(
            data=input_potential_bias,
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

        # Define scales for percieved componential inputs.
        input_componential_scale = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
            ]
        )
        input_componential_scale = nn.init.normal_(
            tensor=input_componential_scale,
            mean=1.0,
            std=self.theta_heterogeneity,
        )
        self.input_componential_scale = nn.Parameter(
            data=input_componential_scale,
        )

        # Define bias for percieved componential inputs.
        input_componential_bias = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
            ]
        )
        input_componential_bias = nn.init.normal_(
            tensor=input_componential_bias,
            mean=0.0,
            std=self.theta_heterogeneity,
        )
        self.input_componential_bias = nn.Parameter(
            data=input_componential_bias,
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

        # Define input modulation bias.
        input_modulation_bias = torch.empty(
            [
                self.out_features,
                self.theta_components_in,
            ]
        )
        input_modulation_bias = nn.init.normal_(
            tensor=input_modulation_bias,
            mean=0.0,
            std=self.theta_heterogeneity,
        )
        self.input_modulation_bias = nn.Parameter(
            data=input_modulation_bias,
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

        alphas_bias = torch.empty(
            [
                self.out_features,
                self.theta_modes_out,
            ]
        )
        alphas_bias = nn.init.normal_(
            tensor=alphas_bias,
            mean=0.0,
            std=self.theta_heterogeneity,
        )
        self.alphas_bias = nn.Parameter(alphas_bias)

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

        betas_bias = torch.empty(
            [
                self.out_features,
                self.theta_modes_out,
            ]
        )
        betas_bias = nn.init.normal_(
            tensor=betas_bias,
            mean=0.0,
            std=self.theta_heterogeneity,
        )
        self.betas_bias = nn.Parameter(betas_bias)

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

        gammas_bias = torch.empty(
            [
                self.out_features,
                self.theta_modes_out,
            ]
        )
        gammas_bias = nn.init.normal_(
            tensor=gammas_bias,
            mean=0.0,
            std=self.theta_heterogeneity,
        )
        self.gammas_bias = nn.Parameter(gammas_bias)

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

        deltas_bias = torch.empty(
            [
                self.out_features,
                self.theta_modes_out,
            ]
        )
        deltas_bias = nn.init.normal_(
            tensor=deltas_bias,
            mean=0.0,
            std=self.theta_heterogeneity,
        )
        self.deltas_bias = nn.Parameter(deltas_bias)

        pass

    def forward(
        self,
        x: SignalComponential,
    ) -> SignalModular:
        x_signal = x.x
        x_componential = x.components

        # Calculate potential inputs.
        sensitivity_input_potential = torch.einsum(
            "b...i,oi -> b...oi",
            x_signal,
            self.sensitivity_input_potential,
        )
        input_potential = sensitivity_input_potential.sum(-1)
        # input_potential: [b, *, features_out]

        # Apply potential scale and bias.
        input_potential = input_potential * self.input_potential_scale.reshape(
            [
                *[1 for _ in range(len(input_potential.shape[:-1]))],
                *self.input_potential_scale.shape,
            ]
        )
        input_potential = input_potential + self.input_potential_bias.reshape(
            [
                *[1 for _ in range(len(input_potential.shape[:-1]))],
                *self.input_potential_bias.shape,
            ]
        )

        # Calculate componential inputs.
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

        # Apply componential scale and bias.
        input_componential = input_componential * self.input_componential_scale.reshape(
            [
                *[1 for _ in range(len(input_componential.shape[:-2]))],
                *self.input_componential_scale.shape,
            ]
        )
        input_componential = input_componential + self.input_componential_bias.reshape(
            [
                *[1 for _ in range(len(input_componential.shape[:-2]))],
                *self.input_componential_bias.shape,
            ]
        )

        # Apply componential input internal interference.
        input_componential_interference = torch.einsum(
            "b...ij,ijk -> b...ik",
            input_componential,
            self.input_componential_interference,
        )

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
        potential_modulation = (
            potential_modulation
            + self.input_modulation_bias.reshape(
                [
                    *[1 for _ in range(len(potential_modulation.shape[:-2]))],
                    *self.input_modulation_bias.shape,
                ]
            )
        )

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

        # Apply distortion to weighted signal to get resulting output.
        output_potential = input_potential + potential_distortion

        # Derive alphas from internal state.
        alphas = internal_state.unsqueeze(-2)
        alphas = torch.einsum(
            "b...ijk,ikl -> b...ijl",
            alphas,
            self.alphas,
        )
        alphas = alphas + self.alphas_bias.reshape(
            [
                *[1 for _ in range(len(alphas.shape[:-3]))],
                *self.alphas_bias.shape,
            ]
        ).unsqueeze(-2)

        # Derive betas from internal state.
        betas = internal_state.unsqueeze(-2)
        betas = torch.einsum(
            "b...ijk,ikl -> b...ijl",
            betas,
            self.betas,
        )
        betas = betas + self.betas_bias.reshape(
            [
                *[1 for _ in range(len(betas.shape[:-3]))],
                *self.betas_bias.shape,
            ]
        ).unsqueeze(-2)

        # Derive gammas from internal state.
        gammas = internal_state.unsqueeze(-2)
        gammas = torch.einsum(
            "b...ijk,ikl -> b...ijl",
            gammas,
            self.gammas,
        )
        gammas = gammas + self.gammas_bias.reshape(
            [
                *[1 for _ in range(len(gammas.shape[:-3]))],
                *self.gammas_bias.shape,
            ]
        ).unsqueeze(-2)

        # Derive deltas from internal state.
        deltas = internal_state.unsqueeze(-2)
        deltas = torch.einsum(
            "b...ijk,ikl -> b...ijl",
            deltas,
            self.deltas,
        )
        deltas = deltas + self.deltas_bias.reshape(
            [
                *[1 for _ in range(len(deltas.shape[:-3]))],
                *self.deltas_bias.shape,
            ]
        ).unsqueeze(-2)

        # Modify modulators to appropriate scales.
        # alphas = alphas # No changes for alphas.
        betas = betas * math.log(self.theta_modes_out)
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

        # print("---")
        # print(f"{x_signal.shape=}")
        # print(f"{x_signal.mean().item()=}")
        # print(f"{x_signal.std().item()=}")
        # print(f"{(x_signal.sum(-1) / self.in_features).mean().item()=}")
        # print(f"{(x_signal.sum(-1) / self.in_features).std().item()=}")
        # print(f"{self.sensitivity_input_potential.shape=}")
        # print(f"{sensitivity_input_potential.shape=}")
        # print(f"{input_potential.shape=}")
        # print(f"{input_potential.mean().item()=}")
        # print(f"{input_potential.std().item()=}")
        # print(f"{potential_modulation.shape=}")
        # print(f"{potential_modulation.mean().item()=}")
        # print(f"{potential_modulation.std().item()=}")
        # print("---")
        # print(f"{x_componential.shape=}")
        # print(f"{x_componential.mean().item()=}")
        # print(f"{x_componential.std().item()=}")
        # print(f"{self.sensitivity_input_componential.shape=}")
        # print(f"{sensitivity_input_componential.shape=}")
        # print(f"{sensitivity_input_componential.mean().item()=}")
        # print(f"{sensitivity_input_componential.std().item()=}")
        # print(f"{input_componential.shape=}")
        # print(f"{input_componential.mean().item()=}")
        # print(f"{input_componential.std().item()=}")
        # print(f"{self.input_componential_interference.shape=}")
        # print(f"{input_componential_interference.shape=}")
        # print(f"{input_componential_interference.mean().item()=}")
        # print(f"{input_componential_interference.std().item()=}")
        # print("---")
        # print(f"{internal_state.shape=}")
        # print(f"{internal_state.mean().item()=}")
        # print(f"{internal_state.std().item()=}")
        # print("--- Output: potential")
        # print(f"{self.potential_distortion.shape=}")
        # print(f"{potential_distortion.shape=}")
        # print(f"{potential_distortion.mean().item()=}")
        # print(f"{potential_distortion.std().item()=}")
        # print(f"{output_potential.shape=}")
        # print(f"{output_potential.mean().item()=}")
        # print(f"{output_potential.std().item()=}")
        # print("--- Output: modular")
        # print(f"{self.alphas.shape=}")
        # print(f"{self.alphas_bias.shape=}")
        # print(f"{alphas.shape=}")
        # print(f"{alphas.mean().item()=}")
        # print(f"{alphas.std().item()=}")
        # print(f"{betas.shape=}")
        # print(f"{betas.mean().item()=}")
        # print(f"{betas.std().item()=}")
        # print(f"{gammas.shape=}")
        # print(f"{gammas.mean().item()=}")
        # print(f"{gammas.std().item()=}")
        # print(f"{deltas.shape=}")
        # print(f"{deltas.mean().item()=}")
        # print(f"{deltas.std().item()=}")
        # print(f"{output_modular.shape=}")
        # print(f"{output_modular.mean().item()=}")
        # print(f"{output_modular.std().item()=}")
        # exit()

        return SignalModular(
            x=output_potential,
            modes=output_modular,
        )
