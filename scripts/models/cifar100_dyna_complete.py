import torch
import torch.nn as nn

# from dyna import ThetaInput, ThetaLinear, ModulatedActivation
from dyna import ThetaInput, ThetaOutput, ModulatedActivation
from dyna.theta_linear_c import ThetaLinear


class CIFAR100DyNAComplete(nn.Module):
    def __init__(
        self,
    ):
        super(CIFAR100DyNAComplete, self).__init__()

        count_modes = 7
        dynamic_range = 7.5

        self.activation_classic = nn.ReLU # or None

        self.a_conv_pre = nn.Conv2d(3, 32, 3, 1, 1)
        self.a_activation_pre = ModulatedActivation(
            passive=True,
            count_modes=count_modes,
            features=32,
            theta_dynamic_range=dynamic_range,
        ) if self.activation_classic is None else self.activation_classic()
        self.a_conv_post = nn.Conv2d(32, 32, 3, 2, 1)
        self.a_activation_post = ModulatedActivation(
            passive=True,
            count_modes=count_modes,
            features=32,
            theta_dynamic_range=dynamic_range,
        ) if self.activation_classic is None else self.activation_classic()
        self.a_layer_norm = nn.LayerNorm([16, 16])

        self.b_conv_pre = nn.Conv2d(32, 32, 3, 1, 1)
        self.b_activation_pre = ModulatedActivation(
            passive=True,
            count_modes=count_modes,
            features=32,
            theta_dynamic_range=dynamic_range,
        ) if self.activation_classic is None else self.activation_classic()
        self.b_conv_post = nn.Conv2d(32, 32, 3, 2, 1)
        self.b_activation_post = ModulatedActivation(
            passive=True,
            count_modes=count_modes,
            features=32,
            theta_dynamic_range=dynamic_range,
        ) if self.activation_classic is None else self.activation_classic()
        self.b_layer_norm = nn.LayerNorm([8, 8])

        self.c_conv_pre = nn.Conv2d(32, 32, 3, 1, 1)
        self.c_activation_pre = ModulatedActivation(
            passive=True,
            count_modes=count_modes,
            features=32,
            theta_dynamic_range=dynamic_range,
        ) if self.activation_classic is None else self.activation_classic()
        self.c_conv_post = nn.Conv2d(32, 32, 3, 2, 1)

        self.d_input = ThetaInput(
            in_features=512,
            out_features=128,
            theta_modes_out=count_modes,
            theta_dynamic_range=dynamic_range,
        )
        self.d_linear = ThetaLinear(
            in_features=128,
            out_features=96,
            theta_components_in=count_modes,
            theta_modes_out=count_modes,
        )
        self.d_activation = ModulatedActivation()

        self.e_linear = ThetaLinear(
            in_features=96,
            out_features=100,
            theta_components_in=count_modes,
            theta_modes_out=count_modes,
        )
        self.e_activation = ModulatedActivation()
        self.e_output = ThetaOutput(
            in_features=100,
            theta_components_in=count_modes,
        )

        self.dropout = nn.Dropout(p=0.25)
        self.log_softmax = nn.LogSoftmax(dim=1)

        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.contiguous()

        x = self.a_conv_pre(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.a_activation_pre(x)
        x = x.x if self.activation_classic is None else x
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.a_conv_post(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.a_activation_post(x)
        x = x.x if self.activation_classic is None else x
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.a_layer_norm(x)

        x = self.b_conv_pre(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.b_activation_pre(x)
        x = x.x if self.activation_classic is None else x
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.b_conv_post(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.b_activation_post(x)
        x = x.x if self.activation_classic is None else x
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.b_layer_norm(x)

        x = self.c_conv_pre(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.c_activation_pre(x)
        x = x.x if self.activation_classic is None else x
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.c_conv_post(x)

        x = x.flatten(1)
        x = self.dropout(x)

        signal = self.d_input(x)
        signal = self.d_linear(signal)
        signal = self.d_activation(signal)

        signal = self.e_linear(signal)
        signal = self.e_activation(signal)
        x = self.e_output(signal)

        x = self.log_softmax(x)

        return x
