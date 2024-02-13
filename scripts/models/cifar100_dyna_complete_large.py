import torch
import torch.nn as nn

from dyna import ThetaInput, ThetaLinear, ModulatedActivation


class CIFAR100DyNACompleteLarge(nn.Module):
    def __init__(
        self,
    ):
        super(CIFAR100DyNACompleteLarge, self).__init__()

        count_modes = 17
        dynamic_range = 10.0
        conv_features_head = 64
        conv_features_tail = 32

        self.a_conv_pre = nn.Conv2d(3, conv_features_head, 3, 1, 1)
        self.a_activation_pre = ModulatedActivation(
            passive=True,
            count_modes=count_modes,
            features=conv_features_head,
            theta_dynamic_range=dynamic_range,
        )
        self.a_conv_post = nn.Conv2d(conv_features_head, conv_features_head, 3, 2, 1)
        self.a_activation_post = ModulatedActivation(
            passive=True,
            count_modes=count_modes,
            features=conv_features_head,
            theta_dynamic_range=dynamic_range,
        )
        self.a_layer_norm = nn.LayerNorm([16, 16])

        self.b_conv_pre = nn.Conv2d(conv_features_head, conv_features_head, 3, 1, 1)
        self.b_activation_pre = ModulatedActivation(
            passive=True,
            count_modes=count_modes,
            features=conv_features_head,
            theta_dynamic_range=dynamic_range,
        )
        self.b_conv_post = nn.Conv2d(conv_features_head, conv_features_head, 3, 2, 1)
        self.b_activation_post = ModulatedActivation(
            passive=True,
            count_modes=count_modes,
            features=conv_features_head,
            theta_dynamic_range=dynamic_range,
        )
        self.b_layer_norm = nn.LayerNorm([8, 8])

        self.c_conv_pre = nn.Conv2d(conv_features_head, conv_features_head, 3, 1, 1)
        self.c_activation_pre = ModulatedActivation(
            passive=True,
            count_modes=count_modes,
            features=conv_features_head,
            theta_dynamic_range=dynamic_range,
        )
        self.c_conv_post = nn.Conv2d(conv_features_head, conv_features_tail, 3, 2, 1)

        self.d_input = ThetaInput(
            in_features=512,
            out_features=256,
            theta_modes=count_modes,
            theta_dynamic_range=dynamic_range,
        )
        self.d_linear = ThetaLinear(
            in_features=256,
            out_features=128,
            theta_components_in=count_modes,
            theta_modes_out=count_modes,
        )
        self.d_activation = ModulatedActivation()
        self.d_batch_norm = nn.BatchNorm1d(96)

        self.e_linear = ThetaLinear(
            in_features=128,
            out_features=128,
            theta_components_in=count_modes,
            theta_modes_out=count_modes,
        )
        self.e_activation = ModulatedActivation()
        self.e_batch_norm = nn.BatchNorm1d(100)

        self.output_linear = nn.Linear(128, 100)

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
        x = self.a_activation_pre(x).x
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.a_conv_post(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.a_activation_post(x).x
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.a_layer_norm(x)

        x = self.b_conv_pre(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.b_activation_pre(x).x
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.b_conv_post(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.b_activation_post(x).x
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.b_layer_norm(x)

        x = self.c_conv_pre(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.c_activation_pre(x).x
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.c_conv_post(x)

        x = x.flatten(1)
        x = self.dropout(x)

        signal = self.d_input(x)
        signal = self.d_linear(signal)
        signal = self.d_activation(signal)

        signal = self.e_linear(signal)
        signal = self.e_activation(signal)
        x = signal.x

        x = self.dropout(x)
        x = self.output_linear(x)
        x = self.log_softmax(x)

        return x
