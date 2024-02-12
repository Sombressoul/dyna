import torch
import torch.nn as nn

from dynaf import DyNAFActivation, DyNAFThetaLinear


class CIFAR100DyNAFComplete(nn.Module):
    def __init__(
        self,
    ):
        super(CIFAR100DyNAFComplete, self).__init__()

        count_modes = 7
        expected_range = [-5.0, +5.0]

        self.a_conv_pre = nn.Conv2d(3, 32, 3, 1, 1)
        self.a_activation_pre = DyNAFActivation(
            passive=True,
            count_modes=count_modes,
            features=1,
            expected_input_min=expected_range[0],
            expected_input_max=expected_range[1],
        )
        self.a_conv_post = nn.Conv2d(32, 32, 3, 2, 1)
        self.a_activation_post = DyNAFActivation(
            passive=True,
            count_modes=count_modes,
            features=1,
            expected_input_min=expected_range[0],
            expected_input_max=expected_range[1],
        )
        self.a_layer_norm = nn.LayerNorm([16, 16])

        self.b_conv_pre = nn.Conv2d(32, 32, 3, 1, 1)
        self.b_activation_pre = DyNAFActivation(
            passive=True,
            count_modes=count_modes,
            features=1,
            expected_input_min=expected_range[0],
            expected_input_max=expected_range[1],
        )
        self.b_conv_post = nn.Conv2d(32, 32, 3, 2, 1)
        self.b_activation_post = DyNAFActivation(
            passive=True,
            count_modes=count_modes,
            features=1,
            expected_input_min=expected_range[0],
            expected_input_max=expected_range[1],
        )
        self.b_layer_norm = nn.LayerNorm([8, 8])

        self.c_conv_pre = nn.Conv2d(32, 32, 3, 1, 1)
        self.c_activation_pre = DyNAFActivation(
            passive=True,
            count_modes=count_modes,
            features=1,
            expected_input_min=expected_range[0],
            expected_input_max=expected_range[1],
        )
        self.c_conv_post = nn.Conv2d(32, 32, 3, 2, 1)
        self.c_activation_post = DyNAFActivation(
            passive=True,
            count_modes=count_modes,
            features=1,
            expected_input_min=expected_range[0],
            expected_input_max=expected_range[1],
        )
        self.c_layer_norm = nn.LayerNorm([4, 4])

        self.d_input_activation = DyNAFActivation(
            passive=True,
            count_modes=count_modes,
            features=1,
            expected_input_min=expected_range[0],
            expected_input_max=expected_range[1],
        )
        self.d_linear = DyNAFThetaLinear(
            in_features=512,
            out_features=96,
            theta_modes_in=count_modes,
            theta_modes_out=count_modes,
            theta_full_features=False,
        )
        self.d_activation = DyNAFActivation(
            passive=False,
        )
        self.d_batch_norm = nn.BatchNorm1d(96)

        self.e_linear = DyNAFThetaLinear(
            in_features=96,
            out_features=100,
            theta_modes_in=count_modes,
            theta_modes_out=count_modes,
            theta_full_features=False,
        )
        self.e_activation = DyNAFActivation(
            passive=False,
        )
        self.e_batch_norm = nn.BatchNorm1d(100)

        self.output_linear = nn.Linear(100, 100)

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
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.a_conv_post(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.a_activation_post(x)
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.a_layer_norm(x)

        x = self.b_conv_pre(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.b_activation_pre(x)
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.b_conv_post(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.b_activation_post(x)
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.b_layer_norm(x)

        x = self.c_conv_pre(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.c_activation_pre(x)
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.c_conv_post(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.c_activation_post(x)
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.c_layer_norm(x)

        x = x.flatten(1)
        x = self.dropout(x)

        x, cmp = self.d_input_activation(x, return_components=True)
        x, cmp = self.d_linear(x, cmp)
        x, cmp = self.d_activation(x, cmp, return_components=True)
        x = self.d_batch_norm(x)

        x, cmp = self.e_linear(x, cmp)
        x, cmp = self.e_activation(x, cmp, return_components=True)
        x = self.e_batch_norm(x)

        x = self.output_linear(x)
        x = self.log_softmax(x)

        return x