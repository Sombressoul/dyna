import torch
import torch.nn as nn
import torch.nn.functional as F

from dyna import ModulatedActivationBell


class CIFAR100DyNAActivationBellPerFeature(nn.Module):
    def __init__(
        self,
    ):
        super(CIFAR100DyNAActivationBellPerFeature, self).__init__()

        count_modes = 21
        conv_features = 32
        # Baseline: 38%
        # ModulatedActivationBell: 41%
        # Training time (s): 359.42
        self.baseline = False

        self.a_conv_pre = nn.Conv2d(3, conv_features, 3, 1, 1)
        self.a_activation_pre = ModulatedActivationBell(
            passive=False,
            count_modes=count_modes,
            features=conv_features,
        )
        self.a_conv_post = nn.Conv2d(conv_features, conv_features, 3, 2, 1)
        self.a_activation_post = ModulatedActivationBell(
            passive=False,
            count_modes=count_modes,
            features=conv_features,
        )
        self.a_layer_norm = nn.LayerNorm([16, 16])

        self.b_conv_pre = nn.Conv2d(conv_features, conv_features, 3, 1, 1)
        self.b_activation_pre = ModulatedActivationBell(
            passive=False,
            count_modes=count_modes,
            features=conv_features,
        )
        self.b_conv_post = nn.Conv2d(conv_features, conv_features, 3, 2, 1)
        self.b_activation_post = ModulatedActivationBell(
            passive=False,
            count_modes=count_modes,
            features=conv_features,
        )
        self.b_layer_norm = nn.LayerNorm([8, 8])

        self.c_conv_pre = nn.Conv2d(conv_features, conv_features, 3, 1, 1)
        self.c_activation_pre = ModulatedActivationBell(
            passive=False,
            count_modes=count_modes,
            features=conv_features,
        )
        self.c_conv_post = nn.Conv2d(conv_features, conv_features, 3, 2, 1)
        self.c_activation_post = ModulatedActivationBell(
            passive=False,
            count_modes=count_modes,
            features=conv_features,
        )
        self.c_layer_norm = nn.LayerNorm([4, 4])

        self.d_linear = nn.Linear(conv_features * 4 * 4, 96)
        self.d_activation = ModulatedActivationBell(
            passive=False,
            count_modes=count_modes,
            features=96,
        )
        self.d_batch_norm = nn.BatchNorm1d(96)

        self.e_linear = nn.Linear(96, 100)
        self.e_activation = ModulatedActivationBell(
            passive=False,
            count_modes=count_modes,
            features=100,
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

        x = self.a_conv_pre(x)
        if self.baseline:
            x = F.tanh(x)
        else:
            x = torch.permute(x, [0, 2, 3, 1])
            x = self.a_activation_pre(x).x
            x = torch.permute(x, [0, 3, 1, 2])
        x = self.a_conv_post(x)
        if self.baseline:
            x = F.tanh(x)
        else:
            x = torch.permute(x, [0, 2, 3, 1])
            x = self.a_activation_post(x).x
            x = torch.permute(x, [0, 3, 1, 2])
        x = self.a_layer_norm(x)

        x = self.b_conv_pre(x)
        if self.baseline:
            x = F.tanh(x)
        else:
            x = torch.permute(x, [0, 2, 3, 1])
            x = self.b_activation_pre(x).x
            x = torch.permute(x, [0, 3, 1, 2])
        x = self.b_conv_post(x)
        if self.baseline:
            x = F.tanh(x)
        else:
            x = torch.permute(x, [0, 2, 3, 1])
            x = self.b_activation_post(x).x
            x = torch.permute(x, [0, 3, 1, 2])
        x = self.b_layer_norm(x)

        x = self.c_conv_pre(x)
        if self.baseline:
            x = F.tanh(x)
        else:
            x = torch.permute(x, [0, 2, 3, 1])
            x = self.c_activation_pre(x).x
            x = torch.permute(x, [0, 3, 1, 2])
        x = self.c_conv_post(x)
        if self.baseline:
            x = F.tanh(x)
        else:
            x = torch.permute(x, [0, 2, 3, 1])
            x = self.c_activation_post(x).x
            x = torch.permute(x, [0, 3, 1, 2])
        x = self.c_layer_norm(x)

        x = x.flatten(1)
        x = self.dropout(x)

        x = self.d_linear(x)
        x = self.d_activation(x).x
        x = self.d_batch_norm(x)

        x = self.e_linear(x)
        x = self.e_activation(x).x
        x = self.e_batch_norm(x)

        x = self.output_linear(x)

        x = self.log_softmax(x)

        return x
