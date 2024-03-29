import torch
import torch.nn as nn


class CIFAR100Baseline(nn.Module):
    def __init__(
        self,
    ):
        super(CIFAR100Baseline, self).__init__()

        activation_conv = nn.ReLU()
        activation_linear = nn.ReLU()

        self.a_conv_pre = nn.Conv2d(3, 32, 3, 1, 1)
        self.a_activation_pre = activation_conv
        self.a_linear = nn.Linear(32, 8)
        self.a_activation_mid = activation_linear
        self.a_conv_post = nn.Conv2d(8, 32, 3, 2, 1)
        self.a_activation_post = activation_conv
        self.a_layer_norm = nn.LayerNorm([16, 16])

        self.b_conv_pre = nn.Conv2d(32, 32, 3, 1, 1)
        self.b_activation_pre = activation_conv
        self.b_linear = nn.Linear(32, 8)
        self.b_activation_mid = activation_linear
        self.b_conv_post = nn.Conv2d(8, 32, 3, 2, 1)
        self.b_activation_post = activation_conv
        self.b_layer_norm = nn.LayerNorm([8, 8])

        self.c_conv_pre = nn.Conv2d(32, 32, 3, 1, 1)
        self.c_activation_pre = activation_conv
        self.c_linear = nn.Linear(32, 8)
        self.c_activation_mid = activation_linear
        self.c_conv_post = nn.Conv2d(8, 32, 3, 2, 1)
        self.c_activation_post = activation_conv
        self.c_layer_norm = nn.LayerNorm([4, 4])

        self.d_linear = nn.Linear(512, 96)
        self.d_activation = activation_linear
        self.d_batch_norm = nn.BatchNorm1d(96)

        self.e_linear = nn.Linear(96, 100)
        self.e_activation = activation_linear
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
        x = self.a_activation_pre(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.a_linear(x)
        x = self.a_activation_mid(x)
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.a_conv_post(x)
        x = self.a_activation_post(x)
        x = self.a_layer_norm(x)

        x = self.b_conv_pre(x)
        x = self.b_activation_pre(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.b_linear(x)
        x = self.b_activation_mid(x)
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.b_conv_post(x)
        x = self.b_activation_post(x)
        x = self.b_layer_norm(x)

        x = self.c_conv_pre(x)
        x = self.c_activation_pre(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.c_linear(x)
        x = self.c_activation_mid(x)
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.c_conv_post(x)
        x = self.c_activation_post(x)
        x = self.c_layer_norm(x)

        x = x.flatten(1)
        x = self.dropout(x)

        x = self.d_linear(x)
        x = self.d_activation(x)
        x = self.d_batch_norm(x)

        x = self.e_linear(x)
        x = self.e_activation(x)
        x = self.e_batch_norm(x)

        x = self.output_linear(x)

        x = self.log_softmax(x)

        return x
