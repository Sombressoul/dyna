from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyna.lib.cpsf.memcell_fused_real import CPSFMemcellFusedReal

# torch.autograd.set_detect_anomaly(True)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        downsample: bool,
        act: callable,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, stride=2 if downsample else 1
        )
        self.act = act

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DeconvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        act: callable,
    ) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1
        )
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = act

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.act(self.deconv(x))
        x = self.act(self.conv(x))
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        act: callable,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.act(self.conv1(x))))


# -----------------------------
# Model
# -----------------------------
class CPSFMemcellAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        N: int = 16,
        M: int = 32,
        S: int = 128,
        bottleneck_channels: int = 4,
        alpha: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.N = N
        self.M = M
        self.S = S
        self.alpha = alpha
        self.bottleneck_channels = bottleneck_channels

        act_enc = lambda x: F.silu(x)
        act_bn = lambda x: F.silu(x)
        act_dec = lambda x: F.silu(x)

        # Encoder
        self.e_dropout = nn.Dropout(0.1)
        self.e0_n = ConvBlock(3, self.N, downsample=True, act=act_enc)
        self.e0_s = ConvBlock(3, self.S, downsample=True, act=act_enc)
        self.e1_n = ConvBlock(self.S, self.N, downsample=True, act=act_enc)
        self.e1_s = ConvBlock(self.S, self.S, downsample=True, act=act_enc)
        self.e2_n = ConvBlock(self.S, self.N, downsample=True, act=act_enc)
        self.e2_s = ConvBlock(self.S, self.S, downsample=True, act=act_enc)
        self.e3_n = ConvBlock(self.S, self.N, downsample=True, act=act_enc)
        self.e3_s = ConvBlock(self.S, self.S, downsample=True, act=act_enc)
        self.e4_n = ConvBlock(self.S, self.N, downsample=True, act=act_enc)
        self.e4_s = ConvBlock(self.S, self.S, downsample=True, act=act_enc)

        # Bottleneck
        self.bn_n = Bottleneck(self.S, self.bottleneck_channels, act=act_bn)
        self.bn_s = Bottleneck(self.S, self.S, act=act_bn)

        # Decoder starting from S channels
        self.d4 = DeconvBlock(self.S, self.N, act=act_dec)
        self.d3 = DeconvBlock(self.S, self.N, act=act_dec)
        self.d2 = DeconvBlock(self.S, self.N, act=act_dec)
        self.d1 = DeconvBlock(self.S, self.N, act=act_dec)
        self.d0 = DeconvBlock(self.S, 3, act=act_dec)

        # Memcells
        self.cell_0 = CPSFMemcellFusedReal(
            N=self.N,
            S=self.S,
            M=self.M,
        )
        self.cell_1 = CPSFMemcellFusedReal(
            N=self.N,
            S=self.S,
            M=self.M,
        )
        self.cell_2 = CPSFMemcellFusedReal(
            N=self.N,
            S=self.S,
            M=self.M,
        )
        # self.cell_3 = CPSFMemcellFusedReal(
        #     N=self.N,
        #     S=self.S,
        #     M=self.M,
        # )
        # self.cell_4 = CPSFMemcellFusedReal(
        #     N=self.N,
        #     S=self.S,
        #     M=self.M,
        # )
        self.cell_bottleneck = CPSFMemcellFusedReal(
            N=self.bottleneck_channels,
            S=self.S,
            M=self.M,
        )

        # DEBUG LAYER
        self.debug_linear = nn.Linear(128, 2048)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder cell 0
        e0_n = self.e0_n(x)
        e0_s = self.e0_s(x)
        e0_m = self.cell_0.read_update(
            z=e0_n.permute([0, 2, 3, 1]).flatten(0, 2),
            T_star=e0_s.permute([0, 2, 3, 1]).flatten(0, 2),
            alpha=self.alpha,
        )
        B, C, H, W = e0_s.shape
        e0_m = e0_m.reshape([B, H, W, C]).permute([0, 3, 1, 2])

        # # Encoder cell 1
        # e1_n = self.e1_n(e0_m)
        # e1_s = self.e1_s(e0_m)
        # e1_m = self.cell_1.read_update(
        #     z=e1_n.permute([0, 2, 3, 1]).flatten(0, 2),
        #     T_star=e1_s.permute([0, 2, 3, 1]).flatten(0, 2),
        #     alpha=self.alpha,
        # )
        # B, C, H, W = e1_s.shape
        # e1_m = e1_m.reshape([B, H, W, C]).permute([0, 3, 1, 2])

        # # Encoder cell 2
        # e2_n = self.e1_n(e1_m)
        # e2_s = self.e1_s(e1_m)
        # e2_m = self.cell_2.read_update(
        #     z=e2_n.permute([0, 2, 3, 1]).flatten(0, 2),
        #     T_star=e2_s.permute([0, 2, 3, 1]).flatten(0, 2),
        #     alpha=self.alpha,
        # )
        # B, C, H, W = e2_s.shape
        # e2_m = e2_m.reshape([B, H, W, C]).permute([0, 3, 1, 2])

        # # Bottleneck
        # bn_n = self.bn_n(e2_m)
        # bn_s = self.bn_s(e2_m)
        # # Bottleneck write
        # self.cell_bottleneck.read_update(
        #     z=bn_n.permute([0, 2, 3, 1]).flatten(0, 2),
        #     T_star=bn_s.permute([0, 2, 3, 1]).flatten(0, 2),
        #     alpha=self.alpha,
        # )
        # # Bottleneck read
        # bn_m = self.cell_bottleneck.read(
        #     z=bn_n.permute([0, 2, 3, 1]).flatten(0, 2),
        # )
        # B, C, H, W = bn_s.shape
        # bn_m = bn_m.reshape([B, H, W, C]).permute([0, 3, 1, 2])

        # SHORTCUT
        d1_n = e0_n

        # Decoder cell 0
        d0_m = self.cell_0.read(
            z=d1_n.permute([0, 2, 3, 1]).flatten(0, 2),
        )
        B, C, H, W = d1_n.shape
        d0_m = d0_m.reshape([B, H, W, -1]).permute([0, 3, 1, 2])
        d0_n = self.d0(d0_m)

        return d0_n
