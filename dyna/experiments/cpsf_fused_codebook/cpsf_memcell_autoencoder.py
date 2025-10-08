# > python -m dyna.experiments.cpsf_fused_codebook.train_autoencoder --data-root "e:\git_AIResearch\!datasets\Img_512-512_4096_01\" --size 512 512 --epochs 1000 --batch 4 --grad_acc 1 --lr 1.0e-5 --device_target cuda --device_cache cpu --log_every 10 --out-dir ./temp --model_type memcell

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyna.lib.cpsf.memcell_fused_real import CPSFMemcellFusedReal, CPSFMemcellFusedRealGradMode

torch.autograd.set_detect_anomaly(True)


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
    ) -> None:
        super().__init__()
        self.N = 16
        self.M = 128
        self.S = 128
        self.alpha_initial = 1.0e-1
        self.alpha_trainable = False
        self.init_scale_T_hat_j = 1.0e-1
        self.tau = 5.0
        self.memcell_act = torch.nn.functional.tanh

        act_enc = lambda x: F.silu(x)
        act_bn = lambda x: F.silu(x)
        act_dec = lambda x: F.silu(x)

        # Utils
        self.dropout = nn.Dropout(0.05)

        # Encoder
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

        # Encoder norms
        self.e0_norm = nn.BatchNorm2d(self.S)
        self.e1_norm = nn.BatchNorm2d(self.S)
        self.e2_norm = nn.BatchNorm2d(self.S)
        self.e3_norm = nn.BatchNorm2d(self.S)
        self.e4_norm = nn.BatchNorm2d(self.S)

        # Bottleneck
        self.bn_n = Bottleneck(self.S, self.N, act=act_bn)
        self.bn_s = Bottleneck(self.S, self.N, act=act_bn)
        self.bn_norm = nn.BatchNorm2d(self.N)

        # Decoder starting from S channels
        self.d4 = DeconvBlock(self.S, self.N, act=act_dec)
        self.d3 = DeconvBlock(self.S, self.N, act=act_dec)
        self.d2 = DeconvBlock(self.S, self.N, act=act_dec)
        self.d1 = DeconvBlock(self.S, self.N, act=act_dec)
        self.d0 = DeconvBlock(self.S, 3, act=act_dec)

        # Decoder norms
        self.d4_norm = nn.BatchNorm2d(self.S)
        self.d3_norm = nn.BatchNorm2d(self.S)
        self.d2_norm = nn.BatchNorm2d(self.S)
        self.d1_norm = nn.BatchNorm2d(self.S)
        self.d0_norm = nn.BatchNorm2d(self.S)

        # Memcells
        memcell_grad_mode = CPSFMemcellFusedRealGradMode.MIXED
        self.cell_0 = CPSFMemcellFusedReal(
            N=self.N,
            S=self.S,
            M=self.M,
            alpha_initial=self.alpha_initial,
            alpha_trainable=self.alpha_trainable,
            grad_mode=memcell_grad_mode,
            init_scale_T_hat_j=self.init_scale_T_hat_j,
            tau=self.tau,
        )
        self.cell_1 = CPSFMemcellFusedReal(
            N=self.N,
            S=self.S,
            M=self.M,
            alpha_initial=self.alpha_initial,
            alpha_trainable=self.alpha_trainable,
            grad_mode=memcell_grad_mode,
            init_scale_T_hat_j=self.init_scale_T_hat_j,
            tau=self.tau,
        )
        self.cell_2 = CPSFMemcellFusedReal(
            N=self.N,
            S=self.S,
            M=self.M,
            alpha_initial=self.alpha_initial,
            alpha_trainable=self.alpha_trainable,
            grad_mode=memcell_grad_mode,
            init_scale_T_hat_j=self.init_scale_T_hat_j,
            tau=self.tau,
        )
        self.cell_3 = CPSFMemcellFusedReal(
            N=self.N,
            S=self.S,
            M=self.M,
            alpha_initial=self.alpha_initial,
            alpha_trainable=self.alpha_trainable,
            grad_mode=memcell_grad_mode,
            init_scale_T_hat_j=self.init_scale_T_hat_j,
            tau=self.tau,
        )
        self.cell_4 = CPSFMemcellFusedReal(
            N=self.N,
            S=self.S,
            M=self.M,
            alpha_initial=self.alpha_initial,
            alpha_trainable=self.alpha_trainable,
            grad_mode=memcell_grad_mode,
            init_scale_T_hat_j=self.init_scale_T_hat_j,
            tau=self.tau,
        )
        self.cell_bottleneck = CPSFMemcellFusedReal(
            N=self.N,
            S=self.N, # IMPORTANT: N
            M=self.M,
            alpha_initial=self.alpha_initial,
            alpha_trainable=self.alpha_trainable,
            grad_mode=memcell_grad_mode,
            init_scale_T_hat_j=self.init_scale_T_hat_j,
            tau=self.tau,
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
        x = self.cell_0.recall(
            z=e0_n.permute([0, 2, 3, 1]).flatten(0, 2),
            T_star=e0_s.permute([0, 2, 3, 1]).flatten(0, 2),
        )
        x = self.memcell_act(x)
        B, C, H, W = e0_s.shape
        x = x.reshape([B, H, W, C]).permute([0, 3, 1, 2])
        x = self.e0_norm(x)
        x = self.dropout(x)

        # # Encoder cell 1
        # e1_n = self.e1_n(x)
        # e1_s = self.e1_s(x)
        # x = self.cell_1.recall(
        #     z=e1_n.permute([0, 2, 3, 1]).flatten(0, 2),
        #     T_star=e1_s.permute([0, 2, 3, 1]).flatten(0, 2),
        # )
        # x = self.memcell_act(x)
        # B, C, H, W = e1_s.shape
        # x = x.reshape([B, H, W, C]).permute([0, 3, 1, 2])
        # x = self.e1_norm(x)
        # x = self.dropout(x)

        # # Encoder cell 2
        # e2_n = self.e2_n(x)
        # e2_s = self.e2_s(x)
        # x = self.cell_2.recall(
        #     z=e2_n.permute([0, 2, 3, 1]).flatten(0, 2),
        #     T_star=e2_s.permute([0, 2, 3, 1]).flatten(0, 2),
        # )
        # x = self.memcell_act(x)
        # B, C, H, W = e2_s.shape
        # x = x.reshape([B, H, W, C]).permute([0, 3, 1, 2])
        # x = self.e2_norm(x)
        # x = self.dropout(x)

        # # Encoder cell 3
        # e3_n = self.e3_n(x)
        # e3_s = self.e3_s(x)
        # x = self.cell_3.recall(
        #     z=e3_n.permute([0, 2, 3, 1]).flatten(0, 2),
        #     T_star=e3_s.permute([0, 2, 3, 1]).flatten(0, 2),
        # )
        # x = self.memcell_act(x)
        # B, C, H, W = e3_s.shape
        # x = x.reshape([B, H, W, C]).permute([0, 3, 1, 2])
        # x = self.e3_norm(x)
        # x = self.dropout(x)

        # # Encoder cell 4
        # e4_n = self.e4_n(x)
        # e4_s = self.e4_s(x)
        # x = self.cell_4.recall(
        #     z=e4_n.permute([0, 2, 3, 1]).flatten(0, 2),
        #     T_star=e4_s.permute([0, 2, 3, 1]).flatten(0, 2),
        # )
        # x = self.memcell_act(x)
        # B, C, H, W = e4_s.shape
        # x = x.reshape([B, H, W, C]).permute([0, 3, 1, 2])
        # x = self.e4_norm(x)
        # x = self.dropout(x)

        # Bottleneck
        bn_n = self.bn_n(x)
        bn_s = self.bn_s(x)
        x = self.cell_bottleneck.recall(
            z=bn_n.permute([0, 2, 3, 1]).flatten(0, 2),
            T_star=bn_s.permute([0, 2, 3, 1]).flatten(0, 2),
        )
        x = self.memcell_act(x)
        B, C, H, W = bn_s.shape
        x = x.reshape([B, H, W, C]).permute([0, 3, 1, 2])
        x = self.bn_norm(x)
        x = self.dropout(x)

        # # Decoder cell 4
        # d4_m = self.cell_4.recall(
        #     z=x.permute([0, 2, 3, 1]).flatten(0, 2),
        # )
        # d4_m = self.memcell_act(d4_m)
        # B, C, H, W = x.shape
        # d4_m = d4_m.reshape([B, H, W, -1]).permute([0, 3, 1, 2])
        # d4_m = self.d4_norm(d4_m)
        # d4_m = self.dropout(d4_m)
        # x = self.d4(d4_m)

        # # Decoder cell 3
        # d3_m = self.cell_3.recall(
        #     z=x.permute([0, 2, 3, 1]).flatten(0, 2),
        # )
        # d3_m = self.memcell_act(d3_m)
        # B, C, H, W = x.shape
        # d3_m = d3_m.reshape([B, H, W, -1]).permute([0, 3, 1, 2])
        # d3_m = self.d3_norm(d3_m)
        # d3_m = self.dropout(d3_m)
        # x = self.d3(d3_m)

        # # Decoder cell 2
        # d2_m = self.cell_2.recall(
        #     z=x.permute([0, 2, 3, 1]).flatten(0, 2),
        # )
        # d2_m = self.memcell_act(d2_m)
        # B, C, H, W = x.shape
        # d2_m = d2_m.reshape([B, H, W, -1]).permute([0, 3, 1, 2])
        # d2_m = self.d2_norm(d2_m)
        # d2_m = self.dropout(d2_m)
        # x = self.d2(d2_m)

        # # Decoder cell 1
        # d1_m = self.cell_1.recall(
        #     z=x.permute([0, 2, 3, 1]).flatten(0, 2),
        # )
        # d1_m = self.memcell_act(d1_m)
        # B, C, H, W = x.shape
        # d1_m = d1_m.reshape([B, H, W, -1]).permute([0, 3, 1, 2])
        # d1_m = self.d1_norm(d1_m)
        # d1_m = self.dropout(d1_m)
        # x = self.d1(d1_m)

        # Decoder cell 0
        d0_m = self.cell_0.recall(
            z=x.permute([0, 2, 3, 1]).flatten(0, 2),
        )
        d0_m = self.memcell_act(d0_m)
        B, C, H, W = x.shape
        d0_m = d0_m.reshape([B, H, W, -1]).permute([0, 3, 1, 2])
        d0_m = self.d0_norm(d0_m)
        d0_m = self.dropout(d0_m)
        x = self.d0(d0_m)

        # Flush temp memory.
        self.cell_0.clear_delta()
        self.cell_1.clear_delta()
        self.cell_2.clear_delta()
        self.cell_3.clear_delta()
        self.cell_4.clear_delta()
        self.cell_bottleneck.clear_delta()

        return x
