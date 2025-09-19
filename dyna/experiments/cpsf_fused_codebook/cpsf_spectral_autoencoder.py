from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyna.lib.cpsf.fused_codebook import CPSFFusedCodebook
from dyna.functional.backward_gradient_normalization import backward_gradient_normalization


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
class CPSFSpectralAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        N: int = 16,
        M: int = 2048,
        S: int = 2048,
        bottleneck_ch: int = 4,
        quad_nodes: int = 6,
        n_chunk: int = 2048,
        m_chunk: int = 2048,
        eps_total: float = 1e-3,
        c_dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        self.N = N
        self.M = M
        self.S = S
        self.c_dtype = c_dtype
        self.navigation_size = 4 * N

        base_act = lambda x: F.leaky_relu(x, 0.3)

        # Encoder
        self.e_dropout = nn.Dropout(0.1)
        self.e0 = ConvBlock(3, 16, downsample=True, act=base_act)
        self.e1 = ConvBlock(16, 32, downsample=True, act=base_act)
        self.e2 = ConvBlock(32, 64, downsample=True, act=base_act)
        self.e3 = ConvBlock(64, 128, downsample=True, act=base_act)
        self.e4 = ConvBlock(128, 256, downsample=True, act=base_act)
        self.bottleneck = Bottleneck(256, bottleneck_ch, act=base_act)

        # Head
        self.head = nn.Conv2d(
            bottleneck_ch,
            self.navigation_size,
            kernel_size=3,
            padding=1,
            stride=1,
            padding_mode="reflect",
        )

        # CPSF codebook
        self.codebook = CPSFFusedCodebook(
            N=self.N,
            M=self.M,
            S=self.S,
            quad_nodes=quad_nodes,
            n_chunk=n_chunk,
            m_chunk=m_chunk,
            eps_total=eps_total,
            autonorm_vec_d=True,
            autonorm_vec_d_j=True,
            overlap_rate=0.01,
            anisotropy=1.5,
            init_S_scale=1.0e-3,
            phase_scale=1.0,
            c_dtype=c_dtype,
        )
        self.codebook_norm = nn.LayerNorm(
            normalized_shape=[self.S],
            elementwise_affine=True,
        )
        self.codebook_dropout = nn.Dropout(0.2)

        # Decoder starting from S channels
        self.d_dropout = nn.Dropout(0.1)
        self.d0 = DeconvBlock(S, 512, act=base_act)
        self.d1 = DeconvBlock(512, 256, act=base_act)
        self.d2 = DeconvBlock(256, 128, act=base_act)
        self.d3 = DeconvBlock(128, 64, act=base_act)
        self.d4 = DeconvBlock(64, 3, act=base_act)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def expected_shapes() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return (1, 3, 256, 256), (1, 3, 256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.e0(x)
        x = self.e_dropout(x)
        x = backward_gradient_normalization(x)
        x = self.e1(x)
        x = self.e_dropout(x)
        x = backward_gradient_normalization(x)
        x = self.e2(x)
        x = self.e_dropout(x)
        x = backward_gradient_normalization(x)
        x = self.e3(x)
        x = self.e_dropout(x)
        x = backward_gradient_normalization(x)
        x = self.e4(x)
        x = self.e_dropout(x)
        x = backward_gradient_normalization(x)
        x = self.bottleneck(x)
        x = backward_gradient_normalization(x)

        # Bottleneck
        x = self.head(x)
        x = backward_gradient_normalization(x)
        x_dtype = x.dtype
        B, C, W, H = x.shape
        x = x.permute([0, 2, 3, 1]).flatten(0, 2)

        # Retrieve
        x = self.codebook(x).abs()
        x = backward_gradient_normalization(x)
        x = self.codebook_norm(x)
        x = self.codebook_dropout(x)
        x = x.reshape([B, W, H, self.S]).permute([0, 3, 1, 2]).to(x_dtype)

        # Decoder
        x = self.d0(x)
        x = self.d_dropout(x)
        x = backward_gradient_normalization(x)
        x = self.d1(x)
        x = self.d_dropout(x)
        x = backward_gradient_normalization(x)
        x = self.d2(x)
        x = self.d_dropout(x)
        x = backward_gradient_normalization(x)
        x = self.d3(x)
        x = self.d_dropout(x)
        x = backward_gradient_normalization(x)
        x = self.d4(x)

        return x


# -----------------------------
# Quick sanity check
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPSFSpectralAutoencoder(N=16, S=256).to(device)
    x = torch.randn(2, 3, 256, 256, device=device)
    y = model(x)
    print("input:", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("params:", sum(p.numel() for p in model.parameters()))
