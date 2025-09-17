from typing import Tuple

import torch
import torch.nn as nn

from dyna.lib.cpsf.fused_codebook import CPSFFusedCodebook
from dyna.lib.cpsf.functional.sv_transform import (
    spectrum_to_vector,
    vector_to_spectrum,
)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, stride=2 if downsample else 1
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1
        )
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.deconv(x))
        x = self.act(self.conv(x))
        return x


class Bottleneck(nn.Module):
    def __init__(self, ch: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.act(self.conv1(x))))


class ClassicAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Encoder (4× down)
        self.e1 = ConvBlock(3, 32, downsample=True)
        self.e2 = ConvBlock(32, 64, downsample=True)
        self.e3 = ConvBlock(64, 64, downsample=True)
        self.e4 = ConvBlock(64, 16, downsample=True)  # → [B,16,16,16]

        # Bottleneck (mix at latent size)
        self.bottleneck = Bottleneck(16)

        # Decoder (4× up)
        self.d1 = DeconvBlock(16, 64)
        self.d2 = DeconvBlock(64, 64)
        self.d3 = DeconvBlock(64, 32)
        self.d4 = DeconvBlock(32, 3)

        # Optional: initialize weights (Kaiming normal for ReLU)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def expected_shapes() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return (1, 3, 256, 256), (1, 3, 256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.e1(x)  # [B,32,128,128]
        x = self.e2(x)  # [B,64,64,64]
        x = self.e3(x)  # [B,64,32,32]
        x = self.e4(x)  # [B,16,16,16]

        # Bottleneck
        x = self.bottleneck(x)  # [B,16,16,16]

        # Decoder
        x = self.d1(x)  # [B,64,32,32]
        x = self.d2(x)  # [B,64,64,64]
        x = self.d3(x)  # [B,32,128,128]
        x = self.d4(x)  # [B,3,256,256]
        return x


# -----------------------------
# Quick sanity test
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassicAutoencoder().to(device)
    x = torch.randn(1, 3, 256, 256, device=device)
    y = model(x)
    print("input:", tuple(x.shape))
    print("output:", tuple(y.shape))
    n_params = sum(p.numel() for p in model.parameters())
    print("params:", n_params)
