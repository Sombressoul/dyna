from typing import Tuple

import torch
import torch.nn as nn

from dyna.lib.cpsf.fused_codebook import CPSFFusedCodebook


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


@torch.no_grad()
def _complex64(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_complex() and x.dtype == torch.complex64 else x.to(torch.complex64)


def spectrum_to_vector(spectrum: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = torch.fft.ifft(_complex64(spectrum), dim=dim)
    return x.real


def vector_to_spectrum(vector: torch.Tensor, n: int, dim: int = -1) -> torch.Tensor:
    x = torch.fft.fft(vector, n, dim=dim)
    return x


# -----------------------------
# Model
# -----------------------------
class CPSFSpectralAutoencoder(nn.Module):
    """
    CPSF-powered spectral autoencoder with clean FFT helpers.

    Args:
        N: number of CPSF spectral modes for (z, vec_d)
        S: spectrum length returned by codebook (and vector length to the decoder)
        quad_nodes, n_chunk, m_chunk, eps_total, c_dtype: forwarded to CPSFFusedCodebook
    """

    def __init__(
        self,
        *,
        N: int = 16,
        S: int = 256,
        quad_nodes: int = 6,
        n_chunk: int = 1024,
        m_chunk: int = 1024,
        eps_total: float = 1e-3,
        c_dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        self.N = N
        self.S = S

        # Encoder → latent [B,16,16,16]
        self.e1 = ConvBlock(3, 32, downsample=True)
        self.e2 = ConvBlock(32, 64, downsample=True)
        self.e3 = ConvBlock(64, 64, downsample=True)
        self.e4 = ConvBlock(64, 16, downsample=True)
        self.bottleneck = Bottleneck(16)

        # Head to complex (z, vec_d): 4N channels (z_re, z_im, d_re, d_im)
        self.head = nn.Conv2d(16, 4 * N, kernel_size=1)

        # CPSF codebook
        self.codebook = CPSFFusedCodebook(
            N=N,
            M=256,
            S=S,
            quad_nodes=quad_nodes,
            n_chunk=n_chunk,
            m_chunk=m_chunk,
            eps_total=eps_total,
            c_dtype=c_dtype,
        )

        # Decoder starting from S channels
        self.d1 = DeconvBlock(S, 128)
        self.d2 = DeconvBlock(128, 64)
        self.d3 = DeconvBlock(64, 32)
        self.d4 = DeconvBlock(32, 3)

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
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        x = self.bottleneck(x)

        # Bottleneck
        x = self.head(x)
        x_dtype = x.dtype
        B, C, W, H = x.shape
        x = x.permute([0, 2, 3, 1]).flatten(0, 2)
        z = vector_to_spectrum(x[..., :32], self.N)
        vec_d = vector_to_spectrum(x[..., 32:], self.N)
        codes = self.codebook(z, vec_d)
        vec = spectrum_to_vector(codes, dim=-1)
        vec = vec.reshape([B, W, H, self.S]).permute([0, 3, 1, 2]).to(x_dtype)

        # Decoder
        y = self.d1(vec)
        y = self.d2(y)
        y = self.d3(y)
        y = self.d4(y)
        return y


# -----------------------------
# Quick sanity check
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPSFSpectralAutoencoder(N=16, S=256, pos_chunk=128).to(device)
    x = torch.randn(2, 3, 256, 256, device=device)
    y = model(x)
    print("input:", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("params:", sum(p.numel() for p in model.parameters()))
