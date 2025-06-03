import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_warmup as warmup
import kornia
import gc

from PIL import Image

from typing import Optional, Callable
from enum import Enum

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

torch.manual_seed(10056)

import dyna


class DynamicAutoencoderModes(Enum):
    CLASSIC = 0
    DYNAMIC = 1
    CLASSIC2DYNAMIC = 2

class DynamicAutoencoder(nn.Module):
    def __init__(
        self,
        model_mode: DynamicAutoencoderModes,
        dtype_weights: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.model_mode = model_mode

        self.dtype_weights = dtype_weights
        self.output_shape = [3, 256, 256]
        self.ctx_len = 256

        convolution_channels = 16
        dynamic_conv_rank = 16
        dynamic_conv_context_use_bias = True
        cfg_conv_dn_a = dict(in_channels=convolution_channels, out_channels=convolution_channels, kernel_size=3, stride=1, padding=0, dtype=self.dtype_weights)
        cfg_conv_dn_b = dict(in_channels=convolution_channels, out_channels=convolution_channels, kernel_size=3, stride=2, padding=0, dtype=self.dtype_weights)
        cfg_conv_up_a = dict(in_channels=convolution_channels, out_channels=convolution_channels, kernel_size=3, stride=1, padding=0, dtype=self.dtype_weights)
        cfg_conv_up_b = dict(in_channels=convolution_channels, out_channels=convolution_channels, kernel_size=3, stride=2, padding=0, dtype=self.dtype_weights)
        cfg_norm = dict(num_features=convolution_channels, dtype=self.dtype_weights)

        # Classical autoencoder
        # =====> ENCODE
        # 256>128
        self.cl_encode_01_a_conv = nn.Conv2d(**cfg_conv_dn_a | dict(in_channels=3))
        self.cl_encode_01_b_conv = nn.Conv2d(**cfg_conv_dn_b)
        self.cl_encode_01_norm = nn.BatchNorm2d(**cfg_norm)

        # 128>64
        self.cl_encode_02_a_conv = nn.Conv2d(**cfg_conv_dn_a)
        self.cl_encode_02_b_conv = nn.Conv2d(**cfg_conv_dn_b)
        self.cl_encode_02_norm = nn.BatchNorm2d(**cfg_norm)

        # 64>32
        self.cl_encode_03_a_conv = nn.Conv2d(**cfg_conv_dn_a)
        self.cl_encode_03_b_conv = nn.Conv2d(**cfg_conv_dn_b)
        self.cl_encode_03_norm = nn.BatchNorm2d(**cfg_norm)

        # 32>16
        self.cl_encode_04_a_conv = nn.Conv2d(**cfg_conv_dn_a)
        self.cl_encode_04_b_conv = nn.Conv2d(**cfg_conv_dn_b)
        self.cl_encode_04_norm = nn.BatchNorm2d(**cfg_norm)

        # 16>8
        self.cl_encode_05_a_conv = nn.Conv2d(**cfg_conv_dn_a)
        self.cl_encode_05_b_conv = nn.Conv2d(**cfg_conv_dn_b)
        self.cl_encode_05_norm = nn.BatchNorm2d(**cfg_norm)

        # =====> DECODE
        # 8>16
        self.cl_decode_05_a_conv = nn.Conv2d(**cfg_conv_up_a)
        self.cl_decode_05_b_conv = nn.Conv2d(**cfg_conv_up_b)
        self.cl_decode_05_norm = nn.BatchNorm2d(**cfg_norm)

        # 16>32
        self.cl_decode_04_a_conv = nn.Conv2d(**cfg_conv_up_a)
        self.cl_decode_04_b_conv = nn.Conv2d(**cfg_conv_up_b)
        self.cl_decode_04_norm = nn.BatchNorm2d(**cfg_norm)

        # 32>64
        self.cl_decode_03_a_conv = nn.Conv2d(**cfg_conv_up_a)
        self.cl_decode_03_b_conv = nn.Conv2d(**cfg_conv_up_b)
        self.cl_decode_03_norm = nn.BatchNorm2d(**cfg_norm)

        # 64>128
        self.cl_decode_02_a_conv = nn.Conv2d(**cfg_conv_up_a)
        self.cl_decode_02_b_conv = nn.Conv2d(**cfg_conv_up_b)
        self.cl_decode_02_norm = nn.BatchNorm2d(**cfg_norm)

        # 128>256
        self.cl_decode_01_a_conv = nn.Conv2d(**cfg_conv_up_a)
        self.cl_decode_01_b_conv = nn.Conv2d(**cfg_conv_up_b | dict(out_channels=3))

        # =====> BRIDGE: Classical-to-Dynamical
        self.bridge_dropout = nn.Dropout(p=0.1)
        self.bridge_fc_1 = nn.Linear(in_features=1024, out_features=self.ctx_len * 4, bias=True)
        self.bridge_fc_1_ln = nn.LayerNorm(normalized_shape=[self.ctx_len * 4])
        self.bridge_fc_2 = nn.Linear(in_features=self.ctx_len * 4, out_features=self.ctx_len * 4, bias=True)
        self.bridge_fc_2_ln = nn.LayerNorm(normalized_shape=[self.ctx_len * 4])
        self.bridge_fc_3 = nn.Linear(in_features=self.ctx_len * 4, out_features=self.ctx_len * 4, bias=True)
        self.bridge_fc_3_ln = nn.LayerNorm(normalized_shape=[self.ctx_len * 4])
        self.bridge_fc_4 = nn.Linear(in_features=self.ctx_len * 4, out_features=self.ctx_len, bias=True)
        self.bridge_fc_4_ln = nn.LayerNorm(normalized_shape=[self.ctx_len])
        self.bridge_fc_5 = nn.Linear(in_features=self.ctx_len, out_features=self.ctx_len, bias=True)

        # Dynamic autoencoder
        cfg_dyn_conv_ds = dict(
            in_channels=convolution_channels,
            out_channels=convolution_channels,
            context_length=self.ctx_len,
            context_use_bias=dynamic_conv_context_use_bias,
            rank=dynamic_conv_rank,
            weights_repulsion_strength=0.10,
            weights_noise_strength=0.01,
            rank_noise_strength=0.01,
            similarity_penalty_strength=0.10,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            padding_dynamic=False,
            dilation=[1, 1],
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        cfg_dyn_conv_us = dict(
            in_channels=convolution_channels,
            out_channels=convolution_channels,
            context_length=self.ctx_len,
            context_use_bias=dynamic_conv_context_use_bias,
            rank=dynamic_conv_rank,
            weights_repulsion_strength=0.10,
            weights_noise_strength=0.01,
            rank_noise_strength=0.01,
            similarity_penalty_strength=0.10,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            padding_dynamic=False,
            dilation=[1, 1],
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        cfg_dyn_norm = dict(num_features=convolution_channels, dtype=self.dtype_weights)

        # =====> ENCODE
        # 256>128
        self.dyn_encode_01 = dyna.module.DynamicConv2DDelta(**cfg_dyn_conv_ds | dict(in_channels=3))
        self.dyn_encode_01_norm = nn.BatchNorm2d(**cfg_dyn_norm)

        # 128>64
        self.dyn_encode_02 = dyna.module.DynamicConv2DDelta(**cfg_dyn_conv_ds)
        self.dyn_encode_02_norm = nn.BatchNorm2d(**cfg_dyn_norm)

        # 64>32
        self.dyn_encode_03 = dyna.module.DynamicConv2DDelta(**cfg_dyn_conv_ds)
        self.dyn_encode_03_norm = nn.BatchNorm2d(**cfg_dyn_norm)

        # 32>16
        self.dyn_encode_04 = dyna.module.DynamicConv2DDelta(**cfg_dyn_conv_ds)
        self.dyn_encode_04_norm = nn.BatchNorm2d(**cfg_dyn_norm)

        # 16>8
        self.dyn_encode_05 = dyna.module.DynamicConv2DDelta(**cfg_dyn_conv_ds)
        self.dyn_encode_05_norm = nn.BatchNorm2d(**cfg_dyn_norm)

        # =====> DECODE
        # 8>16
        self.dyn_decode_05 = dyna.module.DynamicConv2DDelta(**cfg_dyn_conv_us)
        self.dyn_decode_05_norm = nn.BatchNorm2d(**cfg_dyn_norm)

        # 16>32
        self.dyn_decode_04 = dyna.module.DynamicConv2DDelta(**cfg_dyn_conv_us)
        self.dyn_decode_04_norm = nn.BatchNorm2d(**cfg_dyn_norm)

        # 32>64
        self.dyn_decode_03 = dyna.module.DynamicConv2DDelta(**cfg_dyn_conv_us)
        self.dyn_decode_03_norm = nn.BatchNorm2d(**cfg_dyn_norm)

        # 64>128
        self.dyn_decode_02 = dyna.module.DynamicConv2DDelta(**cfg_dyn_conv_us)
        self.dyn_decode_02_norm = nn.BatchNorm2d(**cfg_dyn_norm)

        # 128>256
        self.dyn_decode_01_a = dyna.module.DynamicConv2DDelta(**cfg_dyn_conv_us)
        self.dyn_decode_01_b = dyna.module.DynamicConv2DDelta(**cfg_dyn_conv_ds | dict(out_channels=3))
        self.dyn_decode_01_norm = nn.BatchNorm2d(**cfg_dyn_norm)

        pass

    def forward(self, *args, **kwargs) -> None:
        if self.model_mode == DynamicAutoencoderModes.CLASSIC:
            return self.forward_classic(*args, **kwargs)
        elif self.model_mode == DynamicAutoencoderModes.DYNAMIC:
            return self.forward_dynamic(*args, **kwargs)
        elif self.model_mode == DynamicAutoencoderModes.CLASSIC2DYNAMIC:
            return self.forward_classic2dynamic(*args, **kwargs)
        else:
            raise ValueError(f"Unknown model_mode: {self.model_mode}")
        
    def forward_classic(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        pad = lambda t, p: torch.nn.functional.pad(
            input=t,
            pad=[p, p, p, p],
            mode="replicate",
            value=None,
        )
        interpolate = lambda x, s: torch.nn.functional.interpolate(
            input=x,
            size=s,
            mode="nearest",
            align_corners=None,
            recompute_scale_factor=False,
            antialias=False,
        )

        # Encode
        x = dyna.functional.siglog(self.cl_encode_01_a_conv(pad(x, 1)))
        x = dyna.functional.siglog(self.cl_encode_01_b_conv(pad(x, 1)))
        x = self.cl_encode_01_norm(x)

        x = dyna.functional.siglog(self.cl_encode_02_a_conv(pad(x, 1)))
        x = dyna.functional.siglog(self.cl_encode_02_b_conv(pad(x, 1)))
        x = self.cl_encode_02_norm(x)

        x = dyna.functional.siglog(self.cl_encode_03_a_conv(pad(x, 1)))
        x = dyna.functional.siglog(self.cl_encode_03_b_conv(pad(x, 1)))
        x = self.cl_encode_03_norm(x)
        
        x = dyna.functional.siglog(self.cl_encode_04_a_conv(pad(x, 1)))
        x = dyna.functional.siglog(self.cl_encode_04_b_conv(pad(x, 1)))
        x = self.cl_encode_04_norm(x)

        x = dyna.functional.siglog(self.cl_encode_05_a_conv(pad(x, 1)))
        x = dyna.functional.siglog(self.cl_encode_05_b_conv(pad(x, 1)))
        x = self.cl_encode_05_norm(x)

        # Decode
        x = interpolate(x, [16, 16])
        x = dyna.functional.siglog(self.cl_decode_05_a_conv(pad(x, 1)))
        x = interpolate(x, [32, 32])
        x = dyna.functional.siglog(self.cl_decode_05_b_conv(pad(x, 1)))
        x = self.cl_decode_05_norm(x)

        x = interpolate(x, [32, 32])
        x = dyna.functional.siglog(self.cl_decode_04_a_conv(pad(x, 1)))
        x = interpolate(x, [64, 64])
        x = dyna.functional.siglog(self.cl_decode_04_b_conv(pad(x, 1)))
        x = self.cl_decode_04_norm(x)

        x = interpolate(x, [64, 64])
        x = dyna.functional.siglog(self.cl_decode_03_a_conv(pad(x, 1)))
        x = interpolate(x, [128, 128])
        x = dyna.functional.siglog(self.cl_decode_03_b_conv(pad(x, 1)))
        x = self.cl_decode_03_norm(x)

        x = interpolate(x, [128, 128])
        x = dyna.functional.siglog(self.cl_decode_02_a_conv(pad(x, 1)))
        x = interpolate(x, [256, 256])
        x = dyna.functional.siglog(self.cl_decode_02_b_conv(pad(x, 1)))
        x = self.cl_decode_02_norm(x)

        x = interpolate(x, [256, 256])
        x = dyna.functional.siglog(self.cl_decode_01_a_conv(pad(x, 1)))
        x = interpolate(x, [512, 512])
        x = torch.sigmoid(self.cl_decode_01_b_conv(pad(x, 1)))

        return x

    def forward_classic2dynamic(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        pad = lambda t, p: torch.nn.functional.pad(
            input=t,
            pad=[p, p, p, p],
            mode="replicate",
            value=None,
        )
        interpolate = lambda x, s: torch.nn.functional.interpolate(
            input=x,
            size=s,
            mode="nearest",
            align_corners=None,
            recompute_scale_factor=False,
            antialias=False,
        )

        # Classic encoder
        code = x
        code = dyna.functional.siglog(self.cl_encode_01_a_conv(pad(code, 1)))
        code = dyna.functional.siglog(self.cl_encode_01_b_conv(pad(code, 1)))
        code = self.cl_encode_01_norm(code)

        code = dyna.functional.siglog(self.cl_encode_02_a_conv(pad(code, 1)))
        code = dyna.functional.siglog(self.cl_encode_02_b_conv(pad(code, 1)))
        code = self.cl_encode_02_norm(code)

        code = dyna.functional.siglog(self.cl_encode_03_a_conv(pad(code, 1)))
        code = dyna.functional.siglog(self.cl_encode_03_b_conv(pad(code, 1)))
        code = self.cl_encode_03_norm(code)
        
        code = dyna.functional.siglog(self.cl_encode_04_a_conv(pad(code, 1)))
        code = dyna.functional.siglog(self.cl_encode_04_b_conv(pad(code, 1)))
        code = self.cl_encode_04_norm(code)

        code = dyna.functional.siglog(self.cl_encode_05_a_conv(pad(code, 1)))
        code = dyna.functional.siglog(self.cl_encode_05_b_conv(pad(code, 1)))
        code = self.cl_encode_05_norm(code)

        # Bridge for context encoding
        ctx = code
        ctx = ctx.flatten(1)
        ctx = self.bridge_dropout(ctx)
        ctx = dyna.functional.siglog(self.bridge_fc_1(ctx))
        ctx = self.bridge_fc_1_ln(ctx)
        ctx = dyna.functional.siglog(self.bridge_fc_2(ctx))
        ctx = self.bridge_fc_2_ln(ctx)
        ctx = dyna.functional.siglog(self.bridge_fc_3(ctx))
        ctx = self.bridge_fc_3_ln(ctx)
        ctx = dyna.functional.siglog(self.bridge_fc_4(ctx))
        ctx = self.bridge_fc_4_ln(ctx)
        ctx = dyna.functional.siglog(self.bridge_fc_5(ctx))

        # Dynamic decoder
        code = interpolate(code, [16, 16])
        code = dyna.functional.siglog(self.dyn_decode_05(pad(code, 1), ctx))
        code = self.dyn_decode_05_norm(code)

        code = interpolate(code, [32, 32])
        code = dyna.functional.siglog(self.dyn_decode_04(pad(code, 1), ctx))
        code = self.dyn_decode_04_norm(code)

        code = interpolate(code, [64, 64])
        code = dyna.functional.siglog(self.dyn_decode_03(pad(code, 1), ctx))
        code = self.dyn_decode_03_norm(code)

        code = interpolate(code, [128, 128])
        code = dyna.functional.siglog(self.dyn_decode_02(pad(code, 1), ctx))
        code = self.dyn_decode_02_norm(code)

        code = interpolate(code, [256, 256])
        code = dyna.functional.siglog(self.dyn_decode_01_a(pad(code, 1), ctx))
        code = self.dyn_decode_01_norm(code)
        code = interpolate(code, [512, 512])
        code = torch.sigmoid(self.dyn_decode_01_b(pad(code, 1), ctx))

        return code


    def forward_dynamic(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        pad = lambda t, p: torch.nn.functional.pad(
            input=t,
            pad=[p, p, p, p],
            mode="replicate",
            value=None,
        )
        interpolate = lambda x, s: torch.nn.functional.interpolate(
            input=x,
            size=s,
            mode="nearest",
            align_corners=None,
            recompute_scale_factor=False,
            antialias=False,
        )

        # Context encoder
        ctx = x
        ctx = dyna.functional.siglog(self.cl_encode_01_a_conv(pad(ctx, 1)))
        ctx = dyna.functional.siglog(self.cl_encode_01_b_conv(pad(ctx, 1)))
        ctx = self.cl_encode_01_norm(ctx)

        ctx = dyna.functional.siglog(self.cl_encode_02_a_conv(pad(ctx, 1)))
        ctx = dyna.functional.siglog(self.cl_encode_02_b_conv(pad(ctx, 1)))
        ctx = self.cl_encode_02_norm(ctx)

        ctx = dyna.functional.siglog(self.cl_encode_03_a_conv(pad(ctx, 1)))
        ctx = dyna.functional.siglog(self.cl_encode_03_b_conv(pad(ctx, 1)))
        ctx = self.cl_encode_03_norm(ctx)
        
        ctx = dyna.functional.siglog(self.cl_encode_04_a_conv(pad(ctx, 1)))
        ctx = dyna.functional.siglog(self.cl_encode_04_b_conv(pad(ctx, 1)))
        ctx = self.cl_encode_04_norm(ctx)

        ctx = dyna.functional.siglog(self.cl_encode_05_a_conv(pad(ctx, 1)))
        ctx = dyna.functional.siglog(self.cl_encode_05_b_conv(pad(ctx, 1)))
        ctx = self.cl_encode_05_norm(ctx)

        # Bridge
        ctx = ctx.flatten(1)
        ctx = self.bridge_dropout(ctx)
        ctx = dyna.functional.siglog(self.bridge_fc_1(ctx))
        ctx = self.bridge_fc_1_ln(ctx)
        ctx = dyna.functional.siglog(self.bridge_fc_2(ctx))
        ctx = self.bridge_fc_2_ln(ctx)
        ctx = dyna.functional.siglog(self.bridge_fc_3(ctx))
        ctx = self.bridge_fc_3_ln(ctx)
        ctx = dyna.functional.siglog(self.bridge_fc_4(ctx))
        ctx = self.bridge_fc_4_ln(ctx)
        ctx = dyna.functional.siglog(self.bridge_fc_5(ctx))

        # Dynamic encoder
        code = x
        code = dyna.functional.siglog(self.dyn_encode_01(pad(code, 1), ctx))
        code = self.dyn_encode_01_norm(code)

        code = dyna.functional.siglog(self.dyn_encode_02(pad(code, 1), ctx))
        code = self.dyn_encode_02_norm(code)

        code = dyna.functional.siglog(self.dyn_encode_03(pad(code, 1), ctx))
        code = self.dyn_encode_03_norm(code)

        code = dyna.functional.siglog(self.dyn_encode_04(pad(code, 1), ctx))
        code = self.dyn_encode_04_norm(code)

        code = dyna.functional.siglog(self.dyn_encode_05(pad(code, 1), ctx))
        code = self.dyn_encode_05_norm(code)

        # Dynamic decoder
        code = interpolate(code, [16, 16])
        code = dyna.functional.siglog(self.dyn_decode_05(pad(code, 1), ctx))
        code = self.dyn_decode_05_norm(code)

        code = interpolate(code, [32, 32])
        code = dyna.functional.siglog(self.dyn_decode_04(pad(code, 1), ctx))
        code = self.dyn_decode_04_norm(code)

        code = interpolate(code, [64, 64])
        code = dyna.functional.siglog(self.dyn_decode_03(pad(code, 1), ctx))
        code = self.dyn_decode_03_norm(code)

        code = interpolate(code, [128, 128])
        code = dyna.functional.siglog(self.dyn_decode_02(pad(code, 1), ctx))
        code = self.dyn_decode_02_norm(code)

        code = interpolate(code, [256, 256])
        code = dyna.functional.siglog(self.dyn_decode_01_a(pad(code, 1), ctx))
        code = self.dyn_decode_01_norm(code)
        code = interpolate(code, [512, 512])
        code = torch.sigmoid(self.dyn_decode_01_b(pad(code, 1), ctx))

        return code


def generate_data_from_images(
    shape: list[int],
    images_path_src: str,
    images_sample_count: int,
    starting_from: int = 0,
) -> torch.Tensor:
    data = torch.empty([images_sample_count, 3, *shape], dtype=torch.uint8)
    dir_contents = os.listdir(images_path_src)

    assert len(dir_contents) >= images_sample_count + starting_from, " ".join(
        [
            f"Not enough images in '{images_path_src}'.",
            f"Got: {len(dir_contents)}.",
            f"Need: {images_sample_count + starting_from}.",
        ]
    )

    for i in range(images_sample_count):
        image_name = dir_contents[starting_from + i]
        image_path = os.path.join(images_path_src, image_name)

        try:
            image = Image.open(image_path)
        except ValueError:
            print(f"Failed to load image '{image_path}'.")
            exit()

        image = image.resize([shape[1], shape[0]], Image.LANCZOS)
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0) * 255.0
        image = image.to(dtype=torch.uint8)
        data[i] = image

    return data


def generate_images_from_data(
    data: torch.Tensor,
    images_path_dst: str,
    prefix: str,
) -> None:
    data = data.to(dtype=torch.float16)
    for i in range(data.shape[0]):
        image = data[i].squeeze(0)
        # image = image.clamp(0.0, 1.0)
        image = (image - image.min()) / (image.max() - image.min())
        image = transforms.ToPILImage()(image)
        image_name = f"{prefix}_mat_{i}.png"
        image_path = os.path.join(images_path_dst, image_name)
        image.save(image_path)
    pass


def data_rgb_to_lab(
    data: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        lab = kornia.color.rgb_to_lab(data)
        lab[:, 0, ...].div_(100.0)
        lab[:, 1:, ...].add_(128.0).div_(255.0)
    lab.to(data.device)
    return lab


def data_lab_to_rgb(
    data: torch.Tensor,
) -> torch.Tensor:
    rgb = data
    with torch.no_grad():
        rgb[:, 0, ...].mul_(100.0)
        rgb[:, 1:, ...].mul_(255.0).sub_(128.0)
        rgb = kornia.color.lab_to_rgb(rgb)
    rgb.to(data.device)
    return rgb


def compare_image_sample(
    model: DynamicAutoencoder,
    images_lab: torch.Tensor,
    path: str,
    prefix=f"output",
) -> None:
    images_src = data_lab_to_rgb(images_lab.clone())
    images_dst = data_lab_to_rgb(torch.clamp(model(images_lab), 0.0, 1.0))

    for i in range(images_lab.shape[0]):
        a = images_src[i].squeeze(0)
        a = transforms.ToPILImage()(a)
        b = images_dst[i].squeeze(0)
        b = transforms.ToPILImage()(b)

        c = Image.new("RGB", [a.size[0] + b.size[0], max(a.size[1], b.size[1])])
        c.paste(a, [0,0])
        c.paste(b, [a.size[0],0])

        image_name = f"{prefix}_{i}.png"
        image_path = os.path.join(path, image_name)
        c.save(image_path)


def train(
    data: torch.Tensor,
    total_steps: int,
    batch_size: int,
    grad_accumulation_steps: int,
    model: DynamicAutoencoder,
    optimizer: torch.optim.Optimizer,
    clip_grad_value: float,
    clip_grad_norm: float,
    gradient_global_norm: bool,
    log_nth_update_step: int,
    images_path_dst: str = None,
    save_nth_iteration: int = 100,
    savers: list = [],
    to_save: list[bool] = [],
    show_grads: bool = False,
    show_grads_detailed: bool = False,
    show_max_weights: bool = False,
    show_max_weights_detailed: bool = False,
    warmup_scheduler: Optional[Callable] = None,
    warmup_steps: Optional[int] = None,
    lr_scheduler: Optional[Callable] = None,
) -> None:
    with torch.no_grad():
        data_lab = data.cpu().to(dtype=model.dtype_weights)
        data_lab = data_lab / 255.0
        data_lab = data_rgb_to_lab(data_lab)
        data_lab = data_lab.to(dtype=model.dtype_weights, device=data.device)
        data = data.cpu().detach()
        gc.collect()

    accumulation_step = 0
    epoch_ids = torch.randperm(data_lab.shape[0])
    epoch_idx = 0
    loss_logging_accumulator = []
    weights_update_step_idx = 0

    # initially_decoded_samples = model(data_lab[0:4:1, ::])
    # generate_images_from_data(
    #     data=data_lab_to_rgb(initially_decoded_samples),
    #     images_path_dst=images_path_dst,
    #     prefix="initial_state",
    # )
    compare_image_sample(model, data_lab[0:8], images_path_dst)

    print(f"Starting training. Epoch #{epoch_idx}")
    for step_idx in range(total_steps):
        if len(epoch_ids) < batch_size:
            epoch_idx = epoch_idx + 1
            print(f"\n# ==============> New epoch: #{epoch_idx}")
            epoch_ids = torch.randperm(data_lab.shape[0])

        batch_ids = epoch_ids[0:batch_size]
        epoch_ids = epoch_ids[batch_size:]

        sample = data_lab[batch_ids]
        sample = sample.to(dtype=model.dtype_weights)

        accumulation_step = accumulation_step + 1
        # decoded = model(sample)
        decoded = model(sample)

        # loss_current_step = log_proportional_error(decoded, sample)
        # loss_current_step = F.mse_loss(decoded, sample)
        loss_current_step = F.kl_div(
            input=torch.softmax(decoded.flatten(-2), dim=-1).log(),
            target=torch.softmax(sample.flatten(-2), dim=-1).log(),
            reduction="sum",
            log_target=True,
        ).div(sample.shape[0])

        loss_logging_accumulator.append(loss_current_step.detach().item())
        loss_total = loss_current_step / grad_accumulation_steps
        loss_total.backward()

        if accumulation_step == grad_accumulation_steps:
            weights_update_step_idx = weights_update_step_idx + 1
            accumulation_step = 0

            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    if epoch_idx > warmup_steps:
                        lr_scheduler.step(epoch_idx - warmup_steps)
            
            if clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value, foreach=True)
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm, norm_type=2.0, error_if_nonfinite=False, foreach=True)
            
            if gradient_global_norm:
                normalize_grad(model)

            optimizer.step()

        if (weights_update_step_idx > 0 and (weights_update_step_idx) % log_nth_update_step == 0):
            weights_update_step_idx = 0
            lrs = []
            for g in optimizer.param_groups:
                if g['lr'] not in lrs:
                    lrs.append(g['lr'])
            lrs = [f"{v:.10f}" for v in lrs]
            print(
                "\n# ==============> "
                + "\n".join(
                    [
                        f"Iteration #{step_idx+1}:",
                        "\n".join([f"LR #{idx}: {v}" for idx,v in enumerate(lrs)]),
                        f"Loss current: {loss_current_step.item():.20f}",
                        f"Loss mean: {(sum(loss_logging_accumulator)/len(loss_logging_accumulator)):.20f}",
                    ]
                )
            )

            if show_grads:
                absolute_max = 0.0
                absolute_name = ""
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        g = param.grad.data
                        max_g = g.abs().max()
                        if absolute_max < max_g:
                            absolute_max = max_g
                            absolute_name = name
                        if show_grads_detailed:
                            print(f"Grads for '{name}' min/max/mean/std: {g.min().item()}/{g.max().item()}/{g.mean().item()}/{g.std().item()}")
                print(f"Absolute max grad: '{absolute_name}' -> {absolute_max}")
            
            if show_max_weights:
                absolute_max = 0.0
                absolute_name = ""
                for name, param in model.named_parameters():
                    max_w = param.data.abs().max()
                    if absolute_max < max_w:
                        absolute_max = max_w
                        absolute_name = name
                    if show_max_weights_detailed:
                        print(f"Max weight for '{name}': {max_w}")
                print(f"Absolute max weight: '{absolute_name}' -> {absolute_max}")

            # inspect_grad(model.block_input_x_conv       , "block_input_x_conv   ")
            # inspect_grad(model.encode_block_01          , "encode_block_01      ")
            # inspect_grad(model.encode_block_02          , "encode_block_02      ")
            # inspect_grad(model.encode_block_03          , "encode_block_03      ")
            # inspect_grad(model.encode_block_04          , "encode_block_04      ")
            # inspect_grad(model.encode_block_05          , "encode_block_05      ")
            # inspect_grad(model.bottleneck_block_encode  , "encode_block_05      ")
            # inspect_grad(model.bottleneck_block_decode  , "encode_block_05      ")
            # inspect_grad(model.decode_block_05          , "decode_block_05      ")
            # inspect_grad(model.decode_block_04          , "decode_block_04      ")
            # inspect_grad(model.decode_block_03          , "decode_block_03      ")
            # inspect_grad(model.decode_block_02          , "decode_block_02      ")
            # inspect_grad(model.decode_block_01          , "decode_block_01      ")
            # inspect_grad(model.block_out_conv           , "block_out_conv       ")
            # exit()
            loss_logging_accumulator = []

            decoded = torch.clamp(decoded, 0.0, 1.0)
            compare_image_sample(
                model,
                sample[-1].unsqueeze(0),
                path=images_path_dst,
                prefix=f"output_e{epoch_idx:0>4d}_i{(step_idx+1):0>7d}",
            )

        if accumulation_step == grad_accumulation_steps:
            optimizer.zero_grad()

        if (step_idx + 1) % save_nth_iteration == 0:
            for j, saver in enumerate(savers):
                if to_save[j]:
                    saver(step_idx + 1)
            pass

        del sample
        del decoded

        try:
            del loss_base_decoded
        except NameError:
            loss_base_decoded = None
        
        try:
            del loss_base_targets
        except NameError:
            loss_base_targets = None
        
    print("\n# --------------------------------------------------- #\n")

    generate_images_from_data(
        data=data_lab_to_rgb(decoded[-1].unsqueeze(0).clamp(0.0, 1.0)),
        images_path_dst=images_path_dst,
        prefix=f"output_final_i{step_idx+1}",
    )

    pass


def model_info(
    model: nn.Module,
) -> None:
    info = [
        f"{n: <64s} -> "
        + ", ".join(
            [
                f"{p.abs().min().item()=:.6f}",
                f"{p.abs().max().item()=:.6f}",
                f"{p.abs().mean().item()=:.6f}",
                f"{p.min().item()=:.6f}",
                f"{p.max().item()=:.6f}",
                f"{p.mean().item()=:.6f}",
                f"{p.std().item()=:.6f}",
            ]
        )
        for n, p in model.named_parameters()
    ]
    print("\n".join(info))
    pass


def optim_change_momentum(
    optim: torch.optim.Optimizer,
    momentum: float,
) -> None:
    for group in optim.param_groups:
        group["momentum"] = momentum
        for p in group["params"]:
            state = optim.state[p]
            state["x0"] = torch.zeros_like(state["s"]).detach()

    print(f"optim_change_momentum: {momentum}")
    return None


def model_cast_to_dtype(model, dtype) -> None:
    print("Begin parameters casting.")
    n = 0
    for name, param in model.named_parameters():
        print(f"Casting {param.dtype}->{dtype} parameter '{name}'")
        param = param.to(dtype)
        n = n + 1
    print(f"Parameters casting completed. Named parameters casted: {n}")
    pass


def model_perturb_small_weights(
    model: DynamicAutoencoder,
    a: float = 0.0001,
    b: float = 0.0010,
) -> None:
    print(f"model_perturb_small_weights: {a=}, {b=}")
    for name, param in model.named_parameters():
        if "data_cache" not in name:
            delta = torch.nn.init.uniform_(
                torch.empty_like(param.data),
                a=a,
                b=b,
            )
            delta = delta * torch.randn_like(param.data).sign()
            cnd = torch.where(param.data.abs() < a)
            param.data[cnd] = (param.data + delta)[cnd]
            cnd = torch.where(param.data.abs() < a)
            param.data[cnd] = delta[cnd]
            print(f"model_perturb_small_weights: {name}")
    pass


def model_perturb_weights(
    model: DynamicAutoencoder,
    rate: float = 0.0001,
    exclude: list[str] = []
) -> None:
    print(f"model_perturb_weights: {rate=}")
    for name, param in model.named_parameters():
        if any([s in name for s in exclude]):
            continue

        delta = torch.nn.init.normal_(
            torch.empty_like(param.data),
            mean=param.mean().item(),
            std=param.data.std() * rate,
        )
        param.data = param.data + delta
        print(f"model_perturb_weights: {name}")
    pass


def model_remove_nulls(
    model: DynamicAutoencoder,
    eps: 1.0e-3,
) -> None:
    print(f"Removing null values from model parameters.")
    for name, param in model.named_parameters():
        if "data_cache" not in name:
            noise = torch.nn.init.normal_(
                torch.empty_like(param.data),
                mean=0.0,
                std=eps,
            )
            noise = (noise.abs() + eps) * torch.sign(noise)
            condition = torch.where(param.data == 0.0)
            param.data[condition] = noise[condition]
            print(f"Nulls removed from '{name}'")
    pass


def normalize_grad(model: DynamicAutoencoder) -> None:
    eps: float = 1.0e-6

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            norm_dims = tuple(range(1, grad.dim()))
            grad_norm = grad.norm(p=2, dim=norm_dims, keepdim=True)
            grad_norm = grad_norm + eps
            scaling_factor = grad[0].numel() ** 0.5
            grad = (grad / grad_norm) * scaling_factor
            param.grad = grad

    pass

def inspect_grad(module: torch.nn.Module, identifier: str) -> None:
    for name, param in module.named_parameters():
        if param.grad is not None:
            g = param.grad
            print(f"{identifier} -> grads for '{name}' min/max/mean/std: {g.min().item()}/{g.max().item()}/{g.mean().item()}/{g.std().item()}")


def model_freeze_all(
    model: DynamicAutoencoder,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = False
    print(f"model_freeze_all")
    pass


def model_unfreeze_all(
    model: DynamicAutoencoder,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = True
    print(f"model_unfreeze_all")
    pass


def model_freeze_block(
    model: DynamicAutoencoder,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    block_name: str = "no_block_passed",
) -> None:
    for name, param in model.named_parameters():
        if block_name in name:
            param.requires_grad = False
            print(f"model_freeze_block: {name}")
    pass


def model_unfreeze_block(
    model: DynamicAutoencoder,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    block_name: str = "no_block_passed",
) -> None:
    for name, param in model.named_parameters():
        if block_name in name:
            param.requires_grad = True
            print(f"model_unfreeze_block: {name}")
    pass


if __name__ == "__main__":
    train_mode = True

    load_model = False
    load_optim = False
    onload_model_fn = [
        model_unfreeze_all,
        # model_freeze_all,
        # lambda m, d, t: model_unfreeze_block(m, d, t, "cl_encode"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "cl_decode"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "dyn_encode"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "dyn_decode"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "bridge"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "context_transform"),
        # lambda m, d, t: model_freeze_block(m, d, t, "context_transform"),
        # lambda m, d, t: model_cast_to_dtype(m, torch.bfloat16),
        # lambda m, d, t: model_perturb_weights(m, 5.0e-2, exclude=["bridge", "context_transform"]),
        # lambda m, d, t: model_perturb_small_weights(m, 1.0e-4, 1.0e-3),
        # lambda m, d, t: model_remove_nulls(m, 1.0e-3),
    ]
    onload_optim_fn = [
        # lambda o: optim_change_momentum(o, 0.9),
    ]

    path_prefix_load = "f:\\git_AIResearch\\dyna\\data\\models"
    path_prefix_save = "f:\\git_AIResearch\\dyna\\data\\models"
    load_path_model = f"{path_prefix_load}\\"
    load_path_optim = f"{path_prefix_load}\\"
    save_path_model = f"{path_prefix_save}\\00.model"
    save_path_optim = f"{path_prefix_save}\\00.optim"
    save_model = True
    save_optim = True
    save_initial_model = False
    save_initial_optim = False
    save_nth_iteration = 8192

    # various for optimizers
    optim_update_lr = False
    optim_target_lr = 1.0e-3
    optim_update_wd = False
    optim_target_wd = 0.1
    warmup_active = True
    warmup_steps = 2048
    clip_grad_value = None
    clip_grad_norm = 10.0
    gradient_global_norm = False
    show_grads = False
    show_grads_detailed = False
    show_max_weights = False
    show_max_weights_detailed = False

    model_mode = DynamicAutoencoderModes.DYNAMIC
    batch_size = 32
    nelements = batch_size * 8 # 256
    total_steps = 10**6
    grad_accumulation_steps = 2 # nelements // batch_size
    log_nth_update_step = (nelements // batch_size) // grad_accumulation_steps

    images_sample_count = nelements
    starting_from = 0
    images_path_src = "f:\\Datasets\\Images_512x512\\dataset_01"
    images_path_dst = "f:\\git_AIResearch\\dyna\\data\\img_dst_1"
    output_shape = [256, 256]
    dtype_weights = torch.float32
    device = torch.device("cuda")

    model = DynamicAutoencoder(
        model_mode=model_mode,
        dtype_weights=dtype_weights,
    ).to(device=device, dtype=dtype_weights)

    if load_model:
        inconsistent_keys = model.load_state_dict(
            torch.load(load_path_model),
            strict=False,
        )
        print(f"Model loaded from {load_path_model}")

    if onload_model_fn is not None:
        if callable(onload_model_fn):
            onload_model_fn(model, device, dtype_weights)
        if type(onload_model_fn) is list:
            for fn in onload_model_fn:
                if callable(fn):
                    fn(model, device, dtype_weights)

    print("Model specs:")
    for name, param in model.named_parameters():
        param.data = param.data.to(dtype=dtype_weights)
        print(f"{name}: {param.shape} -> {param.data.dtype}")

    if train_mode:
        model = model.train()
        print("Training mode")
    else:
        model = model.eval()
        print("Evaluation mode")

    print("Using AdamW")
    params_encode_classic = []
    params_decode_classic = []
    params_bridge = []
    params_encode_transform = []
    params_decode_transform = []
    params_encode_dynamic = []
    params_decode_dynamic = []
    params_other = []

    for name, param in model.named_parameters():
        if "cl_encode" in name:
            params_encode_classic.append(param)
        elif "cl_decode" in name:
            params_decode_classic.append(param)
        elif "bridge" in name:
            params_bridge.append(param)
        elif "dyn_encode" in name and "context_transform" in name:
            params_encode_transform.append(param)
        elif "dyn_decode" in name and "context_transform" in name:
            params_decode_transform.append(param)
        elif "dyn_encode" in name and "context_transform" not in name:
            params_encode_dynamic.append(param)
        elif "dyn_decode" in name and "context_transform" not in name:
            params_decode_dynamic.append(param)
        else:
            print(f"Unknown/Unaccounted parameter: {name}")
            params_other.append(param)

    if model_mode == DynamicAutoencoderModes.CLASSIC:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1.0e-4,
            weight_decay=1.0e-2,
            amsgrad=True,
            eps=1.0e-8,
        )
    elif model_mode == DynamicAutoencoderModes.CLASSIC2DYNAMIC:
        optimizer = torch.optim.AdamW(
            [
                # STAGE 1
                {'params': params_encode_classic, 'lr': 1.0e-5, 'weight_decay': 1.0e-2},
                {'params': params_bridge, 'lr': 1.0e-4, 'weight_decay': 1.0e-2},
                {'params': params_decode_transform, 'lr': 1.0e-4, 'weight_decay': 1.0e-2},
                {'params': params_decode_dynamic, 'lr': 1.0e-3, 'weight_decay': 1.0e-2},
            ],
        )
    elif model_mode == DynamicAutoencoderModes.DYNAMIC:
        optimizer = torch.optim.AdamW(
            [
                {'params': params_encode_classic, 'lr': 1.0e-6, 'weight_decay': 1.0e-3},
                {'params': params_bridge, 'lr': 1.0e-6, 'weight_decay': 1.0e-3},
                {'params': params_encode_transform, 'lr': 1.0e-5, 'weight_decay': 1.0e-2},
                {'params': params_encode_dynamic, 'lr': 1.0e-4, 'weight_decay': 1.0e-2},
                {'params': params_decode_dynamic, 'lr': 1.0e-4, 'weight_decay': 1.0e-3},
            ],
        )
    else:
        raise ValueError(f"Unsupported model_mode: {model_mode}")

    if load_optim:
        optimizer.load_state_dict(torch.load(load_path_optim))
        print(f"Optimizer loaded from {load_path_optim}")

    # onload_optim_fn
    if onload_optim_fn is not None:
        if callable(onload_optim_fn):
            onload_optim_fn(optimizer)
        if type(onload_optim_fn) is list:
            for fn in onload_optim_fn:
                if callable(fn):
                    fn(optimizer)

    with torch.no_grad():
        data = generate_data_from_images(
            shape=output_shape,
            images_path_src=images_path_src,
            images_sample_count=images_sample_count,
            starting_from=starting_from,
        ).to(device)

    savers = [
        lambda p: torch.save(model.state_dict(), f"{save_path_model}.{p}.pth"),
        lambda p: torch.save(optimizer.state_dict(), f"{save_path_optim}.{p}.pth"),
    ]

    # Presave initial states.
    if save_initial_model:
        torch.save(model.state_dict(), f"{save_path_model}.__INITIAL__.pth")
    
    if save_initial_optim:
        torch.save(optimizer.state_dict(), f"{save_path_optim}.__INITIAL__.pth")

    if optim_update_lr:
        old_param = None
        for g in optimizer.param_groups:
            old_param = g["lr"] if old_param is None else old_param
            g["lr"] = optim_target_lr
        print(f"Updated learning rate from {old_param} to {optim_target_lr}")

    if optim_update_wd:
        old_param = None
        for g in optimizer.param_groups:
            old_param = g["weight_decay"] if old_param is None else old_param
            g["weight_decay"] = optim_target_wd
        print(f"Updated weight decay from {old_param} to {optim_target_wd}")

    if warmup_active:
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer=optimizer,
            factor=1.00,
            total_iters=1,
        )
        warmup_scheduler = warmup.LinearWarmup(
            optimizer=optimizer,
            warmup_period=warmup_steps,
        )
    else:
        lr_scheduler = None
        warmup_steps = None
        warmup_scheduler = None

    start_training = lambda: train(
        data=data,
        total_steps=total_steps,
        batch_size=batch_size,
        grad_accumulation_steps=grad_accumulation_steps,
        model=model,
        optimizer=optimizer,
        clip_grad_value=clip_grad_value,
        clip_grad_norm=clip_grad_norm,
        gradient_global_norm=gradient_global_norm,
        log_nth_update_step=log_nth_update_step,
        images_path_dst=images_path_dst,
        save_nth_iteration=save_nth_iteration,
        savers=savers,
        to_save=[save_model, save_optim],
        show_grads=show_grads,
        show_max_weights=show_max_weights,
        show_max_weights_detailed=show_max_weights_detailed,
        warmup_scheduler=warmup_scheduler,
        warmup_steps=warmup_steps,
        lr_scheduler=lr_scheduler,
    )

    if train_mode:
        start_training()
