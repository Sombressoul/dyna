import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_warmup as warmup
import kornia
import gc
import bitsandbytes as bnb

from PIL import Image
from madgrad import MADGRAD

from typing import Optional, Callable

script_dir = os.path.dirname(os.path.abspath(__file__))
evals_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(evals_dir)
sys.path.append(project_dir)

torch.manual_seed(10056)

from dyna.functional import mean_proportional_error
from dyna.module import DynamicConv2DAlpha
from dyna.block import Coder2DDynamicAlpha


class DecoderOnlyModel(nn.Module):
    def __init__(
        self,
        data_cache_ctx_len: int = None,
        data_cache_ctx_shape: list[int] = None,
        data_cache_ctx_bound: list[float] = [-0.001, +0.001],
        dtype_weights: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.data_cache_ctx_len = data_cache_ctx_len
        self.data_cache_ctx_shape = data_cache_ctx_shape
        self.data_cache_ctx_bound = data_cache_ctx_bound
        self.eps = 1.0e-3

        self.context_length = self.data_cache_ctx_shape[0]
        self.context_rank = 8

        self.encoder_channels_in = 3
        self.encoder_channels_conv = 8
        self.encoder_channels_contextual = 16
        self.encoder_channels_out = 8

        self.decoder_channels_in = 8
        self.decoder_channels_conv = 8
        self.decoder_channels_contextual = 16
        self.decoder_channels_out = 3

        self.kernel_size_c_sml = [3, 3]
        self.kernel_size_c_lrg = [7, 7]
        self.kernel_size_c_refine = [3, 3]

        self.dtype_weights = dtype_weights

        # ====> Block: data cache
        self.data_cache_ctx = nn.Parameter(
            data=torch.nn.init.uniform_(
                tensor=torch.empty(
                    [self.data_cache_ctx_len, self.context_length],
                    dtype=self.dtype_weights,
                ),
                a=self.data_cache_ctx_bound[0],
                b=self.data_cache_ctx_bound[1],
            ),
        )

        # ====> Block: input
        self.block_input_x_conv = DynamicConv2DAlpha(
            in_channels=self.encoder_channels_in,
            out_channels=self.encoder_channels_conv,
            context_length=self.context_length,
            context_rank=self.context_rank,
            context_use_bias=False,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )
        self.block_input_ctx_norm = nn.LayerNorm(
            normalized_shape=[self.context_length],
            elementwise_affine=True,
            bias=False,
            dtype=self.dtype_weights,
        )

        # ====> Encode block: 01
        self.encode_block_01 = Coder2DDynamicAlpha(
            context_length=self.context_length,
            context_rank=self.context_rank,
            conv_channels_in=self.encoder_channels_conv,
            conv_channels_out=self.encoder_channels_conv,
            conv_channels_intermediate=self.encoder_channels_contextual,
            conv_kernel_small=self.kernel_size_c_sml,
            conv_kernel_large=self.kernel_size_c_lrg,
            conv_kernel_refine=self.kernel_size_c_sml,
            interpolate_scale_factor=0.5,
            dtype_weights=self.dtype_weights,
        )

        # ====> Encode block: 02
        self.encode_block_02 = Coder2DDynamicAlpha(
            context_length=self.context_length,
            context_rank=self.context_rank,
            conv_channels_in=self.encoder_channels_conv,
            conv_channels_out=self.encoder_channels_conv,
            conv_channels_intermediate=self.encoder_channels_contextual,
            conv_kernel_small=self.kernel_size_c_sml,
            conv_kernel_large=self.kernel_size_c_lrg,
            conv_kernel_refine=self.kernel_size_c_sml,
            interpolate_scale_factor=0.5,
            dtype_weights=self.dtype_weights,
        )

        # ====> Encode block: 03
        self.encode_block_03 = Coder2DDynamicAlpha(
            context_length=self.context_length,
            context_rank=self.context_rank,
            conv_channels_in=self.encoder_channels_conv,
            conv_channels_out=self.encoder_channels_conv,
            conv_channels_intermediate=self.encoder_channels_contextual,
            conv_kernel_small=self.kernel_size_c_sml,
            conv_kernel_large=self.kernel_size_c_lrg,
            conv_kernel_refine=self.kernel_size_c_sml,
            interpolate_scale_factor=0.5,
            dtype_weights=self.dtype_weights,
        )

        # ====> Encode block: 04
        self.encode_block_04 = Coder2DDynamicAlpha(
            context_length=self.context_length,
            context_rank=self.context_rank,
            conv_channels_in=self.encoder_channels_conv,
            conv_channels_out=self.encoder_channels_conv,
            conv_channels_intermediate=self.encoder_channels_contextual,
            conv_kernel_small=self.kernel_size_c_sml,
            conv_kernel_large=self.kernel_size_c_lrg,
            conv_kernel_refine=self.kernel_size_c_sml,
            interpolate_scale_factor=0.5,
            dtype_weights=self.dtype_weights,
        )

        # ====> Encode block: 05
        self.encode_block_05 = Coder2DDynamicAlpha(
            context_length=self.context_length,
            context_rank=self.context_rank,
            conv_channels_in=self.encoder_channels_conv,
            conv_channels_out=self.encoder_channels_out,
            conv_channels_intermediate=self.encoder_channels_contextual,
            conv_kernel_small=self.kernel_size_c_sml,
            conv_kernel_large=self.kernel_size_c_lrg,
            conv_kernel_refine=self.kernel_size_c_sml,
            interpolate_scale_factor=0.5,
            dtype_weights=self.dtype_weights,
        )

        # ====> Decode block: 05
        self.decode_block_05 = Coder2DDynamicAlpha(
            context_length=self.context_length,
            context_rank=self.context_rank,
            conv_channels_in=self.decoder_channels_in,
            conv_channels_out=self.decoder_channels_conv,
            conv_channels_intermediate=self.decoder_channels_contextual,
            conv_kernel_small=self.kernel_size_c_sml,
            conv_kernel_large=self.kernel_size_c_lrg,
            conv_kernel_refine=self.kernel_size_c_sml,
            interpolate_scale_factor=2.0,
            dtype_weights=self.dtype_weights,
        )

        # ====> Decode block: 04
        self.decode_block_04 = Coder2DDynamicAlpha(
            context_length=self.context_length,
            context_rank=self.context_rank,
            conv_channels_in=self.decoder_channels_conv,
            conv_channels_out=self.decoder_channels_conv,
            conv_channels_intermediate=self.decoder_channels_contextual,
            conv_kernel_small=self.kernel_size_c_sml,
            conv_kernel_large=self.kernel_size_c_lrg,
            conv_kernel_refine=self.kernel_size_c_sml,
            interpolate_scale_factor=2.0,
            dtype_weights=self.dtype_weights,
        )

        # ====> Decode block: 03
        self.decode_block_03 = Coder2DDynamicAlpha(
            context_length=self.context_length,
            context_rank=self.context_rank,
            conv_channels_in=self.decoder_channels_conv,
            conv_channels_out=self.decoder_channels_conv,
            conv_channels_intermediate=self.decoder_channels_contextual,
            conv_kernel_small=self.kernel_size_c_sml,
            conv_kernel_large=self.kernel_size_c_lrg,
            conv_kernel_refine=self.kernel_size_c_sml,
            interpolate_scale_factor=2.0,
            dtype_weights=self.dtype_weights,
        )

        # ====> Decode block: 02
        self.decode_block_02 = Coder2DDynamicAlpha(
            context_length=self.context_length,
            context_rank=self.context_rank,
            conv_channels_in=self.decoder_channels_conv,
            conv_channels_out=self.decoder_channels_conv,
            conv_channels_intermediate=self.decoder_channels_contextual,
            conv_kernel_small=self.kernel_size_c_sml,
            conv_kernel_large=self.kernel_size_c_lrg,
            conv_kernel_refine=self.kernel_size_c_sml,
            interpolate_scale_factor=2.0,
            dtype_weights=self.dtype_weights,
        )

        # ====> Decode block: 01
        self.decode_block_01 = Coder2DDynamicAlpha(
            context_length=self.context_length,
            context_rank=self.context_rank,
            conv_channels_in=self.decoder_channels_conv,
            conv_channels_out=self.decoder_channels_conv,
            conv_channels_intermediate=self.decoder_channels_contextual,
            conv_kernel_small=self.kernel_size_c_sml,
            conv_kernel_large=self.kernel_size_c_lrg,
            conv_kernel_refine=self.kernel_size_c_sml,
            interpolate_scale_factor=2.0,
            dtype_weights=self.dtype_weights,
        )

        # ====> Block: out
        self.block_out_norm = nn.BatchNorm2d(
            num_features=self.decoder_channels_conv,
            eps=self.eps,
            affine=True,
            momentum=0.1,
            dtype=self.dtype_weights,
        )
        self.block_out_conv = DynamicConv2DAlpha(
            in_channels=self.decoder_channels_conv,
            out_channels=self.decoder_channels_out,
            context_length=self.context_length,
            context_rank=self.context_rank,
            context_use_bias=False,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            transpose=False,
            output_padding=None,
            dtype_weights=self.dtype_weights,
        )

        pass

    def forward(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        base_ctx = self.data_cache_ctx[ids]

        # Rename params.
        ctx = base_ctx

        # Block input
        x = self.block_input_x_conv(x, ctx)
        ctx = self.block_input_ctx_norm(ctx)

        # Encode
        x = self.encode_block_01(x, ctx)
        x = self.encode_block_02(x, ctx)
        x = self.encode_block_03(x, ctx)
        x = self.encode_block_04(x, ctx)
        x = self.encode_block_05(x, ctx)

        # Decode
        x = self.decode_block_05(x, ctx)
        x = self.decode_block_04(x, ctx)
        x = self.decode_block_03(x, ctx)
        x = self.decode_block_02(x, ctx)
        x = self.decode_block_01(x, ctx)

        # Block out
        x = self.block_out_norm(x)
        x = self.block_out_conv(x, ctx)
        x = torch.nn.functional.sigmoid(x)

        return x


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


def train(
    data: torch.Tensor,
    total_steps: int,
    batch_size: int,
    grad_accumulation_steps: int,
    model: DecoderOnlyModel,
    optimizer: torch.optim.Optimizer,
    clip_grad_value: float,
    clip_grad_norm: float,
    log_nth_update_step: int,
    images_path_dst: str = None,
    save_nth_iteration: int = 100,
    savers: list = [],
    to_save: list[bool] = [],
    warmup_scheduler: Optional[Callable] = None,
    warmup_epochs: Optional[int] = None,
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

    initially_decoded_samples = model(data_lab[0:4:1, ::], torch.arange(0, min(4, data_lab.shape[0]), 1))
    generate_images_from_data(
        data=data_lab_to_rgb(initially_decoded_samples),
        images_path_dst=images_path_dst,
        prefix="initial_state",
    )

    params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n# --------------------------------------------------- #\n")
    print(f"Model type: {type(model)}")
    print(f"Model parameters: {params_model}")
    print(f"Output samples shape: {initially_decoded_samples.shape[1:]}")
    print("\n# --------------------------------------------------- #\n")

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
        decoded = model(sample, batch_ids)

        # loss_base_targets = F.log_softmax(sample.flatten(), dim=0)
        # loss_base_decoded = F.log_softmax(decoded.flatten(), dim=0)
        # loss_current_step = F.kl_div(loss_base_decoded, loss_base_targets, reduction="sum", log_target=True)

        loss_current_step = mean_proportional_error(decoded, sample)

        loss_logging_accumulator.append(loss_current_step.detach().item())
        loss_total = loss_current_step / grad_accumulation_steps
        loss_total.backward()

        if accumulation_step == grad_accumulation_steps:
            weights_update_step_idx = weights_update_step_idx + 1
            accumulation_step = 0

            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    if epoch_idx > warmup_epochs:
                        lr_scheduler.step(epoch_idx - warmup_epochs)
            
            if clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value, foreach=True)
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm, norm_type=2.0, error_if_nonfinite=False, foreach=True)

            optimizer.step()

        if (weights_update_step_idx > 0 and (weights_update_step_idx) % log_nth_update_step == 0):
            weights_update_step_idx = 0
            print(
                "\n# ==============> "
                + "\n".join(
                    [
                        f"Iteration #{step_idx+1}:",
                        f"LR: {optimizer.param_groups[0]['lr']:.10f}",
                        f"Loss current: {loss_current_step.item()}",
                        f"Loss mean: {(sum(loss_logging_accumulator)/len(loss_logging_accumulator)):.5f}",
                        f"{model.data_cache_ctx.grad.abs().mean().tolist()=}",
                        f"{model.block_input_x_conv.weights_lib.weights_static.grad.abs().mean().tolist()=}",
                        f"{model.block_input_x_conv.weights_lib.weights_mod.grad.abs().mean().tolist()=}",
                        f"{model.encode_block_01.coder_block_conv_small.weights_lib.weights_static.grad.abs().mean().tolist()=}",
                        f"{model.encode_block_01.coder_block_conv_small.weights_lib.weights_mod.grad.abs().mean().tolist()=}",
                        f"{model.encode_block_01.coder_block_conv_large.weights_lib.weights_static.grad.abs().mean().tolist()=}",
                        f"{model.encode_block_01.coder_block_conv_large.weights_lib.weights_mod.grad.abs().mean().tolist()=}",
                    ]
                )
            )
            loss_logging_accumulator = []

            generate_images_from_data(
                data=data_lab_to_rgb(decoded[-1].unsqueeze(0).clamp(0.0, 1.0)),
                images_path_dst=images_path_dst,
                prefix=f"output_e{epoch_idx:0>4d}_i{(step_idx+1):0>7d}",
            )

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
        
        if accumulation_step == grad_accumulation_steps:
            optimizer.zero_grad()

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


def model_freeze_model(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache" in name:
            continue
        param.requires_grad = False
        print(f"freeze_model: {name}")
    pass


def model_unfreeze_model(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache" in name:
            continue
        param.requires_grad = True
        print(f"unfreeze_model: {name}")
    pass


def model_freeze_ctx(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache_ctx" in name:
            param.requires_grad = False
            print(f"freeze_ctx: {name}")
    pass


def model_unfreeze_ctx(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        if "data_cache_ctx" in name:
            param.requires_grad = True
            print(f"unfreeze_ctx: {name}")
    pass


def model_freeze_all(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = False
    print(f"model_freeze_all")
    pass


def model_unfreeze_all(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = True
    print(f"model_unfreeze_all")
    pass


def model_constant_ctx(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    v = 0.0001
    for name, param in model.named_parameters():
        if "data_cache_ctx" in name:
            param.data = torch.nn.init.constant_(
                param.data,
                val=v,
            )
            print(f"model_constant_ctx: {v}")
    pass


def model_perturb_small_weights(
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
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


def eval_infer(
    model: DecoderOnlyModel,
    start_from: int = 0,
    batches: int = 1,
    batch_size: int = 16,
) -> None:
    training = model.training

    if training:
        model.eval()

    for batch_id in range(0, batches, 1):
        id_start = start_from
        id_end = start_from + batch_size + batch_size * batch_id
        with torch.no_grad():
            model_output = model(torch.arange(id_start, id_end, 1))
            data_rgb = data_lab_to_rgb(model_output)
            generate_images_from_data(data_rgb, images_path_dst, f"infer_b{batch_id}")

    if training:
        model.train()

    return None


def eval_ctx_rand(
    model: DecoderOnlyModel,
) -> None:
    with torch.no_grad():
        ctx = model.data_cache_ctx.data.clone()
        model.data_cache_ctx.data = torch.randn_like(ctx)
        model_output = model(torch.arange(0, 16, 1))
        model.data_cache_ctx.data = ctx
        data_rgb = data_lab_to_rgb(model_output)
        generate_images_from_data(data_rgb, images_path_dst, "ctx_rand")
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


def model_change_data_cache_ctx(
    model: DecoderOnlyModel,
    device: torch.device,
    dtype: torch.dtype,
    ctx_size: list[int],
) -> None:
    for name, param in model.named_parameters():
        if "data_cache_ctx" in name:
            param.data = torch.randn(ctx_size, device=device, dtype=dtype)
            print(f"model_change_data_cache_ctx: {name}")
            continue
    return None


def model_freeze_block(
    model: DecoderOnlyModel,
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
    model: DecoderOnlyModel,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    block_name: str = "no_block_passed",
) -> None:
    for name, param in model.named_parameters():
        if block_name in name:
            param.requires_grad = True
            print(f"model_unfreeze_block: {name}")
    pass


def model_perturb_weights(
    model: DecoderOnlyModel,
    rate: float = 0.025,
    include_data_cache: bool = False,
) -> None:

    for name, param in model.named_parameters():
        if not include_data_cache and "data_cache" in name:
            continue
        std = param.data.std()
        if std == 0.0 or std != std:
            continue
        noise = torch.randn_like(param.data) * param.data.std() * rate
        param.data = param.data + noise
        print(f"model_perturb_weights (rate={rate:.5f}): {name}")

    pass


def clear() -> None:
    os.system("clear")
    pass


if __name__ == "__main__":
    train_mode = True

    load_model = True
    load_optim = False
    drop_ctx_cache = False
    onload_model_fn = [
        # model_constant_ctx,
        # lambda m, d, t: model_change_data_cache_ctx(m, d, t, [32, 256]),
        # lambda m, d, t: model_perturb_small_weights(m, a=0.001, b=0.005),
        # lambda m, d, t: model_perturb_weights(m, 0.10, False),
        # lambda m, d, t: model_extend_data_cache(m, 128, 1.0e-6),
        # lambda m, d, t: model_extend_data_cache(m, 512),
        # model_freeze_model,
        # model_freeze_ctx,
        # model_freeze_all,
        # model_unfreeze_model,
        # model_unfreeze_ctx,
        # model_unfreeze_all,
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_out"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_01"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_02"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_03"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_04"),
        # lambda m, d, t: model_unfreeze_block(m, d, t, "block_05"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_out"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_01"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_02"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_03"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_04"),
        # lambda m, d, t: model_freeze_block(m, d, t, "block_05"),
    ]
    onload_optim_fn = [
        # lambda o: optim_change_momentum(o, 0.9),
    ]

    path_prefix_load = "f:\\git_AIResearch\\dyna\\data\\models"
    path_prefix_save = "f:\\git_AIResearch\\dyna\\data\\models"
    load_path_model = f"{path_prefix_load}\\02\\model.Type-00.G02.__LAST__.pth"
    load_path_optim = f"{path_prefix_load}\\"
    save_path_model = f"{path_prefix_save}\\model.Type-00.G03"
    save_path_optim = f"{path_prefix_save}\\optim.Type-00.G03"
    save_model = True
    save_optim = True
    save_nth_iteration = 8192
    log_nth_update_step = 1

    # optimizer type
    optimizer_type = bnb.optim.AdamW8bit
    # optimizer: torch.optim.SGD
    sgd_learning_rate = 1.0e-1
    sgd_momentum = 0.0
    sgd_dampening = 0.0
    sgd_weight_decay = 0.0
    sgd_nesterov = False
    # optimizer: torch.optim.Adam
    adam_learning_rate = 2.5e-4
    adam_amsgrad = True
    adam_weight_decay = 0.0
    adam_eps = 1.0e-8
    # optimizer: torch.optim.AdamW
    adamw_learning_rate = 1.0e-3
    adamw_amsgrad = True
    adamw_weight_decay = 1.0e-2
    adamw_eps = 1.0e-8
    # optimizer: MADGRAD
    madgrad_learning_rate = 1.0e-2
    madgrad_momentum = 0.9
    madgrad_weight_decay = 0.0
    madgrad_eps = 1.0e-6
    # optimizer: bnb.optim.AdamW8bit
    bnb_adamw8bit_lr=1e-3
    bnb_adamw8bit_betas=(0.9, 0.999)
    bnb_adamw8bit_eps=1e-8
    bnb_adamw8bit_weight_decay=1e-2
    bnb_adamw8bit_amsgrad=True
    bnb_adamw8bit_optim_bits=8
    bnb_adamw8bit_min_8bit_size=4096
    bnb_adamw8bit_percentile_clipping=100
    bnb_adamw8bit_block_wise=True
    bnb_adamw8bit_is_paged=False
    # various for optimizers
    optim_update_lr = False
    optim_target_lr = 1.0e-3
    optim_update_wd = False
    optim_target_wd = 0.1
    warmup_active = False
    warmup_epochs = 1024
    clip_grad_value = None
    clip_grad_norm = 1.0

    data_cache_ctx_bound = [-1.0e-4, +1.0e-4]

    nelements = 32 * 128
    data_cache_ctx_len = nelements
    data_cache_ctx_shape = [128]

    total_steps = 200_000
    batch_size = 32
    grad_accumulation_steps = nelements // batch_size

    images_sample_count = nelements
    starting_from = 1024 * 0
    images_path_src = "f:\\git_AIResearch\\dyna\\data\\img_src_1"
    images_path_dst = "f:\\git_AIResearch\\dyna\\data\\img_dst_1"
    output_shape = [512, 512]
    dtype_weights = torch.bfloat16
    device = torch.device("cuda")

    model = DecoderOnlyModel(
        data_cache_ctx_len=data_cache_ctx_len,
        data_cache_ctx_shape=data_cache_ctx_shape,
        data_cache_ctx_bound=data_cache_ctx_bound,
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

    if drop_ctx_cache:
        model.data_cache_ctx = nn.Parameter(
            torch.nn.init.uniform_(
                model.data_cache_ctx,
                a=data_cache_ctx_bound[0],
                b=data_cache_ctx_bound[1],
            ).to(
                dtype=dtype_weights,
            )
        )
        print("Dropped: data_cache_ctx")

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

    if optimizer_type is torch.optim.Adam:
        print("Using Adam")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=adam_learning_rate,
            weight_decay=adam_weight_decay,
            amsgrad=adam_amsgrad,
            eps=adam_eps,
        )
    elif optimizer_type is torch.optim.AdamW:
        print("Using AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=adamw_learning_rate,
            weight_decay=adamw_weight_decay,
            amsgrad=adamw_amsgrad,
            eps=adamw_eps,
        )
    elif optimizer_type is MADGRAD:
        print("Using MADGRAD")
        optimizer = MADGRAD(
            model.parameters(),
            lr=madgrad_learning_rate,
            weight_decay=madgrad_weight_decay,
            momentum=madgrad_momentum,
            eps=madgrad_eps,
        )
    elif optimizer_type is torch.optim.SGD:
        print("Using SGD")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=sgd_learning_rate,
            momentum=sgd_momentum,
            dampening=sgd_dampening,
            weight_decay=sgd_weight_decay,
            nesterov=sgd_nesterov,
        )
    elif optimizer_type is bnb.optim.AdamW8bit:
        print("Using bnb.optim.AdamW8bit")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=bnb_adamw8bit_lr,
            betas=bnb_adamw8bit_betas,
            eps=bnb_adamw8bit_eps,
            weight_decay=bnb_adamw8bit_weight_decay,
            amsgrad=bnb_adamw8bit_amsgrad,
            optim_bits=bnb_adamw8bit_optim_bits,
            min_8bit_size=bnb_adamw8bit_min_8bit_size,
            percentile_clipping=bnb_adamw8bit_percentile_clipping,
            block_wise=bnb_adamw8bit_block_wise,
            is_paged=bnb_adamw8bit_is_paged,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

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
    torch.save(model.state_dict(), f"{save_path_model}.__INITIAL__.pth")
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
            warmup_period=warmup_epochs,
        )
    else:
        lr_scheduler = None
        warmup_epochs = None
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
        log_nth_update_step=log_nth_update_step,
        images_path_dst=images_path_dst,
        save_nth_iteration=save_nth_iteration,
        savers=savers,
        to_save=[save_model, save_optim],
        warmup_scheduler=warmup_scheduler,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
    )

    if train_mode:
        start_training()
