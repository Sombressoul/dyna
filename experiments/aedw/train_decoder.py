import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import kornia
import math
import gc

from PIL import Image
from madgrad import MADGRAD

script_dir = os.path.dirname(os.path.abspath(__file__))
evals_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(evals_dir)
sys.path.append(project_dir)

from dyna import DynamicConv2D
from time import sleep


class ExperimentalModel(nn.Module):
    def __init__(
        self,
        data_cache_ctx_len: int = None,
        data_cache_latents_len: int = None,
        data_cache_latents_shape: list[int] = None,
        dtype_weights: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.data_cache_ctx_len = data_cache_ctx_len
        self.data_cache_latents_len = data_cache_latents_len
        self.data_cache_latents_shape = data_cache_latents_shape
        self.use_bias = True
        self.bias_static = 0.0
        self.context_length = 32
        self.mod_rank = 16

        self.kernel_size_t = [4, 4]
        self.kernel_size_r = [3, 3]
        self.kernel_size_m = [5, 5]

        self.eps = 1.0e-2
        self.q_levels = 100

        self.channels_io = 3
        self.channels_dynamic_levels = [8, 8, 8, 8, 8]

        self.dtype_weights = dtype_weights

        self.upsample_nearest = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample_bilinear = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv_up_04_t = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[4],
            out_channels=self.channels_dynamic_levels[3],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_t,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=[0, 0],
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_04_r = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[4],
            out_channels=self.channels_dynamic_levels[3],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_r,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_04_m = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[4],
            out_channels=self.channels_dynamic_levels[3],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_m,
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_03_t = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[3],
            out_channels=self.channels_dynamic_levels[2],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_t,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=[0, 0],
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_03_r = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[3],
            out_channels=self.channels_dynamic_levels[2],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_r,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_03_m = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[3],
            out_channels=self.channels_dynamic_levels[2],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_m,
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_02_t = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[2],
            out_channels=self.channels_dynamic_levels[1],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_t,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=[0, 0],
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_02_r = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[2],
            out_channels=self.channels_dynamic_levels[1],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_r,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_02_m = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[2],
            out_channels=self.channels_dynamic_levels[1],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_m,
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_01_t = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[1],
            out_channels=self.channels_dynamic_levels[0],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_t,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=[0, 0],
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_01_r = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[1],
            out_channels=self.channels_dynamic_levels[0],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_r,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_up_01_m = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[1],
            out_channels=self.channels_dynamic_levels[0],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=self.kernel_size_m,
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.conv_out = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[0],
            out_channels=self.channels_io,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )

        self.data_cache_ctx = nn.Parameter(
            data=self._init_context(device=torch.device("cuda")),
        )
        self.data_cache_latents = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [self.data_cache_latents_len, *self.data_cache_latents_shape],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=1.0e-2,
            ),
        )

        pass

    def _init_context(self, device: torch.device) -> torch.Tensor:
        # uniform: -0.01...+0.01
        new_context = torch.nn.init.normal_(
            tensor=torch.empty(
                [self.data_cache_ctx_len, self.context_length],
                dtype=self.dtype_weights,
                device=device,
            ),
            mean=0.0,
            std=1.0e-2,
        )
        return new_context

    def quantizer(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            l = self.q_levels
            shift = 1.0 / (l * 2)
            x_q = (x * l).clamp(-l, +l)
            x_q = (x_q // 1.0).to(dtype=x.dtype, device=x.device)
            x_q = (x_q / l) + shift
        x = x + (x_q - x).detach()
        return x

    def doubleLogNormAbs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.abs()
        x = x.sub(x.min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1))
        x = x.add(torch.e).log().add(self.eps).log()
        return x

    def doubleLogNorm(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_mul = torch.where(x > 0.0, +1.0, -1.0).to(dtype=x.dtype, device=x.device)
        x = x.abs()
        x = x.sub(x.min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1))
        x = x.add(torch.e).log().add(self.eps).log()
        x = x.mul(x_mul)
        return x

    def forward(
        self,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        ids = ids.to(device=self.data_cache_latents.device, dtype=torch.int32)
        context = self.data_cache_ctx[ids]
        x = self.data_cache_latents[ids]

        x_pos, x_neg, x_mul = x, -x, x.abs().add(1.0).log()
        x_pos = self.conv_up_04_r(x_pos, context)
        x_pos = self.doubleLogNorm(x_pos)
        x_pos = self.upsample_nearest(x_pos)
        x_neg = self.conv_up_04_t(x_neg, context)
        x_neg = self.doubleLogNorm(x_neg)
        x_mul = self.upsample_bilinear(x_mul)
        x_mul = self.conv_up_04_m(x_mul, context)
        x_mul = self.doubleLogNorm(x_mul)
        x = (x_pos + x_neg) * x_mul

        x_pos, x_neg, x_mul = x, -x, x.abs().add(1.0).log()
        x_pos = self.conv_up_03_r(x_pos, context)
        x_pos = self.doubleLogNorm(x_pos)
        x_pos = self.upsample_nearest(x_pos)
        x_neg = self.conv_up_03_t(x_neg, context)
        x_neg = self.doubleLogNorm(x_neg)
        x_mul = self.upsample_bilinear(x_mul)
        x_mul = self.conv_up_03_m(x_mul, context)
        x_mul = self.doubleLogNorm(x_mul)
        x = (x_pos + x_neg) * x_mul

        x_pos, x_neg, x_mul = x, -x, x.abs().add(1.0).log()
        x_pos = self.conv_up_02_r(x_pos, context)
        x_pos = self.doubleLogNorm(x_pos)
        x_pos = self.upsample_nearest(x_pos)
        x_neg = self.conv_up_02_t(x_neg, context)
        x_neg = self.doubleLogNorm(x_neg)
        x_mul = self.upsample_bilinear(x_mul)
        x_mul = self.conv_up_02_m(x_mul, context)
        x_mul = self.doubleLogNorm(x_mul)
        x = (x_pos + x_neg) * x_mul

        x_pos, x_neg, x_mul = x, -x, x.abs().add(1.0).log()
        x_pos = self.conv_up_01_r(x_pos, context)
        x_pos = self.doubleLogNorm(x_pos)
        x_pos = self.upsample_nearest(x_pos)
        x_neg = self.conv_up_01_t(x_neg, context)
        x_neg = self.doubleLogNorm(x_neg)
        x_mul = self.upsample_bilinear(x_mul)
        x_mul = self.conv_up_01_m(x_mul, context)
        x_mul = self.doubleLogNorm(x_mul)
        x = (x_pos + x_neg) * x_mul

        x = self.conv_out(x, context)
        x = F.sigmoid(x)

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


def fn_loss(inputs, targets) -> torch.Tensor:
    sq_diff = (inputs**2 - targets**2).abs()
    diff_min = sq_diff.min()
    diff_max = sq_diff.max()
    mult = 1.5 - ((sq_diff - diff_min) / (diff_max - diff_min))
    loss = (inputs - targets).mul(mult).std().sqrt()
    loss = loss.mean()
    return loss


def train(
    data: torch.Tensor,
    total_steps: int,
    batch_size: int,
    grad_accumulation_steps: int,
    sliding_batch: bool,
    loss_channels_weights: torch.Tensor,
    regularization_alpha_model: float,
    regularization_alpha_ctx: float,
    regularization_alpha_latents: float,
    regularization_low_weights_bound: float,
    regularization_low_weights_alpha: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    log_nth_iteration: int,
    images_path_dst: str = None,
    save_nth_step: int = 100,
    savers: list = [],
    to_save: list[bool] = [],
) -> None:
    var_logger = model.conv_out.weights_lib._log_var
    params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n# --------------------------------------------------- #\n")
    print(f"Model type: {type(model)}")
    print(f"Model parameters: {params_model}")
    print("\n# --------------------------------------------------- #\n")

    with torch.no_grad():
        data_lab = data.cpu().to(dtype=model.dtype_weights)
        data_lab = data_lab / 255.0
        data_lab = data_rgb_to_lab(data_lab)
        data_lab = data_lab.to(dtype=torch.float16, device=data.device)
        data = data.cpu().detach()
        gc.collect()

    loss_channels_weights = loss_channels_weights.to(data_lab.device)

    accumulation_step = 0
    loss_accumulator = 0
    epoch_ids = torch.randperm(data_lab.shape[0])
    epoch_idx = 0
    print(f"Starting training. Epoch #{epoch_idx}")
    for step_idx in range(total_steps):
        if len(epoch_ids) - 1 < batch_size:
            epoch_idx = epoch_idx + 1
            print(
                f"\n# ==================================================> New epoch: #{epoch_idx}"
            )
            epoch_ids = torch.randperm(data_lab.shape[0])

        batch_ids = epoch_ids[0:batch_size]
        epoch_ids = epoch_ids[(1 if sliding_batch else batch_size) :]

        sample = data_lab[batch_ids]
        sample = sample.to(dtype=model.dtype_weights)

        accumulation_step = accumulation_step + 1
        decoded = model(batch_ids)

        loss_base_decoded = torch.einsum("ijkl,j->ijkl", decoded, loss_channels_weights)
        loss_base_targets = torch.einsum("ijkl,j->ijkl", sample, loss_channels_weights)
        loss = F.l1_loss(loss_base_decoded, loss_base_targets)

        # Regularizers
        reg_term_model = get_regularization_term_model(
            model=model,
            alpha=regularization_alpha_model,
        )
        reg_term_ctx = get_regularization_term_ctx(
            model=model,
            alpha=regularization_alpha_ctx,
        )
        reg_term_latents = get_regularization_term_latents(
            model=model,
            alpha=regularization_alpha_latents,
        )
        reg_term_low_weights = get_regularization_term_low_weights(
            model=model,
            bound=regularization_low_weights_bound,
            alpha=regularization_low_weights_alpha,
        )

        loss = (
            loss
            + reg_term_model
            + reg_term_ctx
            + reg_term_latents
            + reg_term_low_weights
        )

        if accumulation_step == grad_accumulation_steps:
            accumulation_step = 0

            loss_accumulator = loss_accumulator + loss
            loss_accumulator = loss_accumulator / grad_accumulation_steps

            optimizer.zero_grad()
            loss_accumulator.backward()
            optimizer.step()

            loss_accumulator = 0
        elif loss_accumulator == 0:
            loss_accumulator = loss
        else:
            loss_accumulator = loss_accumulator + loss

        # print(f"step loss: {loss.item()}")

        if (step_idx + 1) % log_nth_iteration == 0:
            print(
                "\n# ==============> "
                + "\n".join(
                    [
                        f"Iteration #{step_idx+1}:",
                        f"Loss: {loss.item()}",
                        f"StdR: {(sample - decoded).std()}",
                    ]
                )
                + "\n# <=============="
            )
            # var_logger(model.data_cache_ctx, "model.data_cache_ctx", False)
            # var_logger(model.data_cache_latents, "model.data_cache_latents", False)
            generate_images_from_data(
                data=data_lab_to_rgb(decoded[-1].unsqueeze(0)),
                images_path_dst=images_path_dst,
                prefix=f"output_e{epoch_idx:0>4d}_i{(step_idx+1):0>7d}",
            )
        if (step_idx + 1) % save_nth_step == 0:
            for j, saver in enumerate(savers):
                if to_save[j]:
                    saver(step_idx + 1)
            pass

    print("\n# --------------------------------------------------- #\n")

    generate_images_from_data(
        data=data_lab_to_rgb(decoded[-1].unsqueeze(0)),
        images_path_dst=images_path_dst,
        prefix=f"output_final_i{step_idx+1}",
    )

    pass


def on_model_load(model: ExperimentalModel) -> None:
    # for param in model.parameters():
    #     param.requires_grad = True

    # model.data_cache_ctx.requires_grad = True
    # model.data_cache_latents.requires_grad = True

    device = model.data_cache_ctx.device

    model.data_cache_ctx = nn.Parameter(
        torch.nn.init.normal_(
            torch.empty(
                [
                    model.data_cache_ctx.shape[0] * 2,
                    *model.data_cache_ctx.shape[1:],
                ],
                device=device,
            ),
            mean=0.0,
            std=1.0e-1,
        )
    ).to(
        dtype=model.data_cache_ctx.dtype,
    )
    model.data_cache_latents = nn.Parameter(
        torch.nn.init.normal_(
            torch.empty(
                [
                    model.data_cache_latents.shape[0] * 2,
                    *model.data_cache_latents.shape[1:],
                ],
                device=device,
            ),
            mean=0.0,
            std=1.0e-1,
        ),
    ).to(
        dtype=model.data_cache_latents.dtype,
    )
    pass


def get_regularization_term_model(
    model: nn.Module,
    alpha: float,
) -> torch.Tensor:
    sum = 0.0
    for name, param in model.named_parameters():
        if "data_cache" in name:
            continue
        else:
            sum = sum + (param**2).sum()
    return alpha * sum


def get_regularization_term_ctx(
    model: nn.Module,
    alpha: float,
) -> torch.Tensor:
    sum = 0.0
    for name, param in model.named_parameters():
        if "data_cache_ctx" in name:
            sum = sum + (param**2).sum()
    return alpha * (sum / model.data_cache_ctx.shape[0])


def get_regularization_term_latents(
    model: nn.Module,
    alpha: float,
) -> torch.Tensor:
    sum = 0.0
    for name, param in model.named_parameters():
        if "data_cache_latents" in name:
            sum = sum + (param**2).sum()
    return alpha * (sum / model.data_cache_latents.shape[0])


def get_regularization_term_low_weights(
    model: nn.Module,
    bound: float,
    alpha: float,
) -> torch.Tensor:
    sum = 0.0
    for param in model.parameters():
        vars = param.abs().clamp(0.0, bound).sub(bound).abs()
        sum = sum + vars.sum()
    return alpha * sum


if __name__ == "__main__":
    train_mode = True

    load_model = True
    load_optim = True
    drop_ctx_cache = False
    drop_latents_cache = False
    onload = None

    path_prefix_load = "/mnt/f/git_AIResearch/dyna/data/models/aedw/decoder_model"
    path_prefix_save = "/mnt/f/git_AIResearch/dyna/data/models/aedw/decoder_model"
    load_path_model = f"{path_prefix_load}/decoder_model_s1.200000.pth"
    load_path_optim = f"{path_prefix_load}/decoder_optim_s1.200000.pth"
    save_path_model = f"{path_prefix_save}/test_model"
    save_path_optim = f"{path_prefix_save}/test_model"
    save_model = True
    save_optim = True
    save_nth_step = 10000
    log_nth_iteration = 100

    learning_rate = 1.0e-2
    momentum = 0.9
    weight_decay = 0.0
    eps = 1.0e-5
    regularization_alpha_model = 2.5e-7
    regularization_alpha_ctx = 2.5e-4
    regularization_alpha_latents = 2.0e-6
    regularization_low_weights_bound = 1.0e-3
    regularization_low_weights_alpha = 1.0e-3

    data_cache_ctx_len = 4096
    data_cache_latents_len = 4096
    data_cache_latents_shape = [8, 32, 32]

    total_steps = 200_000
    batch_size = 64
    sliding_batch = False
    grad_accumulation_steps = 1
    loss_channels_weights = [2.0, 1.0, 1.0]

    images_sample_count = 4096
    starting_from = 1024 * 8
    images_path_src = "/mnt/f/Datasets/Images_512x512/dataset_01"
    images_path_dst = "/mnt/f/git_AIResearch/dyna/data/img_dst"
    output_shape = [512, 512]
    dtype_weights = torch.float32
    device = torch.device("cuda")

    model = ExperimentalModel(
        data_cache_ctx_len=data_cache_ctx_len,
        data_cache_latents_len=data_cache_latents_len,
        data_cache_latents_shape=data_cache_latents_shape,
        dtype_weights=dtype_weights,
    ).to(device=device, dtype=dtype_weights)

    if load_model:
        inconsistent_keys = model.load_state_dict(
            torch.load(load_path_model),
            strict=False,
        )
        print(f"Model loaded from {load_path_model}")

    if onload is not None and callable(onload):
        onload(model)

    if drop_ctx_cache:
        model.data_cache_ctx = nn.Parameter(
            torch.nn.init.normal_(
                model.data_cache_ctx,
                mean=0.0,
                std=1.0e-2,
            ).to(
                dtype=dtype_weights,
            )
        )

    if drop_latents_cache:
        model.data_cache_latents = nn.Parameter(
            torch.nn.init.normal_(
                model.data_cache_latents,
                mean=0.0,
                std=1.0e-2,
            ).to(
                dtype=dtype_weights,
            )
        )

    optimizer = MADGRAD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        eps=eps,
    )

    if load_optim:
        optimizer.load_state_dict(torch.load(load_path_optim))
        print(f"Optimizer loaded from {load_path_optim}")

    with torch.no_grad():
        data = generate_data_from_images(
            shape=output_shape,
            images_path_src=images_path_src,
            images_sample_count=images_sample_count,
            starting_from=starting_from,
        ).to(device)

    print("Generated data specs:")
    print(f"{data.min()=}")
    print(f"{data.max()=}")

    savers = [
        lambda p: torch.save(model.state_dict(), f"{save_path_model}.{p}.pth"),
        lambda p: torch.save(optimizer.state_dict(), f"{save_path_optim}.{p}.pth"),
    ]

    if train_mode:
        train(
            data=data,
            total_steps=total_steps,
            batch_size=batch_size,
            grad_accumulation_steps=grad_accumulation_steps,
            sliding_batch=sliding_batch,
            loss_channels_weights=torch.tensor(loss_channels_weights),
            regularization_alpha_model=regularization_alpha_model,
            regularization_alpha_ctx=regularization_alpha_ctx,
            regularization_alpha_latents=regularization_alpha_latents,
            regularization_low_weights_bound=regularization_low_weights_bound,
            regularization_low_weights_alpha=regularization_low_weights_alpha,
            model=model,
            optimizer=optimizer,
            log_nth_iteration=log_nth_iteration,
            images_path_dst=images_path_dst,
            save_nth_step=save_nth_step,
            savers=savers,
            to_save=[save_model, save_optim],
        )
