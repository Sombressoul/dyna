import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import kornia
import math

from PIL import Image
from madgrad import MADGRAD

script_dir = os.path.dirname(os.path.abspath(__file__))
evals_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(evals_dir)
sys.path.append(project_dir)

from model import AEDW
from dyna import DynamicConv2D


class TestModel(nn.Module):
    def __init__(
        self,
        data_len: int,
        drop_context: bool = False,
        dtype_weights: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.drop_context = drop_context
        self.data_len = data_len
        self.dtype_weights = dtype_weights
        self.use_bias = True
        self.bias_static = 0.0
        self.context_length = 16
        self.mod_rank = 8

        self.eps = 1.0e-2
        self.q_levels = 100

        self.channels_io = 3
        self.channels_pre_output = 4
        self.channels_dynamic_levels = [8, 8, 8, 8, 8, 8]

        self.context = nn.Parameter(
            data=self._init_context(device=torch.device("cuda")),
        )

        self.down_01 = DynamicConv2D(
            in_channels=self.channels_io,
            out_channels=self.channels_dynamic_levels[0],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.down_02 = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[0],
            out_channels=self.channels_dynamic_levels[1],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.down_03 = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[1],
            out_channels=self.channels_dynamic_levels[2],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.down_04 = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[2],
            out_channels=self.channels_dynamic_levels[3],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )
        self.down_05 = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[3],
            out_channels=self.channels_dynamic_levels[4],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )

        self.middle = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[4],
            out_channels=self.channels_dynamic_levels[5],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[2, 2],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=True,
            output_padding=[0, 0],
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )

        self.up_05 = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[5],
            out_channels=self.channels_dynamic_levels[4],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[4, 4],
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
        self.up_04 = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[4],
            out_channels=self.channels_dynamic_levels[3],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[4, 4],
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
        self.up_03 = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[3],
            out_channels=self.channels_dynamic_levels[2],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[4, 4],
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
        self.up_02 = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[2],
            out_channels=self.channels_dynamic_levels[1],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[4, 4],
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
        self.up_01 = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[1],
            out_channels=self.channels_dynamic_levels[0],
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[4, 4],
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

        self.out_up_conv = DynamicConv2D(
            in_channels=self.channels_dynamic_levels[0],
            out_channels=self.channels_pre_output,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[4, 4],
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

        self.out_down_conv = DynamicConv2D(
            in_channels=self.channels_pre_output,
            out_channels=self.channels_io,
            context_length=self.context_length,
            mod_rank=self.mod_rank,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            bias_dynamic=self.use_bias,
            bias_static=self.bias_static,
            transpose=False,
            output_padding=None,
            asymmetry=1.0e-3,
            dtype_weights=self.dtype_weights,
        )

        pass

    def _init_context(self, device: torch.device) -> torch.Tensor:
        # uniform: -0.01...+0.01
        new_context = torch.nn.init.normal_(
            tensor=torch.empty(
                [self.data_len, self.context_length],
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
            shift = 1.0 / (self.q_levels * 2)
            x_q = (x * l).clamp(-l, +l)
            x_q = (x_q // 1.0).to(dtype=x.dtype, device=x.device)
            x_q = (x_q / l) + shift
        x = x + (x_q - x).detach()
        return x

    def logActivationAbs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.abs()
        x = x.sub(x.min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1))
        x = x.add(torch.e).log().add(self.eps).log()
        return x

    def logActivation(
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
        x: torch.Tensor,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        src_dtype = x.dtype
        x = x.to(dtype=self.dtype_weights)

        act_fn = self.logActivationAbs

        try:
            context = self.context[ids]

            if self.drop_context:
                print(
                    "\n".join(
                        [
                            "WARNING: Context is being dropped.",
                            "         This is a temporary workaround.",
                            "         It will be removed in the future.",
                        ]
                    )
                )
                self.drop_context = False
                raise AttributeError()
        except AttributeError:
            self.context.data = nn.Parameter(
                data=self._init_context(device=x.device),
            )
            context = self.context[ids]

        x = self.down_01(x, context)  # 256 (x2)
        x = act_fn(x)
        # x = self.quantizer(x)
        # print(f"down_01.shape: {x.shape}")
        x = self.down_02(x, context)  # 128 (x4)
        x = act_fn(x)
        # x = self.quantizer(x)
        # print(f"down_02.shape: {x.shape}")
        x = self.down_03(x, context)  # 64 (x8)
        x = act_fn(x)
        # x = self.quantizer(x)
        # print(f"down_03.shape: {x.shape}")
        x = self.down_04(x, context) # 32 (x16)
        x = act_fn(x)
        # x = self.quantizer(x)
        # print(f"down_04.shape: {x.shape}")
        # x = self.down_05(x, context) # 16 (x32)
        # x = act_fn(x)
        # x = self.quantizer(x)
        # print(f"down_05.shape: {x.shape}")

        x = self.middle(x, context)
        x = act_fn(x)
        # x = self.quantizer(x)
        # print(f"middle.shape: {x.shape}")

        # x = self.up_05(x, context)
        # x = act_fn(x)
        # x = self.quantizer(x)
        # print(f"up_05.shape: {x.shape}")
        x = self.up_04(x, context)
        x = act_fn(x)
        # x = self.quantizer(x)
        # print(f"up_04.shape: {x.shape}")
        x = self.up_03(x, context)
        x = act_fn(x)
        x = self.quantizer(x)
        # print(f"up_03.shape: {x.shape}")
        x = self.up_02(x, context)
        x = act_fn(x)
        x = self.quantizer(x)
        # print(f"up_02.shape: {x.shape}")
        x = self.up_01(x, context)
        x = act_fn(x)
        x = self.quantizer(x)
        # print(f"up_01.shape: {x.shape}")

        x = self.out_up_conv(x, context)
        x = act_fn(x)
        # x = self.quantizer(x)
        # print(f"out_up_conv.shape: {x.shape}")
        x = self.out_down_conv(x, context)
        x = act_fn(x)
        x = x.to(dtype=src_dtype)
        # print(f"out_down_conv.shape: {x.shape}")
        # exit()

        return x, None, None


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
        image = Image.open(image_path)
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
    samples_per_step: int,
    model: AEDW,
    optimizer: torch.optim.Optimizer,
    iterations: int,
    log_nth_iteration: int,
    results_sample_count: int,
    images_path_dst: str = None,
    save_nth_step: int = 100,
    savers: list = [],
    to_save: list[bool] = [],
) -> None:
    params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n# --------------------------------------------------- #\n")
    print(f"Model type: {type(model)}")
    print(f"Model parameters: {params_model}")

    print("\n# --------------------------------------------------- #\n")
    for i in range(iterations):
        ids = torch.randperm(data.shape[0])[0:samples_per_step]
        sample = data[ids].to(dtype=model.dtype_weights)

        with torch.no_grad():
            sample = sample.mul(1.0 / 255.0)
            sample = data_rgb_to_lab(sample)

        optimizer.zero_grad()
        decoded, encoded, context = model(sample, ids)
        loss = (sample - decoded).std().sqrt()
        # loss = F.mse_loss(decoded, sample)
        print(f"step loss: {loss.item()}")
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 1.0e-5)
        optimizer.step()

        if (i + 1) % log_nth_iteration == 0:
            print(
                "\n# ==============> #"
                + "\n".join(
                    [
                        f"Iteration #{i+1}:",
                        f"Loss: {loss.item()}",
                        f"StdR: {(sample - decoded).std()}",
                    ]
                )
                + "\n# <============== #"
            )
            model.down_01.weights_lib._log_var(model.context, "model.context", False)
            generate_images_from_data(
                data=data_lab_to_rgb(decoded[0:results_sample_count]),
                images_path_dst=images_path_dst,
                prefix=f"output_i{i+1}",
            )
        if (i + 1) % save_nth_step == 0:
            for j, saver in enumerate(savers):
                if to_save[j]:
                    saver()
            pass

    print("\n# --------------------------------------------------- #\n")

    generate_images_from_data(
        data=data_lab_to_rgb(decoded[0:results_sample_count]),
        images_path_dst=images_path_dst,
        prefix=f"output_final_i{i+1}",
    )

    pass


if __name__ == "__main__":
    train_mode = False

    load_model = False
    load_optim = False

    path_prefix = "/mnt/f/git_AIResearch/dyna/data/models/aedw"
    load_path_model = f"{path_prefix}/last_model.pth"
    load_path_optim = f"{path_prefix}/last_optim.pth"
    save_path_model = f"{path_prefix}/last_model.pth"
    save_path_optim = f"{path_prefix}/last_optim.pth"
    save_model = True
    save_optim = True
    save_nth_step = 1000
    drop_context = False

    iterations = 50_000
    log_nth_iteration = 10
    results_sample_count = 1

    learning_rate = 1.0e-2
    momentum = 0.9
    weight_decay = 0.0
    eps = 1.0e-3

    images_sample_count = 8192
    starting_from = 0
    samples_per_step = 64
    images_path_src = "/mnt/f/Datasets/Images_512x512/dataset_01"
    images_path_dst = "/mnt/f/git_AIResearch/dyna/data/img_dst"
    output_shape = [512, 512]
    dtype_weights = torch.float32
    device = torch.device("cuda")

    # model = AEDW(
    #     shape_base=output_shape,
    #     channels_io=3,
    #     depth=5,
    #     context_length=16,
    #     channels_base=16,
    #     channels_dynamic=32,
    #     channels_multiplier=2,
    #     eps=eps,
    #     dtype_weights=dtype_weights,
    # ).to(device=device, dtype=dtype_weights)
    model = TestModel(
        data_len=images_sample_count,
        dtype_weights=dtype_weights,
        drop_context=drop_context,
    ).to(device=device, dtype=dtype_weights)

    if load_model:
        model.load_state_dict(torch.load(load_path_model))
        print(f"Model loaded from {load_path_model}")

    optimizer = MADGRAD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        eps=eps,
    )

    if load_optim:
        optimizer.load_state_dict(torch.load(load_path_optim))

    with torch.no_grad():
        data = generate_data_from_images(
            shape=output_shape,
            images_path_src=images_path_src,
            images_sample_count=images_sample_count,
            starting_from=starting_from,
        ).to(device)

    # generate_images_from_data(
    #     data=data[torch.randperm(data.shape[0])[0 : min(4, data.shape[0])]],
    #     images_path_dst=images_path_dst,
    #     prefix=f"__data_sample__",
    # )

    print("Generated data specs:")
    print(f"{data.min()=}")
    print(f"{data.max()=}")
    # print(f"{data.mean()=}")
    # print(f"{data.std()=}")

    savers = [
        lambda: torch.save(model.state_dict(), save_path_model),
        lambda: torch.save(optimizer.state_dict(), save_path_optim),
    ]

    if train_mode:
        train(
            data=data,
            samples_per_step=samples_per_step,
            model=model,
            optimizer=optimizer,
            iterations=iterations,
            log_nth_iteration=log_nth_iteration,
            results_sample_count=results_sample_count,
            images_path_dst=images_path_dst,
            save_nth_step=save_nth_step,
            savers=savers,
            to_save=[save_model, save_optim],
        )
