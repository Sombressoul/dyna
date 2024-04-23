import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
evals_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(evals_dir)
sys.path.append(project_dir)

from dyna import WeightsLib2DLite


model = None


class Model(nn.Module):
    def __init__(
        self,
        output_shape: list[int] = [128, 128],
        mat_count: int = 16,
        components_count: int = 16,
        mod_rank: int = 8,
        asymmetry: float = 1e-3,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        # ================================================================================= #
        # ____________________________> Parameters.
        # ================================================================================= #
        self.mat_count = mat_count
        self.components_count = components_count
        self.output_shape = output_shape

        # ================================================================================= #
        # ____________________________> Weights.
        # ================================================================================= #
        # Init WeightsLib2DLite.
        self.weights = WeightsLib2DLite(
            output_shape=output_shape,
            mod_rank=mod_rank,
            components_count=components_count,
            asymmetry=asymmetry,
            dtype_weights=dtype_weights,
        )

        # Init coefficients.
        coefficients = nn.init.uniform_(
            tensor=torch.empty(
                [
                    self.mat_count,
                    self.components_count,
                ],
                dtype=self.weights.dtype_weights,
            ),
            a=-2.0,
            b=+2.0,
        )

        self.coefficients = nn.Parameter(coefficients)

        pass

    def forward(
        self,
    ) -> torch.Tensor:
        return self.weights(self.coefficients)


def generate_data_deviative(
    shape: list[int],
    mat_count: int,
    mat_deviation: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = torch.nn.init.uniform_(
        tensor=torch.empty([1, *shape]),
        a=-1.0,
        b=+1.0,
    )
    mods = (
        torch.nn.init.uniform_(
            tensor=torch.empty([mat_count, *shape]),
            a=-1.0,
            b=+1.0,
        )
        * mat_deviation
    )
    return (base.expand_as(mods) + mods).to(dtype)


def generate_data_random(
    shape: list[int],
    mat_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = torch.nn.init.uniform_(
        tensor=torch.empty([mat_count, *shape]),
        a=-1.0,
        b=+1.0,
    ).to(dtype)
    return base


def generate_data_from_images(
    shape: list[int],
    images_path_src: str,
    mat_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    data = torch.empty([mat_count, *shape], dtype=dtype)
    dir_contents = os.listdir(images_path_src)

    for i in range(mat_count):
        image_name = dir_contents[i]
        image_path = os.path.join(images_path_src, image_name)
        image = Image.open(image_path)
        image = image.resize([shape[1], shape[0]], Image.LANCZOS)
        image = transforms.ToTensor()(image).mean(dim=0, keepdim=False)
        image = image.unsqueeze(0)
        image = (image - image.min()) / (image.max() - image.min())
        image = (image - 0.5) * 2.0
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
        image = (image - image.min()) / (image.max() - image.min())
        image = transforms.ToPILImage()(image)
        image_name = f"{prefix}_mat_{i}.png"
        image_path = os.path.join(images_path_dst, image_name)
        image.save(image_path)
    pass


def sample_results(
    target: torch.Tensor,
    output: torch.Tensor,
    count_samples: int,
) -> None:
    target = target.to(dtype=torch.float16)
    output = output.to(dtype=torch.float16)
    for i in range(count_samples):
        print(f"\nMat #{i} samples:")
        print(f"Target: {target[i, 0, 0:16].cpu().detach().numpy().tolist()}")
        print(f"Output: {output[i, 0, 0:16].cpu().detach().numpy().tolist()}")


def train(
    data: torch.Tensor,
    model: Model,
    optimizer: torch.optim.Optimizer,
    iterations: int,
    log_nth_iteration: int,
    results_sample_count: int,
    mode: str = "matrix",
    images_path_dst: str = None,
) -> None:
    preheat_output = model()  # Preheat.
    params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_data = data.numel()

    print("\n# --------------------------------------------------- #\n")
    print(f"Model parameters: {params_model}")
    print(f"Data parameters: {params_data}")
    print(f"Ratio: {params_model / params_data}")
    print(f"Model weights data types:")
    print(f"{model.weights.weights_base.dtype=}")
    print(f"{model.weights.mod_i.dtype=}")
    print(f"{model.weights.mod_j.dtype=}")
    print(f"Model output data type: {preheat_output.dtype}")

    print("\n# --------------------------------------------------- #\n")
    for i in range(iterations):
        optimizer.zero_grad()
        output = model()
        loss = (data - output).std().sqrt()
        loss.backward()
        optimizer.step()

        if (i + 1) % log_nth_iteration == 0:
            print(
                f"Iteration #{i+1}: \nLoss: {loss.item()}\nStdR: {(data - output).std()}"
            )
            if mode == "images":
                generate_images_from_data(
                    data=output,
                    images_path_dst=images_path_dst,
                    prefix=f"output_i{i+1}",
                )

    print("\n# --------------------------------------------------- #\n")
    sample_results(
        target=data,
        output=output,
        count_samples=min(model.mat_count, results_sample_count),
    )

    if mode == "images":
        generate_images_from_data(
            data=output,
            images_path_dst=images_path_dst,
            prefix=f"output_final_i{i+1}",
        )

    pass


def main():
    global model

    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="matrix",
        choices=["matrix", "images"],
        help="mode (default: matrix)",
    )
    parser.add_argument(
        "--images-path-src",
        type=str,
        default=None,
        help="path to source images (default: None)",
    )
    parser.add_argument(
        "--images-path-dst",
        type=str,
        default=None,
        help="path to reconstructed images (default: None)",
    )
    parser.add_argument(
        "--output-shape",
        nargs=2,
        type=int,
        default=[128, 128],
        help="dimensionality of test matrices (default: [128, 128])",
    )
    parser.add_argument(
        "--mat-count",
        type=int,
        default=16,
        help="count of test matrices (default: 16)",
    )
    parser.add_argument(
        "--components-count",
        type=int,
        default=16,
        help="count of components for dynamic weights (default: 16)",
    )
    parser.add_argument(
        "--random-data",
        default=False,
        action="store_true",
        help="use random data, instead of deviative (default: False)",
    )
    parser.add_argument(
        "--mat-deviation",
        type=float,
        default=0.1,
        help="standard deviation of test matrices (default: 0.1)",
    )
    parser.add_argument(
        "--mod-rank",
        type=int,
        default=16,
        help="mod rank of the library matrices (default: 16)",
    )
    parser.add_argument(
        "--asymmetry",
        type=float,
        default=1e-3,
        help="asymmetry (default: 1e-3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5_000,
        help="iterations (default: 5_000)",
    )
    parser.add_argument(
        "--log-nth-iteration",
        type=int,
        default=100,
        help="how many batches to wait before logging training status (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--results-sample-count",
        type=int,
        default=4,
        help="how many results to sample (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-3,
        help="learning rate (default: 1.0e-3)",
    )
    args = parser.parse_args()

    if args.mode == "images":
        assert (
            args.images_path_src is not None
        ), "Path to source images must be specified."
        assert (
            args.images_path_dst is not None
        ), "Path to reconstructed images must be specified."
        assert os.path.isdir(
            args.images_path_src
        ), "Path to source images must be a directory."
        assert os.path.isdir(
            args.images_path_dst
        ), "Path to reconstructed images must be a directory."

    print("\n# --------------------------------------------------- #\n")
    print(f"Running with arguments:")
    print(" ".join(f"\t{k}={v}\n" for k, v in vars(args).items()))

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    if args.dtype == "bfloat16":
        dtype_weights = torch.bfloat16
    elif args.dtype == "float16":
        dtype_weights = torch.float16
    elif args.dtype == "float32":
        dtype_weights = torch.float32
    elif args.dtype == "float64":
        dtype_weights = torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    if args.mode == "matrix":
        data = (
            generate_data_deviative(
                shape=args.output_shape,
                mat_count=args.mat_count,
                mat_deviation=args.mat_deviation,
                dtype=dtype_weights,
            ).to(device)
            if not args.random_data
            else generate_data_random(
                shape=args.output_shape,
                mat_count=args.mat_count,
                dtype=dtype_weights,
            ).to(device)
        )
    elif args.mode == "images":
        data = generate_data_from_images(
            shape=args.output_shape,
            images_path_src=args.images_path_src,
            mat_count=args.mat_count,
            dtype=dtype_weights,
        ).to(device)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print("\n# --------------------------------------------------- #\n")
    print("Generated data specs:")
    print(f"{data.min()=}")
    print(f"{data.max()=}")
    print(f"{data.mean()=}")
    print(f"{data.std()=}")

    model = Model(
        output_shape=args.output_shape,
        mat_count=args.mat_count,
        components_count=args.components_count,
        mod_rank=args.mod_rank,
        asymmetry=args.asymmetry,
        dtype_weights=dtype_weights,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train(
        data=data,
        model=model,
        optimizer=optimizer,
        iterations=args.iterations,
        log_nth_iteration=args.log_nth_iteration,
        results_sample_count=args.results_sample_count,
        mode=args.mode,
        images_path_dst=args.images_path_dst,
    )


if __name__ == "__main__":
    main()
