import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from dyna.lib import WeightsLib2DDev


model = None


class Model(nn.Module):
    def __init__(
        self,
        shape: list[int] = [128, 128],
        count_weights_variations: int = 16,
        count_weights_components: int = 16,
        complex_components: bool = True,
        rank_mod: int = 8,
        use_deltas: bool = True,
        rank_deltas: int = 4,
        complex: bool = True,
        complex_output: bool = True,
        use_exponentiation: bool = True,
        trainable_exponents_base: bool = True,
        trainable_exponents_mod: bool = True,
        trainable_exponents_deltas: bool = True,
        exponents_initial_value_real: float = 2.0,
        exponents_initial_value_imag: float = 0.0,
        use_bias: bool = True,
        use_scale: bool = True,
        asymmetry: float = 1e-3,
        activation_type: str = "identity",
        transformation_type: str = "inversion",
        preserve_i: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        if complex_output:
            assert (
                complex_components
            ), "Complex output is only supported with complex components."

        # ================================================================================= #
        # ____________________________> Parameters.
        # ================================================================================= #
        self.count_weights_variations = count_weights_variations
        self.count_weights_components = count_weights_components
        self.preserve_i = preserve_i
        self.shape = shape

        # ================================================================================= #
        # ____________________________> Weights.
        # ================================================================================= #
        # Init WeightsLib2D.
        self.weights = WeightsLib2DDev(
            shape=shape,
            rank_mod=rank_mod,
            use_deltas=use_deltas,
            rank_deltas=rank_deltas,
            complex=complex,
            complex_output=complex_output,
            use_exponentiation=use_exponentiation,
            trainable_exponents_base=trainable_exponents_base,
            trainable_exponents_mod=trainable_exponents_mod,
            trainable_exponents_deltas=trainable_exponents_deltas,
            exponents_initial_value_real=exponents_initial_value_real,
            exponents_initial_value_imag=exponents_initial_value_imag,
            use_bias=use_bias,
            use_scale=use_scale,
            asymmetry=asymmetry,
            activation_type=activation_type,
            transformation_type=transformation_type,
            dtype=dtype,
        )

        # Init coefficients.
        coefficients_r = torch.empty(
            [
                self.count_weights_variations,
                self.count_weights_components,
                1,
                1,
            ],
            dtype=self.weights.dtype_real,
        )
        coefficients_r = nn.init.uniform_(
            tensor=coefficients_r,
            a=-2.0,
            b=+2.0,
        )

        if complex_components:
            coefficients_i = torch.empty_like(coefficients_r)
            coefficients_i = nn.init.normal_(
                tensor=coefficients_i,
                mean=0.0,
                std=self.weights.asymmetry,
            )
            coefficients = torch.complex(
                real=coefficients_r,
                imag=coefficients_i,
            ).to(
                dtype=self.weights.dtype_complex,
            )
        else:
            coefficients = coefficients_r

        self.coefficients = nn.Parameter(coefficients)

        pass

    def forward(
        self,
    ) -> torch.Tensor:
        components_names = [f"x_{i}" for i in range(self.count_weights_components)]
        components_weights = self.weights.get_weights(components_names).unsqueeze(0)
        weights = components_weights.mul(self.coefficients)
        weights = weights.sum(dim=1, keepdim=False)
        weights = weights if self.preserve_i else weights.real

        return weights


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
    print(f"{model.weights.weights_mod_i.dtype=}")
    print(f"{model.weights.weights_mod_j.dtype=}")
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
        count_samples=min(model.count_weights_variations, results_sample_count),
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
        "--mat-shape",
        nargs=2,
        type=int,
        default=[256, 256],
        help="dimensionality of test matrices (default: [256, 256])",
    )
    parser.add_argument(
        "--mat-count",
        type=int,
        default=16,
        help="count of test matrices (default: 16)",
    )
    parser.add_argument(
        "--weights-count-components",
        type=int,
        default=16,
        help="count of components for dynamic weights (default: 16)",
    )
    parser.add_argument(
        "--no-complex-components",
        default=False,
        action="store_true",
        help="do not use complex numbers for the dynamic weights (default: False)",
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
        "--lib-rank-mod",
        type=int,
        default=None,
        help="mod rank of the library matrices (default: None (auto-select))",
    )
    parser.add_argument(
        "--use-deltas",
        default=False,
        action="store_true",
        help="use deltas (default: False)",
    )
    parser.add_argument(
        "--lib-rank-deltas",
        type=int,
        default=16,
        help="deltas rank of the library matrices (default: 16)",
    )
    parser.add_argument(
        "--no-complex",
        default=False,
        action="store_true",
        help="do not use complex numbers (default: False)",
    )
    parser.add_argument(
        "--no-complex-output",
        default=False,
        action="store_true",
        help="do not use complex numbers for the model output (default: False)",
    )
    parser.add_argument(
        "--use-exponentiation",
        default=False,
        action="store_true",
        help="use exponentiation (default: False)",
    )
    parser.add_argument(
        "--no-trainable-exponents-base",
        default=False,
        action="store_true",
        help="do not train the base exponents (default: False)",
    )
    parser.add_argument(
        "--no-trainable-exponents-mod",
        default=False,
        action="store_true",
        help="do not train the mod exponents (default: False)",
    )
    parser.add_argument(
        "--no-trainable-exponents-deltas",
        default=False,
        action="store_true",
        help="do not train the deltas exponents (default: False)",
    )
    parser.add_argument(
        "--exponents-initial-value-real",
        type=float,
        default=1.0,
        help="initial value for the real part of the exponents (default: 1.0)",
    )
    parser.add_argument(
        "--exponents-initial-value-imag",
        type=float,
        default=0.0,
        help="initial value for the imaginary part of the exponents (default: 0.0)",
    )
    parser.add_argument(
        "--no-bias",
        default=False,
        action="store_true",
        help="do not use bias (default: False)",
    )
    parser.add_argument(
        "--no-scale",
        default=False,
        action="store_true",
        help="do not use scale (default: False)",
    )
    parser.add_argument(
        "--asymmetry",
        type=float,
        default=1e-3,
        help="asymmetry (default: 1e-3)",
    )
    parser.add_argument(
        "--activation-type",
        type=str,
        default="identity",
        choices=["identity", "cardioid"],
        help="internal activation type (default: identity)",
    )
    parser.add_argument(
        "--transformation-type",
        type=str,
        default="inversion",
        choices=["inversion", "translation"],
        help="transformation type (default: inversion)",
    )
    parser.add_argument(
        "--preserve-i",
        default=False,
        action="store_true",
        help="preserve imaginary part of the model output (default: False)",
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
        default="float32",
        help="dtype (default: float32)",
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
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float64":
        dtype = torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    if args.mode == "matrix":
        data = (
            generate_data_deviative(
                shape=args.mat_shape,
                mat_count=args.mat_count,
                mat_deviation=args.mat_deviation,
                dtype=dtype,
            ).to(device)
            if not args.random_data
            else generate_data_random(
                shape=args.mat_shape,
                mat_count=args.mat_count,
                dtype=dtype,
            ).to(device)
        )
    elif args.mode == "images":
        data = generate_data_from_images(
            shape=args.mat_shape,
            images_path_src=args.images_path_src,
            mat_count=args.mat_count,
            dtype=dtype,
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
        shape=args.mat_shape,
        count_weights_variations=args.mat_count,
        count_weights_components=args.weights_count_components,
        complex_components=not args.no_complex_components,
        rank_mod=args.lib_rank_mod,
        use_deltas=args.use_deltas,
        rank_deltas=args.lib_rank_deltas,
        complex=not args.no_complex,
        complex_output=not args.no_complex_output,
        use_exponentiation=args.use_exponentiation,
        trainable_exponents_base=not args.no_trainable_exponents_base,
        trainable_exponents_mod=not args.no_trainable_exponents_mod,
        trainable_exponents_deltas=not args.no_trainable_exponents_deltas,
        exponents_initial_value_real=args.exponents_initial_value_real,
        exponents_initial_value_imag=args.exponents_initial_value_imag,
        use_bias=not args.no_bias,
        use_scale=not args.no_scale,
        asymmetry=args.asymmetry,
        activation_type=args.activation_type,
        transformation_type=args.transformation_type,
        preserve_i=args.preserve_i,
        dtype=dtype,
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
