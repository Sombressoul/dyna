import os
import sys
import argparse
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
evals_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(evals_dir)
sys.path.append(project_dir)

from dyna import WeightsLib2D


model = None


class Model(nn.Module):
    def __init__(
        self,
        shape: list[int] = [128, 128],
        rank_mod: int = 8,
        rank_deltas: int = 4,
        complex: bool = True,
        complex_output: bool = True,
        use_exponentiation: bool = True,
        trainable_exponents_base: bool = True,
        trainable_exponents_mod: bool = True,
        trainable_exponents_deltas: bool = True,
        exponents_initial_value_real: float = 2.0,
        exponents_initial_value_imag: float = 0.0,
        asymmetry: float = 1e-2,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.shape = shape

        self.weights = WeightsLib2D(
            shape=shape,
            rank_mod=rank_mod,
            rank_deltas=rank_deltas,
            complex=complex,
            complex_output=complex_output,
            use_exponentiation=use_exponentiation,
            trainable_exponents_base=trainable_exponents_base,
            trainable_exponents_mod=trainable_exponents_mod,
            trainable_exponents_deltas=trainable_exponents_deltas,
            exponents_initial_value_real=exponents_initial_value_real,
            exponents_initial_value_imag=exponents_initial_value_imag,
            asymmetry=asymmetry,
            dtype=dtype,
        )

        pass

    def forward(
        self,
        mat_count: int,
    ) -> torch.Tensor:
        mat_names = [f"mat_{i}" for i in range(mat_count)]
        mats = self.weights.get_weights(mat_names)

        return mats


def generate_data(
    shape: list[int],
    mat_count: int,
    mat_deviation: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = torch.randn(1, *shape)
    mods = torch.randn(mat_count, *shape) * mat_deviation
    return (base.expand_as(mods) + mods).to(dtype)


def train(
    data: torch.Tensor,
    model: Model,
    optimizer: torch.optim.Optimizer,
    iterations: int,
    log_nth_iteration: int,
    mat_count: int,
) -> None:
    preheat_output = model(mat_count=mat_count)  # Preheat.
    params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_data = data.numel()

    print(f"Model parameters: {params_model}")
    print(f"Data parameters: {params_data}")
    print(f"Ratio: {params_model / params_data}")
    print(f"Model weights data types:")
    print(f"{model.weights.weights_base.dtype=}")
    print(f"{model.weights.weights_mod_i.dtype=}")
    print(f"{model.weights.weights_mod_j.dtype=}")
    print(f"Model output data type: {preheat_output.dtype}")

    for i in range(iterations):
        optimizer.zero_grad()
        output = model(mat_count=mat_count)
        loss = (data - output).std()
        loss.backward()
        optimizer.step()

        if (i + 1) % log_nth_iteration == 0:
            print(f"Iteration #{i+1:<10d}: Loss: {loss.item()}")

    print(f"{data[0, 0, 0:16]=}")
    print(f"{output[0, 0, 0:16]=}")

    params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_data = data.numel()

    pass


def main():
    global model

    parser = argparse.ArgumentParser(description="evaluation")
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
        "--lib-rank-delta",
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
        "--complex-output",
        default=False,
        action="store_true",
        help="use complex numbers for the output (default: False)",
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
        default="float32",
        help="dtype (default: float32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed (default: 1)",
    )
    args = parser.parse_args()

    print(f"Running with arguments:")
    print(' '.join(f'\t{k}={v}\n' for k, v in vars(args).items()))

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

    data = generate_data(
        shape=args.mat_shape,
        mat_count=args.mat_count,
        mat_deviation=args.mat_deviation,
        dtype=dtype,
    ).to(device)
    model = Model(
        shape=args.mat_shape,
        rank_mod=args.lib_rank_mod,
        rank_deltas=args.lib_rank_delta,
        complex=not args.no_complex,
        complex_output=args.complex_output,
        use_exponentiation=args.use_exponentiation,
        trainable_exponents_base=not args.no_trainable_exponents_base,
        trainable_exponents_mod=not args.no_trainable_exponents_mod,
        trainable_exponents_deltas=not args.no_trainable_exponents_deltas,
        exponents_initial_value_real=args.exponents_initial_value_real,
        exponents_initial_value_imag=args.exponents_initial_value_imag,
        asymmetry=args.asymmetry,
        dtype=dtype,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    train(
        data=data,
        model=model,
        optimizer=optimizer,
        iterations=args.iterations,
        log_nth_iteration=args.log_nth_iteration,
        mat_count=args.mat_count,
    )


if __name__ == "__main__":
    main()
