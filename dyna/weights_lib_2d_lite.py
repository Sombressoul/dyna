import torch
import torch.nn as nn
import math

from typing import Union, List


class WeightsLib2DLite(nn.Module):
    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],
        components_count: int = 16,
        mod_rank: int = 16,
        asymmetry: float = 1e-3,
        dtype_weights: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        assert len(output_shape) == 2, "Shape must be 2D."
        assert components_count > 1, "Components must be greater than 1."
        assert mod_rank > 1, "Rank must be greater than 1."

        # ================================================================================= #
        # ____________________________> Parameters.
        # ================================================================================= #
        self.output_shape = output_shape
        self.count_components = components_count
        self.mod_rank = mod_rank
        self.asymmetry = asymmetry
        self.dtype_weights = dtype_weights

        # ================================================================================= #
        # ____________________________> Weights.
        # ================================================================================= #
        # Init: weights_i
        self.weights_i = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [1, 1, self.output_shape[0], self.output_shape[0], 2],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=1.0 / math.log(self.output_shape[0] ** 2, math.e),
            ).contiguous(),
        )
        # Init: weights_j
        self.weights_j = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [1, 1, self.output_shape[1], self.output_shape[1], 2],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=1.0 / math.log(self.output_shape[1] ** 2, math.e),
            ).contiguous(),
        )
        # Init: weights_base
        self.weights_base = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [1, 1, *self.output_shape, 2],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=1.0 / math.log(math.prod(self.output_shape), math.e),
            ).contiguous(),
        )
        # Init: bias
        self.translate_base = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [1, self.count_components, 1, 1, 2],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=self.asymmetry,
            ).contiguous(),
        )
        # Init: scale
        self.rotate_base = nn.Parameter(
            data=torch.cat(
                [
                    torch.nn.init.normal_(
                        tensor=torch.empty(
                            [1, self.count_components, 1, 1, 1],
                            dtype=self.dtype_weights,
                        ),
                        mean=1.0,
                        std=self.asymmetry,
                    ),
                    torch.nn.init.normal_(
                        tensor=torch.empty(
                            [1, self.count_components, 1, 1, 1],
                            dtype=self.dtype_weights,
                        ),
                        mean=0.0,
                        std=self.asymmetry,
                    ),
                ],
                dim=-1,
            ).contiguous(),
        )
        # Init: bias
        self.translate_dynamic = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [1, self.count_components, 1, 1, 2],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=self.asymmetry,
            ).contiguous(),
        )
        # Init: scale
        self.rotate_dynamic = nn.Parameter(
            data=torch.cat(
                [
                    torch.nn.init.normal_(
                        tensor=torch.empty(
                            [1, self.count_components, 1, 1, 1],
                            dtype=self.dtype_weights,
                        ),
                        mean=1.0,
                        std=self.asymmetry,
                    ),
                    torch.nn.init.normal_(
                        tensor=torch.empty(
                            [1, self.count_components, 1, 1, 1],
                            dtype=self.dtype_weights,
                        ),
                        mean=0.0,
                        std=self.asymmetry,
                    ),
                ],
                dim=-1,
            ).contiguous(),
        )
        # Init: mod_i
        self.mod_i = nn.Parameter(
            data=torch.cat(
                [
                    torch.nn.init.uniform_(
                        tensor=torch.empty(
                            [
                                1,
                                self.count_components,
                                self.mod_rank,
                                self.output_shape[0],
                                1,
                            ],
                            dtype=self.dtype_weights,
                        ),
                        a=-(
                            1.0 / math.log(self.mod_rank * self.output_shape[0], math.e)
                        ),
                        b=+(
                            1.0 / math.log(self.mod_rank * self.output_shape[0], math.e)
                        ),
                    ),
                    torch.nn.init.normal_(
                        tensor=torch.empty(
                            [
                                1,
                                self.count_components,
                                self.mod_rank,
                                self.output_shape[0],
                                1,
                            ],
                            dtype=self.dtype_weights,
                        ),
                        mean=0.0,
                        std=self.asymmetry,
                    ),
                ],
                dim=-1,
            ).contiguous(),
        )
        # Init: mod_j
        self.mod_j = nn.Parameter(
            data=torch.cat(
                [
                    torch.nn.init.uniform_(
                        tensor=torch.empty(
                            [
                                1,
                                self.count_components,
                                self.mod_rank,
                                self.output_shape[1],
                                1,
                            ],
                            dtype=self.dtype_weights,
                        ),
                        a=-(self.mod_rank * self.output_shape[1]),
                        b=+(self.mod_rank * self.output_shape[1]),
                    ),
                    torch.nn.init.normal_(
                        tensor=torch.empty(
                            [
                                1,
                                self.count_components,
                                self.mod_rank,
                                self.output_shape[1],
                                1,
                            ],
                            dtype=self.dtype_weights,
                        ),
                        mean=0.0,
                        std=math.log(self.mod_rank * self.output_shape[1], math.e),
                    ),
                ],
                dim=-1,
            ).contiguous(),
        )
        # Init: inversions
        self.inversions = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [1, self.count_components, 1, 1, 2],
                    dtype=self.dtype_weights,
                ),
                mean=-1.0,
                std=self.asymmetry,
            ).contiguous(),
        )
        # Init: mod_i_transforms
        self.mod_transforms = nn.Parameter(
            data=torch.nn.init.xavier_uniform_(
                tensor=torch.empty(
                    [4, self.count_components, self.count_components * 2],
                    dtype=self.dtype_weights,
                ),
            ).contiguous(),
        )
        torch.nn.init._calculate_correct_fan

        pass

    def _log_var(
        self,
        x: torch.Tensor,
        name: str,
        is_breakpoint: bool = False,
    ) -> None:
        mem = (x.element_size() * x.nelement()) / (1024 * 1024)
        unk = "<UNKNOWN>"

        print(f"\n")
        print(f"# =====> Name: {name if name is not None else unk}")
        print(f"# =====> Memory: {mem:.2f} MB")
        print(f"# =====> Elements: {x.numel():_}")

        if x.shape[-1] == 2:
            real = x[..., 0]
            imag = x[..., 1]
            abs = torch.sqrt(real**2 + imag**2)

            print(f"{x.shape=}")
            print(f"{real.min()=}")
            print(f"{real.max()=}")
            print(f"{real.mean()=}")
            print(f"{real.std()=}")
            print(f"{imag.min()=}")
            print(f"{imag.max()=}")
            print(f"{imag.mean()=}")
            print(f"{imag.std()=}")
            print(f"{abs.min()=}")
            print(f"{abs.max()=}")
            print(f"{abs.mean()=}")
            print(f"{abs.std()=}")
        else:
            print(f"{x.shape=}")
            print(f"{x.min()=}")
            print(f"{x.max()=}")
            print(f"{x.mean()=}")
            print(f"{x.std()=}")

        print(f"\n")

        if is_breakpoint:
            exit()

        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        assert not torch.is_complex(x), f"Input must be real. Got: {x.dtype}."
        assert len(x.shape) == 2, f"Input must be 2D. Got: {x.shape}."
        assert x.shape[1] == self.count_components, " ".join(
            [
                f"Input shape should match the number of weights",
                f"components: [batch_dim, {self.count_components}]. Got: {x.shape}",
            ]
        )

        # ================================================================================= #
        # ____________________________> Casting.
        # ================================================================================= #
        x = x.to(dtype=self.dtype_weights)

        # ================================================================================= #
        # ____________________________> Notes.
        # ================================================================================= #
        #
        # Hints for complex: given z1 = {a, b} and z2 = {c, d},
        #   Addition:
        #       z1 + z2 = {(a + c), (b + d)}
        #   Subtraction:
        #       z1 - z2 = {(a - c), (b - d)}
        #   Multiplication (algebraic):
        #       z1 * z2 = {(a * c - b * d), (a * d + b * c)}
        #   Division (algebraic):
        #       denom = (c * c + d * d)
        #       z1 / z2 = {(a * c + b * d) / denom, (b * c - a * d) / denom}
        #
        # ================================================================================= #
        # ____________________________> Dynamic weights computation.
        # ================================================================================= #
        weights = self.weights_base.repeat([1, self.translate_base.shape[1], 1, 1, 1])
        weights = weights + self.translate_base.expand_as(weights)
        z = self.rotate_base.expand_as(weights)
        r = (weights * z).diff(dim=-1)
        i = (weights * z[..., [1, 0]]).sum(dim=-1, keepdim=True)
        weights = torch.cat([r, i], dim=-1)

        transforms = torch.einsum("ij,kjl -> ikl", x, self.mod_transforms).contiguous()
        transforms = transforms.view(
            [*transforms.shape[0:-1], transforms.shape[-1] // 2, 1, 1, 2]
        )  # [n, (i_translate|i_rotate|j_translate|j_rotate), self.components_count, 1, 1, 2]

        mod_i = self.mod_i.repeat(
            [transforms.shape[0], *[1 for _ in range(len(self.mod_i.shape) - 1)]]
        )
        mod_i = mod_i + transforms[::, 0, ...]
        z = transforms[::, 1, ...]
        r = (mod_i * z).diff(dim=-1)
        i = (mod_i * z[..., [1, 0]]).sum(dim=-1, keepdim=True)
        mod_i = torch.cat([r, i], dim=-1)

        mod_j = self.mod_j.repeat(
            [transforms.shape[0], *[1 for _ in range(len(self.mod_j.shape) - 1)]]
        )
        mod_j = mod_j + transforms[::, 2, ...]
        z = transforms[::, 3, ...]
        r = (mod_j * z).diff(dim=-1)
        i = (mod_j * z[..., [1, 0]]).sum(dim=-1, keepdim=True)
        mod_j = torch.cat([r, i], dim=-1)

        r = (mod_i * self.inversions).diff(dim=-1)
        i = (mod_i * self.inversions[..., [1, 0]]).sum(dim=-1, keepdim=True)
        mod_i = torch.cat([r, i], dim=-1).contiguous()
        mod_i = mod_i.sum(dim=1, keepdim=True)

        denom = (mod_j * mod_j).sum(dim=-1, keepdim=True)
        r = (self.inversions * mod_j).sum(dim=-1, keepdim=True) / denom
        i = (self.inversions[..., [1, 0]] * mod_j).diff(dim=-1) / denom
        mod_j = torch.cat([r, i], dim=-1).contiguous()
        mod_j = mod_j.sum(dim=1, keepdim=True)

        A = mod_i.permute([0, 1, 3, 2, 4])
        A = A.unsqueeze(-2)
        A = torch.cat([A, A], dim=-2)
        A[..., 0, 1] = -A[..., 0, 1]  # [[a, -b], [a, b]]
        B = mod_j.unsqueeze(-2)
        B = torch.cat([B, B[..., [1, 0]]], dim=-2)  # [[c, d], [d, c]]
        mod = torch.einsum("...ijlm,...jkml ->...ikl", A, B)

        if torch.isnan(mod).any() or torch.isinf(mod).any():
            raise ValueError("mod has NaN or Inf elements.")

        mod_r = mod[..., 0]
        r_num = mod_r - mod_r.mean(dim=[-1, -2], keepdim=True)
        r_denom = mod_r.std(dim=[-1, -2], keepdim=True)
        r = (r_num / r_denom).unsqueeze(-1)
        i = mod[..., 1].unsqueeze(-1)
        mod = torch.cat([r, i], dim=-1)
        weights_dynamic = self.weights_base + (self.weights_base * mod)

        A = weights_dynamic.permute([0, 1, 3, 2, 4])
        A = A.unsqueeze(-2)
        A = torch.cat([A, A], dim=-2)
        A[..., 0, 1] = -A[..., 0, 1]
        B = self.weights_i.unsqueeze(-2)
        B = torch.cat([B, B[..., [1, 0]]], dim=-2)
        weights_dynamic = torch.einsum("...ijlm,...jkml ->...ikl", A, B)

        A = weights_dynamic.permute([0, 1, 3, 2, 4])
        A = A.unsqueeze(-2)
        A = torch.cat([A, A], dim=-2)
        A[..., 0, 1] = -A[..., 0, 1]
        B = self.weights_j.unsqueeze(-2)
        B = torch.cat([B, B[..., [1, 0]]], dim=-2)
        weights_dynamic = torch.einsum("...ijlm,...jkml ->...ikl", A, B)

        weights_dynamic = weights_dynamic + self.translate_dynamic
        r = (weights_dynamic * self.rotate_dynamic).diff(dim=-1)
        i = (weights_dynamic * self.rotate_dynamic[..., [1, 0]]).sum(
            dim=-1, keepdim=True
        )
        weights_dynamic = torch.cat([r, i], dim=-1).sum(dim=1)[..., 0]  # Real only.

        return weights_dynamic
