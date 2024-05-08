import torch
import torch.nn as nn

from typing import Union, List


class WeightsLib2DLite(nn.Module):
    def __init__(
        self,
        output_shape: Union[torch.Size, List[int]],
        components_count: int = 16,
        mod_rank: int = 16,
        asymmetry: float = 1e-3,
        eps: float = 1.0e-5,
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
        self.eps = eps
        self.dtype_weights = dtype_weights

        # ================================================================================= #
        # ____________________________> Weights.
        # ================================================================================= #
        # Init: weights_i
        self.weights_i = nn.Parameter(
            data=torch.nn.init.xavier_uniform_(
                tensor=torch.empty(
                    [1, 1, self.output_shape[0], self.output_shape[0], 2],
                    dtype=self.dtype_weights,
                ),
            ).contiguous(),
        )
        # Init: weights_j
        self.weights_j = nn.Parameter(
            data=torch.nn.init.xavier_uniform_(
                tensor=torch.empty(
                    [1, 1, self.output_shape[1], self.output_shape[1], 2],
                    dtype=self.dtype_weights,
                ),
            ).contiguous(),
        )
        # Init: weights_base
        self.weights_base = nn.Parameter(
            data=torch.nn.init.xavier_uniform_(
                tensor=torch.empty(
                    [1, 1, *self.output_shape, 2],
                    dtype=self.dtype_weights,
                ),
            ).contiguous(),
        )
        # Init: bias
        self.translate_dynamic = nn.Parameter(
            data=torch.nn.init.uniform_(
                tensor=torch.empty(
                    [1, 1, 1, 2],
                    dtype=self.dtype_weights,
                ),
                a=-1.0,
                b=+1.0,
            ).contiguous(),
        )
        # Init: scale
        self.rotate_dynamic = nn.Parameter(
            data=torch.cat(
                [
                    torch.nn.init.uniform_(
                        tensor=torch.empty(
                            [1, 1, 1, 1],
                            dtype=self.dtype_weights,
                        ),
                        a=-1.0,
                        b=+1.0,
                    ),
                    torch.nn.init.uniform_(
                        tensor=torch.empty(
                            [1, 1, 1, 1],
                            dtype=self.dtype_weights,
                        ),
                        a=-1.0,
                        b=+1.0,
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
                        a=-1.0,
                        b=+1.0,
                    ),
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
                        a=-1.0,
                        b=+1.0,
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
                        a=-1.0,
                        b=+1.0,
                    ),
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
                        a=-1.0,
                        b=+1.0,
                    ),
                ],
                dim=-1,
            ).contiguous(),
        )
        # Init: inversions
        self.inversions = nn.Parameter(
            data=torch.nn.init.uniform_(
                tensor=torch.empty(
                    [1, self.count_components, 1, 1, 2],
                    dtype=self.dtype_weights,
                ),
                a=-1.0,
                b=+1.0,
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
        # Init: projections
        self.projections = nn.Parameter(
            data=torch.nn.init.normal_(
                tensor=torch.empty(
                    [1, *self.output_shape, 2],
                    dtype=self.dtype_weights,
                ),
                mean=0.0,
                std=self.asymmetry,
            ).contiguous(),
        )

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
            print(f"{real.abs().min()=}")
            print(f"{real.abs().max()=}")
            print(f"{real.abs().mean()=}")
            print(f"{real.abs().std()=}")
            print(f"{imag.min()=}")
            print(f"{imag.max()=}")
            print(f"{imag.mean()=}")
            print(f"{imag.std()=}")
            print(f"{imag.abs().min()=}")
            print(f"{imag.abs().max()=}")
            print(f"{imag.abs().mean()=}")
            print(f"{imag.abs().std()=}")
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
            print(f"{x.abs().min()=}")
            print(f"{x.abs().max()=}")
            print(f"{x.abs().mean()=}")
            print(f"{x.abs().std()=}")

        print(f"\n")

        if is_breakpoint:
            exit()

        pass

    def norm_polar(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_abs = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        a = torch.atan2(x[..., 1], x[..., 0])
        h = x_abs / x_abs.max()  # normalize per whole batch
        r = (h * a.cos()).unsqueeze(-1)
        i = (h * a.sin()).unsqueeze(-1)
        x = torch.cat([r, i], dim=-1)
        return x

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
        transforms = torch.einsum("ij,kjl -> ikl", x, self.mod_transforms).contiguous()
        transforms = transforms.view(
            [*transforms.shape[0:-1], transforms.shape[-1] // 2, 1, 1, 2]
        )  # [n, (i_translate|i_rotate|j_translate|j_rotate), self.components_count, 1, 1, 2]

        mod_i = self.mod_i.repeat(
            [
                transforms.shape[0],
                *[1 for _ in range(len(self.mod_i.shape) - 1)],
            ]
        )
        mod_i = mod_i + transforms[::, 0, ...]
        z = transforms[::, 1, ...]
        r = (mod_i * z).diff(dim=-1)
        i = (mod_i * z[..., [1, 0]]).sum(dim=-1, keepdim=True)
        mod_i = torch.cat([r, i], dim=-1)

        mod_j = self.mod_j.repeat(
            [
                transforms.shape[0],
                *[1 for _ in range(len(self.mod_j.shape) - 1)],
            ]
        )
        mod_j = mod_j + transforms[::, 2, ...]
        z = transforms[::, 3, ...]
        r = (mod_j * z).diff(dim=-1)
        i = (mod_j * z[..., [1, 0]]).sum(dim=-1, keepdim=True)
        mod_j = torch.cat([r, i], dim=-1)

        r = (mod_i * self.inversions).diff(dim=-1)
        i = (mod_i * self.inversions[..., [1, 0]]).sum(dim=-1, keepdim=True)
        mod_i = torch.cat([r, i], dim=-1)
        mod_i = mod_i.sum(dim=1, keepdim=True).contiguous()

        denom = (mod_j * mod_j).sum(dim=-1, keepdim=True)
        # avoiding division by zero ->
        denom[torch.where(denom < self.asymmetry)] = self.asymmetry
        # <- avoiding division by zero
        r = (self.inversions * mod_j).sum(dim=-1, keepdim=True).mul(1.0 / denom)
        i = (self.inversions[..., [1, 0]] * mod_j).diff(dim=-1).mul(1.0 / denom)
        mod_j = torch.cat([r, i], dim=-1)
        mod_j = mod_j.sum(dim=1, keepdim=True).contiguous()

        A = mod_i.permute([0, 1, 3, 2, 4])
        A = A.unsqueeze(-2)
        A = torch.cat([A, A], dim=-2)
        A[..., 0, 1] = -A[..., 0, 1]  # [[a, -b], [a, b]]
        B = mod_j.unsqueeze(-2)
        B = torch.cat([B, B[..., [1, 0]]], dim=-2)  # [[c, d], [d, c]]
        mod = torch.einsum("...ijlm,...jkml ->...ikl", A, B)
        mod = self.norm_polar(mod)

        if torch.isnan(mod).any() or torch.isinf(mod).any():
            self._log_var(self.weights_base, "self.weights_base", False)
            self._log_var(self.mod_transforms, "self.mod_transforms", False)
            self._log_var(self.mod_i, "self.mod_i", False)
            self._log_var(self.mod_j, "self.mod_j", False)
            self._log_var(self.inversions, "self.inversions", False)
            raise ValueError("mod has NaN or Inf elements.")

        mod_r = mod[..., 0]
        r_num = mod_r - mod_r.mean(dim=[-1, -2], keepdim=True)
        r_denom = mod_r.std(dim=[-1, -2], keepdim=True)
        r = (r_num / r_denom).unsqueeze(-1)
        i = mod[..., 1].unsqueeze(-1)
        mod = torch.cat([r, i], dim=-1)

        r = (self.weights_base * mod).diff(dim=-1)
        i = (self.weights_base * mod[..., [1, 0]]).sum(dim=-1, keepdim=True)
        weights_dynamic = torch.cat([r, i], dim=-1)

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

        weights_dynamic = weights_dynamic.squeeze(1)
        weights_dynamic = weights_dynamic + self.translate_dynamic
        r = weights_dynamic * self.rotate_dynamic
        r = r.diff(dim=-1)
        i = weights_dynamic * self.rotate_dynamic[..., [1, 0]]
        i = i.sum(dim=-1, keepdim=True)
        weights_dynamic = torch.cat([r, i], dim=-1)

        projection_denom = torch.sqrt(self.projections[..., 0]**2 + self.projections[..., 1]**2 + self.eps)
        theta_cos = self.projections[..., 0] / projection_denom
        theta_sin = self.projections[..., 1] / projection_denom
        weights_dynamic = weights_dynamic[..., 0] * theta_cos + weights_dynamic[..., 1] * theta_sin

        return weights_dynamic
