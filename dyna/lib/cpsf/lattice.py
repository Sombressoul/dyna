import torch

from typing import Sequence, Optional, Union

from dyna.lib.cpsf.structures import CPSFLatticeSumPolicyKind


class CPSFLattice:
    def __init__(
        self,
        kind: CPSFLatticeSumPolicyKind,
        window: Optional[Union[int, Sequence[torch.LongTensor]]],
    ):
        self.kind = kind
        self.window = window

    def fixed_window(
        self,
        N: int,
    ) -> torch.LongTensor:
        if self.kind != CPSFLatticeSumPolicyKind.WINDOW:
            raise NotImplementedError(
                'CPSFLatticeSumPolicyKind.FULL is not supported yet'
            )

        W = self.window
        if isinstance(W, int):
            if W < 0:
                raise ValueError("window radius must be non-negative")
            if N <= 0:
                raise ValueError("N must be positive")

            axis = torch.arange(-W, W + 1, dtype=torch.long)
            grids = [axis] * N
            if N == 1:
                lattice = axis.unsqueeze(1)
            else:
                lattice = torch.cartesian_prod(*grids)
            return lattice.to(dtype=torch.long)

        if W is None:
            return torch.zeros((1, N), dtype=torch.long)

        if not isinstance(W, (list, tuple)):
            raise TypeError(
                "window must be either int, None, or a sequence of LongTensors"
            )

        parts = []
        for t in W:
            if not isinstance(t, torch.Tensor) or t.dtype != torch.long:
                raise TypeError("each element of window sequence must be a LongTensor")
            if t.dim() == 1:
                if t.numel() != N:
                    raise ValueError(
                        f"1D window tensor must have length N={N}, got {t.numel()}"
                    )
                parts.append(t.view(1, N))
            elif t.dim() == 2:
                if t.shape[1] != N:
                    raise ValueError(
                        f"2D window tensor must have shape [K, N] with N={N}, got {tuple(t.shape)}"
                    )
                parts.append(t)
            else:
                raise ValueError("window tensors must be 1D or 2D (…, N)")

        if not parts:
            return torch.zeros((1, N), dtype=torch.long)

        lattice = torch.cat(parts, dim=0).to(dtype=torch.long)

        return lattice
