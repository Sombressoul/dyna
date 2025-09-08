import torch
import math

from typing import Optional, Union

from dyna.lib.cpsf.structures import (
    CPSFPeriodizationBackend,
    CPSFPeriodizationKind,
)


class CPSFPeriodization:
    def __init__(
        self,
        kind: CPSFPeriodizationKind,
        window: Optional[Union[int, torch.LongTensor]] = None,
        tolerance: Optional[Union[float, torch.FloatTensor]] = None,
        max_radius: Optional[Union[int, torch.LongTensor]] = None,
        backend: CPSFPeriodizationBackend = CPSFPeriodizationBackend.AUTO,
    ):
        # Enum guards
        if not isinstance(kind, CPSFPeriodizationKind):
            raise TypeError(
                f"CPSFPeriodization: 'kind' must be CPSFPeriodizationKind, got {type(kind)}"
            )
        if not isinstance(backend, CPSFPeriodizationBackend):
            raise TypeError(
                f"CPSFPeriodization: 'backend' must be CPSFPeriodizationBackend, got {type(backend)}"
            )

        # Helpers
        def _raise_tinfo(x, name, target_type):
            if isinstance(x, torch.Tensor):
                tinfo = f"type={type(x)}, dtype={x.dtype}, shape={tuple(x.shape)}, numel={x.numel()}"
            else:
                tinfo = f"type={type(x)}"
            raise TypeError(
                "\n".join(
                    [
                        f"CPSFPeriodization: '{name}' must be '{target_type}' scalar.",
                        f"Got: {tinfo}",
                    ]
                )
            )

        def _to_int_scalar(x, name):
            if x is None:
                return None
            if isinstance(x, int) and not isinstance(x, bool):
                return int(x)
            if (
                isinstance(x, torch.Tensor)
                and x.dtype != torch.bool
                and x.numel() == 1
                and not torch.is_floating_point(x)
                and not torch.is_complex(x)
            ):
                return int(x.item())
            _raise_tinfo(x, name, int)

        def _to_float_scalar(x, name):
            if x is None:
                return None
            if isinstance(x, (float, int)) and not isinstance(x, bool):
                return float(x)
            if (
                isinstance(x, torch.Tensor)
                and x.dtype != torch.bool
                and x.numel() == 1
                and torch.is_floating_point(x)
                and not torch.is_complex(x)
            ):
                return float(x.item())
            _raise_tinfo(x, name, float)

        # Pre-normalize arguments
        normal_window = _to_int_scalar(window, "window")
        normal_tolerance = _to_float_scalar(tolerance, "tolerance")
        normal_max_radius = _to_int_scalar(max_radius, "max_radius")

        # Ensure finite tolerance if provided
        if normal_tolerance is not None and not math.isfinite(normal_tolerance):
            raise ValueError("CPSFPeriodization: 'tolerance' must be finite")

        # Per-kind validation
        if kind == CPSFPeriodizationKind.WINDOW:
            if normal_window is None:
                raise ValueError("CPSFPeriodization(WINDOW): 'window' is required")
            if normal_window < 1:
                raise ValueError("CPSFPeriodization(WINDOW): 'window' must be >= 1")
            if normal_tolerance is not None or normal_max_radius is not None:
                raise ValueError(
                    "CPSFPeriodization(WINDOW): 'tolerance' and 'max_radius' must be None"
                )
            if backend != CPSFPeriodizationBackend.AUTO:
                raise ValueError("CPSFPeriodization(WINDOW): 'backend' must be AUTO")
        elif kind == CPSFPeriodizationKind.FULL:
            if normal_window is not None:
                raise ValueError("CPSFPeriodization(FULL): 'window' must be None")
            if normal_tolerance is None and normal_max_radius is None:
                raise ValueError(
                    "CPSFPeriodization(FULL): either 'tolerance' or 'max_radius' must be provided"
                )
            if normal_tolerance is not None and not (normal_tolerance > 0.0):
                raise ValueError("CPSFPeriodization(FULL): 'tolerance' must be > 0")
            if normal_max_radius is not None and normal_max_radius < 1:
                raise ValueError("CPSFPeriodization(FULL): 'max_radius' must be >= 1")
        else:
            raise ValueError(f"CPSFPeriodization: unsupported kind={kind}")

        # Assign values
        self.window = normal_window
        self.tolerance = normal_tolerance
        self.max_radius = normal_max_radius
        self.kind = kind
        self.backend = backend

    def _cartesian_window_points(
        self,
        N: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        if W < 0:
            raise ValueError(f"_cartesian_window_points: W must be >= 0, got {W}")
        if N < 2:
            raise ValueError(f"_cartesian_window_points: N must be >= 2, got {N}")
        if W == 0:
            return torch.zeros(1, N, dtype=torch.long, device=device)

        axes = [
            torch.arange(-W, W + 1, device=device, dtype=torch.long) for _ in range(N)
        ]

        return torch.cartesian_prod(*axes)

    def _shell_points(
        self,
        N: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        if W < 0:
            raise ValueError(f"_shell_points: W must be >= 0, got {W}")
        if N < 2:
            raise ValueError(f"_shell_points: N must be >= 2, got {N}")
        if W == 0:
            return torch.zeros(1, N, dtype=torch.long, device=device)

        axes = [
            torch.arange(-W, W + 1, device=device, dtype=torch.long) for _ in range(N)
        ]

        grid = torch.cartesian_prod(*axes)
        mask = grid.abs().amax(dim=-1) == W

        return grid[mask]

    def iter_offsets(
        self,
        N: int,
        device: torch.device,
    ):
        if N < 2:
            raise ValueError(f"CPSFPeriodization.iter_offsets: N must be >= 2, got {N}")

        if self.kind.name == "WINDOW":
            yield self._cartesian_window_points(N, self.window, device=device)
            return

        if self.kind.name == "FULL":
            if self.backend == CPSFPeriodizationBackend.DUAL:
                raise NotImplementedError(
                    "FULL with backend=DUAL is not implemented yet"
                )

            W = 0
            while True:
                yield self._shell_points(N, W, device=device)
                W += 1
                if self.max_radius is not None and W > self.max_radius:
                    break
            return

        raise ValueError(
            f"CPSFPeriodization.iter_offsets: unsupported kind={self.kind}"
        )
