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
                f"CPSFPeriodizationPolicy: 'kind' must be CPSFPeriodizationPolicyKind, got {type(kind)}"
            )
        if not isinstance(backend, CPSFPeriodizationBackend):
            raise TypeError(
                f"CPSFPeriodizationPolicy: 'backend' must be CPSFPeriodizationPolicyBackend, got {type(backend)}"
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
                        f"CPSFPeriodizationPolicy: '{name}' must be '{target_type}' scalar.",
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
            raise ValueError("CPSFPeriodizationPolicy: 'tolerance' must be finite")
        
        # Per-kind validation
        if kind == CPSFPeriodizationKind.WINDOW:
            if normal_window is None:
                raise ValueError(
                    "CPSFPeriodizationPolicy(WINDOW): 'window' is required"
                )
            if normal_window < 1:
                raise ValueError(
                    "CPSFPeriodizationPolicy(WINDOW): 'window' must be >= 1"
                )
            if normal_tolerance is not None or normal_max_radius is not None:
                raise ValueError(
                    "CPSFPeriodizationPolicy(WINDOW): 'tolerance' and 'max_radius' must be None"
                )
            if backend != CPSFPeriodizationBackend.AUTO:
                raise ValueError(
                    "CPSFPeriodizationPolicy(WINDOW): 'backend' must be AUTO"
                )
        elif kind == CPSFPeriodizationKind.FULL:
            if normal_window is not None:
                raise ValueError("CPSFPeriodizationPolicy(FULL): 'window' must be None")
            if normal_tolerance is None and normal_max_radius is None:
                raise ValueError(
                    "CPSFPeriodizationPolicy(FULL): either 'tolerance' or 'max_radius' must be provided"
                )
            if normal_tolerance is not None and not (normal_tolerance > 0.0):
                raise ValueError(
                    "CPSFPeriodizationPolicy(FULL): 'tolerance' must be > 0"
                )
            if normal_max_radius is not None and normal_max_radius < 1:
                raise ValueError(
                    "CPSFPeriodizationPolicy(FULL): 'max_radius' must be >= 1"
                )
        else:
            raise ValueError(f"CPSFPeriodizationPolicy: unsupported kind={kind}")

        # Assign values
        self.window = normal_window
        self.tolerance = normal_tolerance
        self.max_radius = normal_max_radius
        self.kind = kind
        self.backend = backend
