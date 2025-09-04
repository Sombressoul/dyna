import torch

from dataclasses import (
    dataclass,
    field as dataclasses_field,
)
from enum import Enum, auto as enum_auto
from typing import Sequence, Optional, Union, Literal


CPSFIndexLike = Union[torch.Tensor, Sequence[int]]
CPSFSelection = Union[slice, torch.LongTensor, list, tuple]


class CPSFConsistency(Enum):
    snapshot = enum_auto()
    live = enum_auto()


@dataclass
class CPSFModuleReadFlags:
    """
    Optional behavior overrides for CPSFModule for a single call.
    Does not change the canon; only affects the Store read policy.

    active_buffer  — True: read from buffer (new entries) as well,
                     False: ignore buffer (permanent only).
    active_overlay — True: consider overlay deltas when reading,
                     False: ignore overlay when reading.
    """

    active_buffer: Optional[bool] = None
    active_overlay: Optional[bool] = None


@dataclass
class CPSFContributionSet:
    idx: Optional[CPSFIndexLike] = None
    z: Optional[torch.Tensor] = None
    vec_d: Optional[torch.Tensor] = None
    t_hat: Optional[torch.Tensor] = None
    sigma_par: Optional[torch.Tensor] = None
    sigma_perp: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None


@dataclass
class CPSFContributionStoreIDList:
    permanent: list[int] = dataclasses_field(default_factory=list)
    buffer: list[int] = dataclasses_field(default_factory=list)


class CPSFContributionField(Enum):
    Z = enum_auto()
    VEC_D = enum_auto()
    T_HAT = enum_auto()
    SIGMA_PAR = enum_auto()
    SIGMA_PERP = enum_auto()
    ALPHA = enum_auto()


@dataclass
class CPSFChunkPolicy:
    J_tile: int
    Q_tile: int
    S_tile: Optional[int] = None


class CPSFLatticeSumPolicyKind(Enum):
    FULL = enum_auto()
    WINDOW = enum_auto()


@dataclass
class CPSFLatticeSumPolicy:
    kind: Literal["full", "window"] = "window"
    window: Optional[Union[int, Sequence[torch.LongTensor]]] = None

    def fixed_window(
        self,
        N: int,
    ) -> torch.LongTensor:
        from dyna.lib.cpsf.functional.lattice import fixed_window as _fw
        return _fw(self, N)


@dataclass
class CPSFIntegrationPolicy:
    kind: Literal["quad", "mc", "strat"] = "quad"
    samples: int = 0
    seed: Optional[int] = None


@dataclass
class CPSFDTypes:
    dtype_r: torch.dtype
    dtype_c: torch.dtype
    accum_dtype: torch.dtype
    device: torch.device
