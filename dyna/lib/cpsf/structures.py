import torch

from dataclasses import (
    dataclass,
    field as dataclasses_field,
)
from enum import Enum, auto as enum_auto
from typing import Sequence, Optional, Union


IndexLike = Union[torch.Tensor, Sequence[int]]


@dataclass
class ContributionSet:
    idx: Optional[IndexLike] = None
    z: Optional[torch.Tensor] = None
    vec_d: Optional[torch.Tensor] = None
    t_hat: Optional[torch.Tensor] = None
    sigma_par: Optional[torch.Tensor] = None
    sigma_perp: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None


@dataclass
class ContributionStoreIDList:
    permanent: list[int] = dataclasses_field(default_factory=list)
    buffer: list[int] = dataclasses_field(default_factory=list)


class ContributionField(Enum):
    Z = enum_auto()
    VEC_D = enum_auto()
    T_HAT = enum_auto()
    SIGMA_PAR = enum_auto()
    SIGMA_PERP = enum_auto()
    ALPHA = enum_auto()
