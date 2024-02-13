import torch

from typing import Optional
from dataclasses import dataclass


@dataclass
class SignalComponential:
    x: torch.Tensor
    components: Optional[torch.Tensor]
    nonlinearity: Optional[torch.Tensor]

@dataclass
class SignalModular:
    x: torch.Tensor
    modes: Optional[torch.Tensor]
