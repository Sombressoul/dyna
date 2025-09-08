import torch

from typing import Optional, Union

from dyna.lib.cpsf.structures import (
    CPSFPeriodizationPolicyBackend,
    CPSFPeriodizationPolicyKind,
)


class CPSFPeriodizationPolicy:
    def __init__(
        self,
        kind: CPSFPeriodizationPolicyKind,
        window: Optional[Union[int, torch.LongTensor]] = None,
        tolerance: Optional[Union[float, torch.FloatTensor]] = None,
        max_radius: Optional[Union[int, torch.LongTensor]] = None,
        backend: CPSFPeriodizationPolicyBackend = CPSFPeriodizationPolicyBackend.AUTO,
    ):
        self.kind = kind
        self.window = window
        self.tolerance = tolerance
        self.max_radius = max_radius
        self.backend = backend
