import torch


from dyna.lib.cpsf.structures import (
    CPSFChunkPolicy,
    CPSFIntegrationPolicy,
    CPSFDTypes,
)
from dyna.lib.cpsf.periodization import CPSFPeriodization


class CPSFContext:
    def __init__(
        self,
        chunk: CPSFChunkPolicy,
        periodization: CPSFPeriodization,
        integration: CPSFIntegrationPolicy,
        dtypes: CPSFDTypes,
        exp_clip_q_max: float = 60.0,
    ):
        self.chunk = chunk
        self.lattice = periodization
        self.integration = integration
        self.dtypes = dtypes
        self.exp_clip_q_max = exp_clip_q_max
        self.epoch: int = 0
