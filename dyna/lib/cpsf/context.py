import torch


from dyna.lib.cpsf.structures import (
    CPSFChunkPolicy,
    CPSFIntegrationPolicy,
    CPSFDTypes,
)
from dyna.lib.cpsf.periodization_policy import CPSFPeriodizationPolicy


class CPSFGeometryCache:
    """Cache by j: R_j, R_ext_j, Σ_j, L_j; invalidate by d, σ∥, σ⊥."""

    pass


class CPSFDerivedCache:
    """Cache by j: v_j = α_j * t_hat_j; invalidate by α or t_hat."""

    pass


class CPSFContext:
    def __init__(
        self,
        chunk: CPSFChunkPolicy,
        periodization: CPSFPeriodizationPolicy,
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
        self.geometry_cache = CPSFGeometryCache()
        self.derived_cache = CPSFDerivedCache()
