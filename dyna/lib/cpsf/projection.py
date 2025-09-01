import torch

from typing import Optional

from dyna.lib.cpsf.context import CPSFContext


class CPSFProjection:
    def __init__(
        self,
        context: CPSFContext,
    ):
        self.ctx = context

    def resolvent_delta_T_hat(
        self,
        alpha_j: torch.Tensor,
        v_j: torch.Tensor,
        A: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
