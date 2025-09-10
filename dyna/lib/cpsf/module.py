import torch

from typing import Optional, Tuple, Union

from dyna.lib.cpsf.structures import (
    CPSFConsistency,
    CPSFModuleReadFlags,
)
from dyna.lib.cpsf.contribution_store_facade import CPSFContributionStoreFacade


class CPSFModule:
    def __init__(
        self,
        store_facade: CPSFContributionStoreFacade,
    ):
        self.store_facade = store_facade

    def _resolve_read_flags(
        self,
        consistency: CPSFConsistency,
        overrides: Optional[CPSFModuleReadFlags],
    ) -> CPSFModuleReadFlags:
        if consistency == CPSFConsistency.snapshot:
            flags = CPSFModuleReadFlags(
                active_buffer=False,
                active_overlay=False,
            )
        elif consistency == CPSFConsistency.live:
            flags = CPSFModuleReadFlags(
                active_buffer=True,
                active_overlay=True,
            )
        else:
            raise ValueError(f"Unknown consistency: {consistency}")

        if overrides is not None:
            if not isinstance(overrides, CPSFModuleReadFlags):
                raise TypeError(
                    "\n".join(
                        [
                            f"overrides must be CPSFModuleReadFlags.",
                            f"Got {type(overrides)}",
                        ]
                    )
                )

            flags.active_buffer = (
                overrides.active_buffer
                if overrides.active_buffer is not None
                else flags.active_buffer
            )
            flags.active_overlay = (
                overrides.active_overlay
                if overrides.active_overlay is not None
                else flags.active_overlay
            )

        return flags

    def read(
        self,
        z: torch.Tensor,
        d: torch.Tensor,
        consistency: CPSFConsistency = CPSFConsistency.snapshot,
        overrides: Optional[CPSFModuleReadFlags] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        raise NotImplementedError

    def find(
        self,
        T_star: torch.Tensor,
        consistency: CPSFConsistency = CPSFConsistency.snapshot,
        overrides: Optional[CPSFModuleReadFlags] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def project_error(
        self,
        z: torch.Tensor,
        d: torch.Tensor,
        T_ref: torch.Tensor,
        consistency: CPSFConsistency = CPSFConsistency.snapshot,
        overrides: Optional[CPSFModuleReadFlags] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        raise NotImplementedError

    def generative_recall(
        self,
        T_star: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
        consistency: CPSFConsistency = CPSFConsistency.snapshot,
        overrides: Optional[CPSFModuleReadFlags] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
