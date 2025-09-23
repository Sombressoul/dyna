import torch

from typing import Optional, Tuple

from dyna.lib.cpsf.structures import (
    CPSFConsistency,
    CPSFContributionField,
    CPSFContributionSet,
    CPSFIndexLike,
    CPSFModuleReadFlags,
)
from dyna.lib.cpsf.contribution_store import CPSFContributionStore
from dyna.lib.cpsf.core import CPSFCore


class CPSFModule:
    def __init__(
        self,
        store: CPSFContributionStore,
        consistency: CPSFConsistency = CPSFConsistency.live,
    ):
        # Guards
        if not isinstance(consistency, CPSFConsistency):
            raise ValueError(
                f"CPSFModule: 'consistency' should be an instance of CPSFConsistency, got '{type(consistency)}'"
            )
        if not isinstance(store, CPSFContributionStore):
            raise ValueError(
                f"CPSFModule: 'store' should be an instance of CPSFContributionStore, got '{type(store)}'"
            )

        # Assign
        self.core = CPSFCore()
        self.consistency = consistency
        self.store = store

    def _store_create(
        self,
        contribution_set: CPSFContributionSet,
    ) -> None:
        return self.store.create(
            contribution_set=contribution_set,
        )

    def _store_read(
        self,
        idx: CPSFIndexLike,
        fields: list[CPSFContributionField] = None,
        active_buffer: bool = True,
        active_overlay: bool = True,
    ) -> CPSFContributionSet:
        return self.store.read(
            idx=idx,
            fields=fields,
            active_buffer=active_buffer,
            active_overlay=active_overlay,
        )

    def _store_update(
        self,
        contribution_set: CPSFContributionSet,
        fields: list[CPSFContributionField] = None,
        preserve_grad: bool = True,
    ) -> None:
        return self.store.update(
            contribution_set=contribution_set,
            fields=fields,
            preserve_grad=preserve_grad,
        )

    def _store_delete(
        self,
        idx: CPSFIndexLike,
    ) -> None:
        return self.store.delete(
            idx=idx,
        )

    def _store_consolidate(
        self,
    ) -> bool:
        return self.store.consolidate()

    def _resolve_read_flags(
        self,
        consistency_overrides: Optional[CPSFModuleReadFlags],
    ) -> CPSFModuleReadFlags:
        if self.consistency == CPSFConsistency.snapshot:
            flags = CPSFModuleReadFlags(
                active_buffer=False,
                active_overlay=False,
            )
        elif self.consistency == CPSFConsistency.live:
            flags = CPSFModuleReadFlags(
                active_buffer=True,
                active_overlay=True,
            )
        else:
            raise ValueError(f"Unknown consistency: {self.consistency}")

        if consistency_overrides is not None:
            if not isinstance(consistency_overrides, CPSFModuleReadFlags):
                raise TypeError(
                    "\n".join(
                        [
                            f"overrides must be CPSFModuleReadFlags.",
                            f"Got {type(consistency_overrides)}",
                        ]
                    )
                )

            flags.active_buffer = (
                consistency_overrides.active_buffer
                if consistency_overrides.active_buffer is not None
                else flags.active_buffer
            )
            flags.active_overlay = (
                consistency_overrides.active_overlay
                if consistency_overrides.active_overlay is not None
                else flags.active_overlay
            )

        return flags

    def read(
        self,
        z: torch.Tensor,
        d: torch.Tensor,
        consistency_overrides: Optional[CPSFModuleReadFlags] = None,
    ) -> torch.Tensor:
        storage_flags = self._resolve_read_flags(consistency_overrides)
        active_buffer = storage_flags.active_buffer
        active_overlay = storage_flags.active_overlay

        # TODO: Implement CPSFRouter to route contributions between T-field reading
        #       backends (T_classic_full, T_classic_window, other...).

        raise NotImplementedError("TODO")

    def find(
        self,
        T_star: torch.Tensor,
        consistency_overrides: Optional[CPSFModuleReadFlags] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        storage_flags = self._resolve_read_flags(consistency_overrides)
        active_buffer = storage_flags.active_buffer
        active_overlay = storage_flags.active_overlay

        # TODO: Find best match [z, vec_d] for target T_star in the field.
        #       See "Semantic Inverse Projection".

        raise NotImplementedError("TODO")

    def project(
        self,
        z: torch.Tensor,
        d: torch.Tensor,
        T_star: torch.Tensor,
        learning_rate: float,
        consistency_overrides: Optional[CPSFModuleReadFlags] = None,
    ) -> None:
        storage_flags = self._resolve_read_flags(consistency_overrides)
        active_buffer = storage_flags.active_buffer
        active_overlay = storage_flags.active_overlay

        # TODO: Project error back to contributions (see "Semantic Error Projection")
        #       with controllable learning rate.

        raise NotImplementedError("TODO")

    def generate(
        self,
        T_star: torch.Tensor,
        deviation_limit: float,
        consistency_overrides: Optional[CPSFModuleReadFlags] = None,
    ) -> CPSFContributionSet:
        storage_flags = self._resolve_read_flags(consistency_overrides)
        active_buffer = storage_flags.active_buffer
        active_overlay = storage_flags.active_overlay

        # TODO: Generate new contributions (see "Generative Recall") with limited
        #       max deviation.

        raise NotImplementedError("TODO")

    def consolidate(
        self,
    ) -> bool:
        # TODO: Any additional logic:
        #       consolidation hooks, cache invalidation, etc.
        return self._store_consolidate()
