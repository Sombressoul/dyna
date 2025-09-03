import torch

from typing import Optional, Iterable


from dyna.lib.cpsf.structures import (
    CPSFContributionField,
    CPSFContributionSet,
    CPSFIndexLike,
)
from dyna.lib.cpsf.context import CPSFContext
from dyna.lib.cpsf.contribution_store import CPSFContributionStore


class CPSFContributionStoreFacade:
    def __init__(
        self,
        store: CPSFContributionStore,
        context: CPSFContext,
    ):
        self.store = store
        self.ctx = context

    def read_full(
        self,
        idx: CPSFIndexLike,
        active_buffer: bool = True,
        active_overlay: bool = True,
    ) -> CPSFContributionSet:
        raise NotImplementedError

    def read_fields(
        self,
        idx: CPSFIndexLike,
        fields: list[CPSFContributionField],
        active_buffer: bool = True,
        active_overlay: bool = True,
    ) -> CPSFContributionSet:
        raise NotImplementedError

    def update(
        self,
        cs: CPSFContributionSet,
        fields: Optional[Iterable[CPSFContributionField]] = None,
        preserve_grad: bool = True,
    ) -> None:
        raise NotImplementedError

    def delete(
        self,
        idx: CPSFIndexLike,
    ) -> None:
        raise NotImplementedError

    def consolidate(
        self,
    ) -> None:
        raise NotImplementedError

    def begin_snapshot(
        self,
    ) -> int:
        self.ctx.epoch += 1
        return self.ctx.epoch
