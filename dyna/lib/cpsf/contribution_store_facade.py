from typing import Optional, Iterable

from dyna.lib.cpsf.structures import (
    CPSFContributionField,
    CPSFContributionSet,
    CPSFIndexLike,
)
from dyna.lib.cpsf.contribution_store import CPSFContributionStore


class CPSFContributionStoreFacade:
    def __init__(
        self,
        store: CPSFContributionStore,
    ):
        self.store = store

    def read(
        self,
        idx: CPSFIndexLike,
        fields: Optional[Iterable[CPSFContributionField]] = None,
        active_buffer: bool = True,
        active_overlay: bool = True,
    ) -> CPSFContributionSet:
        return self.store.read(
            idx=idx,
            fields=fields,
            active_buffer=active_buffer,
            active_overlay=active_overlay,
        )

    def update(
        self,
        contribution_set: CPSFContributionSet,
        fields: Optional[Iterable[CPSFContributionField]] = None,
        preserve_grad: bool = True,
    ) -> None:
        self.store.update(
            contribution_set=contribution_set,
            fields=fields,
            preserve_grad=preserve_grad,
        )

    def delete(
        self,
        idx: CPSFIndexLike,
    ) -> None:
        self.store.delete(
            idx=idx,
        )

    def consolidate(
        self,
    ) -> bool:
        return self.store.consolidate()
