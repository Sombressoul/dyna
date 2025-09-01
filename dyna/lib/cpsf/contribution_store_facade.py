import torch

from typing import Optional, Iterable


from dyna.lib.cpsf.structures import (
    CPSFContributionSet,
    CPSFContributionField,
    CPSFIndexLike,
)
from dyna.lib.cpsf.contribution_store import CPSFContributionStore


class CPSFContributionStoreFacade:
    def __init__(
        self,
        store: CPSFContributionStore,
    ):
        self.store = store
        self.epoch: int = 0

    def read_full(
        self,
        idx: CPSFIndexLike,
    ) -> CPSFContributionSet:
        raise NotImplementedError

    def read_fields(
        self,
        idx: CPSFIndexLike,
        fields: list[CPSFContributionField],
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
        raise NotImplementedError
