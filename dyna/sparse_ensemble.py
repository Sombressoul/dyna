import torch
import torch.nn as nn
import math

from typing import Union


class SparseEnsemble(nn.Module):
    def __init__(
        self,
        input_shape: Union[torch.Size, tuple[int, int]],
        cluster_count: int = 2,
        group_count: int = 4,
        group_overlap: float = 0.5,
        node_connectivity: float = 0.25,
        **kwargs
    ) -> None:
        super(SparseEnsemble, self).__init__(**kwargs)

        # ================================================================================= #
        # ____________________________> Initial checks.
        # ================================================================================= #
        input_shape = (
            torch.Size(input_shape) if type(input_shape) == tuple else input_shape
        )

        assert len(input_shape) == 2, "Input shape must be 2D."
        assert cluster_count > 1, "Cluster count must be greater than 1."
        assert group_count > 1, "Group count must be greater than 1."
        assert group_overlap >= 0, "Group overlap must be greater or equal to 0."
        assert group_overlap <= 1, "Group overlap must be less or equal to 1."
        assert node_connectivity <= 1, "Node connectivity must be less or equal to 1."
        assert node_connectivity > 0, "Node connectivity must be greater than 0."

        # ================================================================================= #
        # ____________________________> Arguments.
        # ================================================================================= #
        self.input_shape = input_shape
        self.cluster_count = cluster_count
        self.group_count = group_count
        self.group_overlap = group_overlap
        self.node_connectivity = node_connectivity

        # Internal variables.
        self.dim_i = input_shape[0]
        self.dim_j = input_shape[1]

        # ================================================================================= #
        # ____________________________> Connectivity.
        # ================================================================================= #
        # TODO: implement.

        # ================================================================================= #
        # ____________________________> Weights.
        # ================================================================================= #
        # TODO: implement.

        pass

    def _get_cluster_connections(
        self,
        group_count: int,
        group_overlap: float,
        node_connectivity: float,
        cluster_shuffle: bool = True,
    ) -> torch.Tensor:
        group_size_base = math.ceil(self.dim_i / group_count)
        group_size_overlap = math.ceil(group_size_base * group_overlap)
        group_size_full = group_size_base + group_size_overlap

        node_count = group_size_base
        node_connections_base = math.ceil(group_size_base * node_connectivity)
        node_connections_base = node_connections_base if node_connections_base > 0 else 1
        node_connections_overlap = math.ceil(group_size_overlap * node_connectivity)
        node_connections_overlap = node_connections_overlap if node_connections_overlap > 0 else 1
        node_connections_total = node_connections_base + node_connections_overlap

        cluster_size_extra = abs(self.dim_i - group_size_full * group_count)
        cluster_indices_base = torch.cat(
            [
                torch.arange(0, self.dim_i, 1),
                torch.randperm(self.dim_i)[0:cluster_size_extra],
            ],
            dim=0,
        )
        cluster_connections = torch.empty(
            [
                group_count,
                group_size_base,
                node_connections_total,
            ],
            dtype=torch.int32,
        )

        for group_index in range(group_count):
            indices_base = cluster_indices_base[
                (
                    torch.randperm(cluster_indices_base.shape[0])
                    if cluster_shuffle
                    else torch.arange(cluster_indices_base.shape[0])
                )
            ]
            indices_overlap = indices_base[torch.randperm(indices_base.shape[0])]

            group_start = group_index * group_size_base
            group_end = group_start + group_size_base

            group_inner = indices_base[group_start:group_end]
            group_outer = torch.cat([indices_base[0:group_start], indices_base[group_end::]], dim=0)

            # TODO: the rest...

        pass
