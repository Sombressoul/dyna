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
        **kwargs,
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
        self.input_dim_i = input_shape[0]
        self.input_dim_j = input_shape[1]

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
        cluster_shuffle: bool = True,
    ) -> torch.Tensor:
        # Group params.
        group_size_base = math.ceil(self.input_dim_i / self.group_count)
        group_size_overlap = math.ceil(group_size_base * self.group_overlap)

        # Nodes params.
        node_count = group_size_base
        node_connections_base = math.ceil(group_size_base * self.node_connectivity)
        node_connections_base = (
            node_connections_base if node_connections_base > 0 else 1
        )
        node_connections_overlap = math.ceil(
            group_size_overlap * self.node_connectivity
        )
        node_connections_overlap = (
            node_connections_overlap if node_connections_overlap > 0 else 1
        )
        node_connections_total = node_connections_base + node_connections_overlap

        # Create indices base with ceiling error compensation.
        cluster_size_extra = abs(self.input_dim_i - group_size_base * self.group_count)
        cluster_indices_base = torch.cat(
            [
                torch.arange(0, self.input_dim_i, 1),
                torch.randperm(self.input_dim_i)[0:cluster_size_extra],
            ],
            dim=0,
        )

        # Accumulator for indices.
        cluster_connections = torch.empty(
            [
                self.group_count,
                node_count,
                node_connections_total,
            ],
            dtype=torch.int32,
        )

        # Assembly connectivity per group.
        for group_index in range(self.group_count):
            indices_base = cluster_indices_base[
                (
                    torch.randperm(cluster_indices_base.shape[0])
                    if cluster_shuffle
                    else torch.arange(cluster_indices_base.shape[0])
                )
            ]

            group_start = group_index * group_size_base
            group_end = group_start + group_size_base

            group_inner_base = indices_base[group_start:group_end]
            group_inner = group_inner_base.clone()
            group_outer_base = torch.cat(
                [
                    indices_base[0:group_start],
                    indices_base[group_end::],
                ],
                dim=0,
            )
            group_outer = group_outer_base.clone()
            group_outer = group_outer[torch.randperm(group_outer.shape[0])]

            for node_index in range(node_count):
                node_inner_start = node_index * node_connections_base
                node_inner_end = node_inner_start + node_connections_base
                node_outer_start = node_index * node_connections_overlap
                node_outer_end = node_outer_start + node_connections_overlap

                if node_inner_end > group_inner.shape[0]:
                    group_inner = torch.cat(
                        [
                            group_inner,
                            group_inner_base[torch.randperm(group_inner_base.shape[0])],
                        ],
                        dim=0,
                    )

                if node_outer_end > group_outer.shape[0]:
                    group_outer = torch.cat(
                        [
                            group_outer,
                            group_outer_base[torch.randperm(group_outer_base.shape[0])],
                        ],
                        dim=0,
                    )

                node_inner = group_inner[node_inner_start:node_inner_end]
                node_outer = group_outer[node_outer_start:node_outer_end]

                cluster_connections[group_index, node_index, :] = torch.cat(
                    [node_inner, node_outer],
                    dim=0,
                )

        cluster_connections = cluster_connections.clone().contiguous()

        return cluster_connections
