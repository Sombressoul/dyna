import torch

from collections.abc import Iterable
from dataclasses import (
    dataclass,
    field as dataclasses_field,
    fields as dataclasses_fields,
)
from enum import Enum, auto as enum_auto
from typing import Sequence, Optional, Union


IndexLike = Union[torch.Tensor, Sequence[int]]


@dataclass
class CPSFContributionSet:
    idx: Optional[IndexLike] = None
    z: Optional[torch.Tensor] = None
    vec_d: Optional[torch.Tensor] = None
    T_hat: Optional[torch.Tensor] = None
    sigma_par: Optional[torch.Tensor] = None
    sigma_perp: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None


@dataclass
class CPSFContributionStoreIDList:
    permanent: list[int] = dataclasses_field(default_factory=list)
    buffer: list[int] = dataclasses_field(default_factory=list)


class CPSFContributionField(Enum):
    Z = enum_auto()
    VEC_D = enum_auto()
    T_HAT = enum_auto()
    SIGMA_PAR = enum_auto()
    SIGMA_PERP = enum_auto()
    ALPHA = enum_auto()


class CPSFContributionStore:
    def __init__(
        self,
        N: int,
        S: int,
        dtype_r: torch.dtype,
        dtype_c: torch.dtype,
    ) -> None:
        if torch.is_complex(torch.empty((), dtype=dtype_r)):
            raise TypeError("dtype_r must be a real floating dtype.")

        if not torch.is_complex(torch.empty((), dtype=dtype_c)):
            raise TypeError("dtype_c must be a complex dtype.")

        self.N = int(N)
        self.S = int(S)
        self.target_dtype_r = dtype_r
        self.target_dtype_c = dtype_c

        # Define slices.
        # Data order: (z, vec_d, T_hat, sigma_par, sigma_perp, alpha)
        self._slice_z = slice(
            0,
            self.N * 2,
            1,
        )
        self._slice_vec_d = slice(
            self._slice_z.stop,
            self._slice_z.stop + self.N * 2,
            1,
        )
        self._slice_T_hat = slice(
            self._slice_vec_d.stop,
            self._slice_vec_d.stop + self.S * 2,
            1,
        )
        self._slice_sigma_par = slice(
            self._slice_T_hat.stop,
            self._slice_T_hat.stop + 1,
            1,
        )
        self._slice_sigma_perp = slice(
            self._slice_sigma_par.stop,
            self._slice_sigma_par.stop + 1,
            1,
        )
        self._slice_alpha = slice(
            self._slice_sigma_perp.stop,
            self._slice_sigma_perp.stop + 1,
            1,
        )

        # Define full contribution length.
        self._contribution_length = self._slice_alpha.stop

        # Main contributions storage.
        self._C = torch.nn.Parameter(
            data=torch.empty(
                size=[0, self._contribution_length],
                dtype=self.target_dtype_r,
                requires_grad=True,
            )
        )

        # Buffer contributions storage for dynamic interactions.
        self._C_buffer = []

        # A list of IDs of inactive contributions awaiting deletion.
        self._C_inactive = CPSFContributionStoreIDList()

        pass

    def __len__(self) -> int:
        return len(self._C) + len(self._C_buffer)

    def _is_full_contribution_set(
        self,
        contribution_set: CPSFContributionSet,
    ) -> bool:
        return all(
            getattr(contribution_set, field_name) is not None
            for field_name in (
                "z",
                "vec_d",
                "T_hat",
                "sigma_par",
                "sigma_perp",
                "alpha",
            )
        )

    def _flat_to_set(
        self,
        contribution_flat: torch.Tensor,
    ) -> CPSFContributionSet:
        if contribution_flat.shape[1] != self._contribution_length:
            raise ValueError(
                f"Flat contributions must have shape (batch, {self._contribution_length}), "
                f"but got {contribution_flat.shape}."
            )

        contributions_set = CPSFContributionSet(
            idx=None,
            z=torch.complex(
                real=contribution_flat[:, self._slice_z][:, : self.N],
                imag=contribution_flat[:, self._slice_z][:, self.N :],
            ).to(dtype=self.target_dtype_c),
            vec_d=torch.complex(
                real=contribution_flat[:, self._slice_vec_d][:, : self.N],
                imag=contribution_flat[:, self._slice_vec_d][:, self.N :],
            ).to(dtype=self.target_dtype_c),
            T_hat=torch.complex(
                real=contribution_flat[:, self._slice_T_hat][:, : self.S],
                imag=contribution_flat[:, self._slice_T_hat][:, self.S :],
            ).to(dtype=self.target_dtype_c),
            sigma_par=contribution_flat[:, self._slice_sigma_par],
            sigma_perp=contribution_flat[:, self._slice_sigma_perp],
            alpha=contribution_flat[:, self._slice_alpha],
        )

        return contributions_set

    def _set_to_flat(
        self,
        contribution_set: CPSFContributionSet,
    ) -> torch.Tensor:
        if not self._is_full_contribution_set(contribution_set):
            raise ValueError("_set_to_flat() requires a complete CPSFContributionSet.")

        z_real = torch.real(contribution_set.z)
        z_imag = torch.imag(contribution_set.z)
        z_flat = torch.cat([z_real, z_imag], dim=1)

        vec_d_real = torch.real(contribution_set.vec_d)
        vec_d_imag = torch.imag(contribution_set.vec_d)
        vec_d_flat = torch.cat([vec_d_real, vec_d_imag], dim=1)

        T_hat_real = torch.real(contribution_set.T_hat)
        T_hat_imag = torch.imag(contribution_set.T_hat)
        T_hat_flat = torch.cat([T_hat_real, T_hat_imag], dim=1)

        sigma_par_flat = contribution_set.sigma_par
        sigma_perp_flat = contribution_set.sigma_perp
        alpha_flat = contribution_set.alpha

        flat_contribution = torch.cat(
            [
                z_flat,
                vec_d_flat,
                T_hat_flat,
                sigma_par_flat,
                sigma_perp_flat,
                alpha_flat,
            ],
            dim=1,
        )

        if flat_contribution.shape[1] != self._contribution_length:
            raise ValueError(
                f"Packed contribution length {flat_contribution.shape[1]} "
                f"does not match expected {self._contribution_length}."
            )

        return flat_contribution.to(dtype=self.target_dtype_r)

    def _idx_format(
        self,
        idx: IndexLike,
    ) -> list[int]:
        if isinstance(idx, torch.Tensor):
            idx_list = idx.flatten().tolist()
        elif isinstance(idx, int):
            idx_list = [idx]
        elif isinstance(idx, (list, tuple)):
            idx_list = list(idx)
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

        idx_list = [int(i) for i in idx_list]
        if not all(i >= 0 for i in idx_list):
            raise IndexError("Index must be non-negative")

        total_len = len(self._C) + len(self._C_buffer)
        if not all(i < total_len for i in idx_list):
            raise IndexError("Index out of range")

        return idx_list

    def _normalize_fields_arg(
        self,
        fields: Optional[Union[CPSFContributionField, Iterable[CPSFContributionField]]],
    ) -> Optional[list[CPSFContributionField]]:
        if fields is None:
            return None

        if isinstance(fields, CPSFContributionField):
            return [fields]

        if not isinstance(fields, Iterable):
            raise TypeError("fields must be iterable or a CPSFContributionField.")
        if not all(isinstance(field, CPSFContributionField) for field in fields):
            raise TypeError("fields must be instances of CPSFContributionField.")

        return list(fields)

    def _validate_contribution_set(
        self,
        contribution_set: CPSFContributionSet,
    ) -> None:
        if not isinstance(contribution_set, CPSFContributionSet):
            raise TypeError("Expected CPSFContributionSet instance.")

        field_names = [
            f.name for f in dataclasses_fields(CPSFContributionSet) if f.name != "idx"
        ]

        batch_sizes = [
            getattr(contribution_set, name).shape[0]
            for name in field_names
            if getattr(contribution_set, name) is not None
        ]

        if not batch_sizes:
            raise ValueError("CPSFContributionSet contains no data.")
        if any(s != batch_sizes[0] for s in batch_sizes):
            raise ValueError("Inconsistent batch size among fields.")
        if batch_sizes[0] == 0:
            raise ValueError("CPSFContributionSet is empty.")

    def idx_format_to_internal(
        self,
        idx: IndexLike,
    ) -> CPSFContributionStoreIDList:
        idx_internal = CPSFContributionStoreIDList()

        idx_list = self._idx_format(idx)

        for i in idx_list:
            if i < len(self._C):
                idx_internal.permanent.append(i)
            else:
                idx_internal.buffer.append(i - len(self._C))

        return idx_internal

    def idx_format_to_external(
        self,
        idx: CPSFContributionStoreIDList,
    ) -> list[int]:
        idx_external = list(idx.permanent)

        offset = len(self._C)
        idx_external.extend(i + offset for i in idx.buffer)

        return idx_external

    def create(
        self,
        contribution_set: CPSFContributionSet,
    ) -> None:
        self._validate_contribution_set(contribution_set)

        if not self._is_full_contribution_set(contribution_set):
            raise ValueError("CPSFContributionSet must be complete for create().")

        contribution_flat = self._set_to_flat(contribution_set)
        target = dict(device=self._C.device, dtype=self.target_dtype_r)

        if contribution_flat.numel() == 0:
            raise ValueError("Cannot create from an empty CPSFContributionSet.")

        for entry in contribution_flat.unbind(dim=0):
            self._C_buffer.append(
                torch.nn.Parameter(
                    data=entry.unsqueeze(0).to(**target),
                    requires_grad=True,
                )
            )

    def read(
        self,
        idx: IndexLike,
        fields: list[CPSFContributionField] = None,
    ) -> CPSFContributionSet:
        idx_list = self._idx_format(idx)
        fields = self._normalize_fields_arg(fields)
        inactive_set = set(self.idx_inactive())

        if any(i in inactive_set for i in idx_list):
            raise IndexError("Requested index refers to an inactive contribution.")

        if fields is None:  # Direct full read (fallback)
            buffer_offset = len(self._C)
            parts = []
            for i in idx_list:
                if i < buffer_offset:
                    parts.append(self._C[i].unsqueeze(0))
                else:
                    parts.append(self._C_buffer[i - buffer_offset])

            contributions_flat = torch.cat(parts, dim=0)

            contributions_set = self._flat_to_set(contributions_flat)
        else:  # Partial read.
            # TODO: Implement partial read operation.

            raise NotImplementedError("Partial read is not yet implemented.")

        contributions_set.idx = idx_list

        return contributions_set

    def update(
        self,
        contribution_set: CPSFContributionSet,
        fields: list[CPSFContributionField] = None,
    ) -> None:
        self._validate_contribution_set(contribution_set)

        fields = self._normalize_fields_arg(fields)
        if fields is None:  # Full update
            if not self._is_full_contribution_set(contribution_set):
                raise ValueError("CPSFContributionSet must be complete for non-partial update().")
            # TODO: Full update implementation.
            ...
        else:
            # TODO: Partial update implementation.
            ...

        raise NotImplementedError("update is not yet implemented.")

    def delete(
        self,
        idx: Union[int, IndexLike],
    ) -> None:
        idx_delete = self.idx_format_to_internal(idx)
        self._C_inactive.permanent = sorted(
            set(self._C_inactive.permanent) | set(idx_delete.permanent)
        )
        self._C_inactive.buffer = sorted(
            set(self._C_inactive.buffer) | set(idx_delete.buffer)
        )

    def is_active(
        self,
        idx: Union[int, IndexLike],
    ) -> list[bool]:
        idx_test = self._idx_format(idx)
        idx_active = set(self.idx_active())  # O(1) check
        return [id in idx_active for id in idx_test]

    def idx_active(
        self,
    ) -> list[int]:
        inactive_set = set(self.idx_inactive())
        return [i for i in range(len(self)) if i not in inactive_set]

    def idx_inactive(
        self,
    ) -> list[int]:
        buffer_offset = len(self._C)
        inactive_set = set(self._C_inactive.permanent) | {
            buffer_offset + i for i in self._C_inactive.buffer
        }
        return sorted(inactive_set)

    def idx_permanent(
        self,
        active: bool = True,
    ) -> list[int]:
        if active:
            inactive = set(self._C_inactive.permanent)
            return [i for i in range(len(self._C)) if i not in inactive]
        else:
            return list(range(len(self._C)))

    def idx_buffer(
        self,
        active: bool = True,
    ) -> list[int]:
        buffer_offset = len(self._C)
        idx_buffer = set(range(buffer_offset, len(self)))

        if active:
            inactive = {buffer_offset + i for i in self._C_inactive.buffer}
            return sorted(idx_buffer - inactive)
        else:
            return sorted(idx_buffer)

    def clear_buffer(
        self,
    ) -> bool:
        changed = bool(self._C_buffer or self._C_inactive.buffer)
        if changed:
            self._C_buffer.clear()
            self._C_inactive.buffer.clear()
        return changed

    def read_all_active(
        self,
    ) -> CPSFContributionSet:
        return self.read(idx=self.idx_active())

    def consolidate(self) -> bool:
        """
        Merge active contributions from the main storage and buffer into a single
        contiguous Parameter, while removing all contributions marked as inactive.

        Returns
        -------
        bool
            True if any changes were made (removal of inactive contributions or
            merging of buffered ones), False otherwise.

        Notes
        -----
        - Optimizer State:
        Since a new `torch.nn.Parameter` object is created, any optimizer state
        (e.g., momentum in SGD, moving averages in Adam) for the old parameter
        will be lost. The optimizer must be re-initialized.
        - Order Preservation:
        All active permanent contributions appear first, followed by all active
        buffered contributions.
        """

        if not (
            self._C_inactive.permanent or self._C_inactive.buffer or self._C_buffer
        ):
            return False

        target = dict(dtype=self.target_dtype_r, device=self._C.device)
        empty_C = torch.empty(size=[0, self._contribution_length], **target)

        # Collect active permanent.
        if self._C_inactive.permanent:
            active_permanent_indices = [
                i for i in range(len(self._C)) if i not in self._C_inactive.permanent
            ]
            active_permanent = (
                self._C[active_permanent_indices]
                if active_permanent_indices
                else empty_C
            )
        else:
            active_permanent = self._C

        # Collect active bufferized.
        if self._C_buffer:
            if self._C_inactive.buffer:
                active_buffer_indices = [
                    i
                    for i in range(len(self._C_buffer))
                    if i not in self._C_inactive.buffer
                ]
                if active_buffer_indices:
                    active_buffer = torch.cat(
                        tensors=[self._C_buffer[i] for i in active_buffer_indices],
                        dim=0,
                    )
                else:
                    active_buffer = empty_C
            else:
                active_buffer = torch.cat(self._C_buffer, dim=0)
        else:
            active_buffer = empty_C

        # Check device and dtype: active_permanent.
        if (
            active_permanent.dtype != self.target_dtype_r
            or active_permanent.device != self._C.device
        ):
            active_permanent = active_permanent.to(**target)

        # Check device and dtype: active_buffer.
        if (
            active_buffer.dtype != self.target_dtype_r
            or active_buffer.device != self._C.device
        ):
            active_buffer = active_buffer.to(**target)

        # Update permanent contributions.
        self._C = torch.nn.Parameter(
            data=torch.cat(
                tensors=[
                    active_permanent,
                    active_buffer,
                ],
                dim=0,
            ).contiguous(),
            requires_grad=True,
        )
        self._C_buffer = []
        self._C_inactive = CPSFContributionStoreIDList()

        return True
