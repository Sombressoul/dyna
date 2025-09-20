import torch

from collections import OrderedDict
from typing import Generator, Optional, Tuple, TypeAlias

_INT_DTYPES: Tuple[torch.dtype, ...] = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.long,
)

LRUKey: TypeAlias = Tuple[int, int, Tuple[str, int]]
LRUCache: TypeAlias = "OrderedDict[LRUKey, torch.Tensor]"


class CPSFPeriodization:
    def __init__(
        self,
        *,
        enable_cache: bool = True,
        max_cache_entries_per_kind: int = 16,
        max_cache_bytes_per_tensor: int = 256 * 1024 * 1024,
        dtype: torch.dtype = torch.long,
    ) -> None:
        if not isinstance(dtype, torch.dtype):
            raise ValueError("dtype: must be a torch.dtype.")

        if dtype not in _INT_DTYPES or dtype is torch.bool:
            raise ValueError("dtype: must be an integer dtype (int8/16/32/64).")

        if not isinstance(enable_cache, (bool, int)):
            raise ValueError("enable_cache: must be bool.")

        if not isinstance(max_cache_entries_per_kind, int):
            raise ValueError("max_cache_entries_per_kind: must be an integer.")
        if max_cache_entries_per_kind < 0:
            raise ValueError("max_cache_entries_per_kind: must be >= 0.")

        if not isinstance(max_cache_bytes_per_tensor, int):
            raise ValueError("max_cache_bytes_per_tensor: must be an integer.")
        if max_cache_bytes_per_tensor < 0:
            raise ValueError("max_cache_bytes_per_tensor: must be >= 0.")

        self.enable_cache = bool(enable_cache)
        self.max_cache_entries = int(max_cache_entries_per_kind)
        self.max_cache_bytes_per_tensor = int(max_cache_bytes_per_tensor)
        self.dtype = dtype

        self._elem_size_bytes = torch.empty(0, dtype=self.dtype).element_size()

        if self.enable_cache:
            if self.max_cache_bytes_per_tensor < self._elem_size_bytes:
                raise ValueError(
                    "max_cache_bytes_per_tensor: is too small for chosen dtype."
                )

        self._cache_window: LRUCache = OrderedDict()
        self._cache_shell: LRUCache = OrderedDict()

    @staticmethod
    def window_size(
        *,
        N: int,
        W: int,
    ) -> int:
        if type(N) is not int or N < 1:
            raise ValueError("window_size: N must be int >= 1 (complex dimension).")
        if type(W) is not int or W < 0:
            raise ValueError("window_size: W must be int >= 0.")

        if W == 0:
            return 1

        tw = (W << 1) + 1

        return int(pow(tw, 2 * N))

    @staticmethod
    def shell_size(
        *,
        N: int,
        W: int,
    ) -> int:
        if type(N) is not int or N < 1:
            raise ValueError("shell_size: N must be int >= 1 (complex dimension).")
        if type(W) is not int or W < 0:
            raise ValueError("shell_size: W must be int >= 0.")

        if W == 0:
            return 1

        tw = (W << 1) + 1
        tm = (W << 1) - 1

        return int(pow(tw, 2 * N) - pow(tm, 2 * N))

    def window(
        self,
        *,
        N: int,
        W: int,
        device: Optional[torch.device] = None,
        sorted: bool = False,
    ) -> torch.Tensor:
        if type(N) is not int or N < 1:
            raise ValueError("window: N must be int >= 1 (complex dimension).")
        if type(W) is not int or W < 0:
            raise ValueError("window: W must be int >= 0.")
        if type(sorted) is not bool:
            raise ValueError("window: sorted must be bool.")

        D = 2 * N
        device = self._canonical_device(device=device)
        key = (D, W, self._device_key(device=device), bool(sorted))

        if self.enable_cache:
            cached = self._lru_get(
                lru=self._cache_window,
                key=key,
            )
            if cached is not None:
                return cached

        if W == 0:
            out = torch.zeros((1, D), dtype=self.dtype, device=device)
        else:
            out = self._cartesian_window_points(
                D=D,
                W=W,
                device=device,
                dtype=self.dtype,
            ).contiguous()

        if sorted:
            out = self._lexsort_rows(x=out)

        if self._should_cache(t=out):
            self._lru_put(lru=self._cache_window, key=key, value=out)

        return out

    def shell(
        self,
        *,
        N: int,
        W: int,
        device: Optional[torch.device] = None,
        sorted: bool = False,
    ) -> torch.Tensor:
        if type(N) is not int or N < 1:
            raise ValueError("shell: N must be int >= 1 (complex dimension).")
        if type(W) is not int or W < 0:
            raise ValueError("shell: W must be int >= 0.")
        if type(sorted) is not bool:
            raise ValueError("window: sorted must be bool.")

        D = 2 * N
        device = self._canonical_device(device=device)
        key = (D, W, self._device_key(device=device), bool(sorted))

        if self.enable_cache:
            cached = self._lru_get(lru=self._cache_shell, key=key)
            if cached is not None:
                return cached

        if W == 0:
            out = torch.zeros((1, D), dtype=self.dtype, device=device)
        else:
            out = self._shell_points(
                D=D,
                W=W,
                device=device,
                dtype=self.dtype,
            ).contiguous()

        if sorted:
            out = self._lexsort_rows(x=out)

        if self._should_cache(t=out):
            self._lru_put(lru=self._cache_shell, key=key, value=out)

        return out

    def iter_shells(
        self,
        *,
        N: int,
        start_radius: int = 0,
        max_radius: Optional[int] = None,
        device: Optional[torch.device] = None,
        sorted: bool = False,
    ) -> Generator[Tuple[int, torch.Tensor], None, None]:
        if type(N) is not int or N < 1:
            raise ValueError("iter_shells: N must be int >= 1 (complex dimension).")
        if type(start_radius) is not int or start_radius < 0:
            raise ValueError("iter_shells: start_radius must be int >= 0.")
        if max_radius is not None and (type(max_radius) is not int or max_radius < 0):
            raise ValueError("iter_shells: max_radius must be int >= 0 or None.")
        if type(sorted) is not bool:
            raise ValueError("window: sorted must be bool.")

        device = self._canonical_device(device=device)

        if max_radius is not None and start_radius > max_radius:
            return
            yield  # pragma: no cover

        if device.type == "cuda" and max_radius is not None:
            D = 2 * N
            win = self._cartesian_window_points(
                D=D,
                W=max_radius,
                device=device,
                dtype=self.dtype,
            )
            absmax = win.abs().amax(dim=1)
            for W in range(start_radius, max_radius + 1):
                m = absmax == W
                shell = win[m]

                if sorted:
                    shell = self._lexsort_rows(x=shell)

                yield W, shell.contiguous()
            return

        W = start_radius
        while True:
            yield W, self.shell(N=N, W=W, device=device, sorted=sorted)
            if max_radius is not None and W >= max_radius:
                break
            W += 1

    def pack_offsets(
        self,
        *,
        N: int,
        max_radius: int,
        device: Optional[torch.device] = None,
        sorted: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if type(N) is not int or N < 1:
            raise ValueError("pack_offsets: N must be int >= 1 (complex dimension).")
        if type(max_radius) is not int or max_radius < 0:
            raise ValueError("pack_offsets: max_radius must be int >= 0.")
        if type(sorted) is not bool:
            raise ValueError("window: sorted must be bool.")

        device = self._canonical_device(device=device)
        D = 2 * N

        shells: list[torch.Tensor] = []
        lengths: list[int] = []

        for W, S in self.iter_shells(
            N=N,
            start_radius=0,
            max_radius=max_radius,
            device=device,
            sorted=sorted,
        ):
            shells.append(S)
            lengths.append(S.shape[0])

        if not shells:
            return (
                torch.empty((0, D), dtype=self.dtype, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
            )

        offsets = torch.cat(shells, dim=0).contiguous()
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)

        return offsets, lengths_t

    def iter_packed(
        self,
        *,
        N: int,
        target_points_per_pack: int,
        start_radius: int = 0,
        max_radius: Optional[int] = None,
        device: Optional[torch.device] = None,
        sorted: bool = False,
    ) -> Generator[Tuple[int, int, torch.Tensor], None, None]:
        if type(N) is not int or N < 1:
            raise ValueError("iter_packed: N must be int >= 1 (complex dimension).")
        if type(target_points_per_pack) is not int or target_points_per_pack <= 0:
            raise ValueError("iter_packed: target_points_per_pack must be int > 0.")
        if type(start_radius) is not int or start_radius < 0:
            raise ValueError("iter_packed: start_radius must be int >= 0.")
        if max_radius is not None and (type(max_radius) is not int or max_radius < 0):
            raise ValueError("iter_packed: max_radius must be int >= 0 or None.")
        if type(sorted) is not bool:
            raise ValueError("window: sorted must be bool.")

        device = self._canonical_device(device=device)

        if max_radius is not None and start_radius > max_radius:
            return
            yield  # pragma: no cover

        acc: list[torch.Tensor] = []
        acc_count = 0
        w_start: Optional[int] = None
        w_last: Optional[int] = None

        for W, S in self.iter_shells(
            N=N,
            start_radius=start_radius,
            max_radius=max_radius,
            device=device,
            sorted=sorted,
        ):
            if acc_count > 0 and acc_count + S.shape[0] > target_points_per_pack:
                pack = torch.cat(acc, dim=0).contiguous()

                if sorted:
                    pack = self._lexsort_rows(x=pack)

                assert w_start is not None and w_last is not None
                yield (w_start, w_last, pack)
                acc.clear()
                acc_count = 0
                w_start = None
                w_last = None

            if w_start is None:
                w_start = W

            acc.append(S)
            acc_count += S.shape[0]
            w_last = W

        if acc_count > 0:
            pack = torch.cat(acc, dim=0).contiguous()

            if sorted:
                pack = self._lexsort_rows(x=pack)

            assert w_start is not None and w_last is not None
            yield (w_start, w_last, pack)

    @staticmethod
    def _cartesian_window_points(
        *,
        D: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if type(D) is not int or D < 1:
            raise ValueError("_cartesian_window_points: D must be int >= 1.")
        if type(W) is not int or W < 0:
            raise ValueError("_cartesian_window_points: W must be int >= 0.")
        if W == 0:
            return torch.zeros((1, D), dtype=dtype, device=device)

        axis = torch.arange(-W, W + 1, dtype=dtype, device=device)
        grids = torch.meshgrid(*([axis] * D), indexing="ij")

        return torch.stack([g.reshape(-1) for g in grids], dim=1).contiguous()

    @staticmethod
    def _shell_points(
        *,
        D: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
        threshold_CUDA: int = 1_000_000,
    ) -> torch.Tensor:
        if type(D) is not int or D < 1:
            raise ValueError("_shell_points: D must be int >= 1.")
        if type(W) is not int or W < 0:
            raise ValueError("_shell_points: W must be int >= 0.")
        if W == 0:
            return torch.zeros((1, D), dtype=dtype, device=device)
        if type(threshold_CUDA) is not int or threshold_CUDA < 0:
            raise ValueError("_shell_points: threshold_CUDA must be int >= 0.")

        use_mask = device.type == "cuda"

        if not use_mask:
            tw = (W << 1) + 1
            est_M = int(pow(tw, D))
            use_mask = est_M <= threshold_CUDA

        if use_mask:
            win = CPSFPeriodization._cartesian_window_points(
                D=D,
                W=W,
                device=device,
                dtype=dtype,
            )
            m = win.abs().amax(dim=1) == W
            return win[m].contiguous()

        axis = torch.arange(-W, W + 1, dtype=dtype, device=device)
        parts: list[torch.Tensor] = []

        for j in range(D):
            if D - 1 == 0:
                other = torch.empty((1, 0), dtype=dtype, device=device)
            else:
                other_axes = [axis for _ in range(D - 1)]
                other = torch.cartesian_prod(*other_axes)
                if other.dim() == 1:
                    other = other.view(-1, 1)

            if j == 0:
                left = torch.empty((other.shape[0], 0), dtype=dtype, device=device)
                right = other
            elif j == D - 1:
                left = other
                right = torch.empty((other.shape[0], 0), dtype=dtype, device=device)
            else:
                left = other[:, :j]
                right = other[:, j:]

            base_mask = (
                torch.ones((other.shape[0],), dtype=torch.bool, device=device)
                if left.numel() == 0
                else (left.abs().amax(dim=1) < W)
            )

            for s in (-1, 1):
                x = torch.empty((other.shape[0], D), dtype=dtype, device=device)

                if left.numel() > 0:
                    x[:, :j] = left

                x[:, j] = s * W

                if right.numel() > 0:
                    x[:, j + 1 :] = right

                x = x[base_mask]
                parts.append(x)

        if not parts:
            return torch.empty((0, D), dtype=dtype, device=device)

        return torch.cat(parts, dim=0).contiguous()

    @staticmethod
    def _canonical_device(
        *,
        device: Optional[torch.device],
    ) -> torch.device:
        return torch.device("cpu") if device is None else device

    @staticmethod
    def _device_key(
        *,
        device: torch.device,
    ) -> Tuple[str, int]:
        return (device.type, -1 if device.index is None else int(device.index))

    def _should_cache(
        self,
        *,
        t: torch.Tensor,
    ) -> bool:
        if not self.enable_cache or t.numel() == 0:
            return False

        est_bytes = int(t.numel()) * int(t.element_size())

        return est_bytes <= self.max_cache_bytes_per_tensor

    def _lru_get(
        self,
        *,
        lru: "OrderedDict[Tuple[int, int, Tuple[str, int]], torch.Tensor]",
        key: Tuple[int, int, Tuple[str, int]],
    ) -> Optional[torch.Tensor]:
        if not self.enable_cache:
            return None

        if key in lru:
            t = lru.pop(key)
            lru[key] = t
            return t

        return None

    def _lru_put(
        self,
        *,
        lru: "OrderedDict[Tuple[int, int, Tuple[str, int]], torch.Tensor]",
        key: Tuple[int, int, Tuple[str, int]],
        value: torch.Tensor,
    ) -> None:
        if not self.enable_cache:
            return

        lru[key] = value

        while len(lru) > self.max_cache_entries:
            lru.popitem(last=False)

    @staticmethod
    def _lexsort_rows(
        *,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0 or x.shape[0] <= 1:
            return x

        idx = torch.arange(x.shape[0], device=x.device)

        for k in range(x.shape[1] - 1, -1, -1):
            vals = x[idx, k]
            _, order = torch.sort(vals, stable=True)
            idx = idx[order]

        return x[idx]

    def cache_stats(
        self,
    ) -> dict:
        stats = {
            "enabled": bool(self.enable_cache),
            "max_entries_per_kind": int(self.max_cache_entries),
            "max_bytes_per_tensor": int(self.max_cache_bytes_per_tensor),
            "dtype": str(self.dtype),
            "window_entries": len(self._cache_window),
            "shell_entries": len(self._cache_shell),
        }

        return stats

    def cache_clear(
        self,
        *,
        clear_window: Optional[bool] = False,
        clear_shell: Optional[bool] = False,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is not None and not isinstance(device, torch.device):
            raise ValueError("cache_clear: device must be a torch.device or None.")
        if clear_window is not None and type(clear_window) is not bool:
            raise ValueError("cache_clear: clear_window must be bool or None.")
        if clear_shell is not None and type(clear_shell) is not bool:
            raise ValueError("cache_clear: clear_shell must be bool or None.")

        def _drop_for_device(lru, dev_key):
            if dev_key is None:
                lru.clear()
                return

            to_del = [
                k
                for k in lru.keys()
                if isinstance(k, tuple) and len(k) >= 3 and k[2] == dev_key
            ]
            for k in to_del:
                del lru[k]

        if device is None:
            dev_key = None
        else:
            dev_key = (device.type, -1 if device.index is None else int(device.index))

        clear_window = clear_window if clear_window is not None else False
        clear_shell = clear_shell if clear_shell is not None else False

        if clear_window:
            _drop_for_device(self._cache_window, dev_key)
        if clear_shell:
            _drop_for_device(self._cache_shell, dev_key)

    def set_cache_enabled(
        self,
        *,
        enabled: bool,
        drop_existing: bool = False,
    ) -> None:
        if type(enabled) is not bool:
            raise ValueError("set_cache_enabled: enabled must be bool.")
        if type(drop_existing) is not bool:
            raise ValueError("set_cache_enabled: drop_existing must be bool.")

        self.enable_cache = bool(enabled)
        if drop_existing:
            self._cache_window.clear()
            self._cache_shell.clear()

    def set_cache_limits(
        self,
        *,
        max_cache_entries_per_kind: Optional[int] = None,
        max_cache_bytes_per_tensor: Optional[int] = None,
        drop_excess_now: bool = True,
    ) -> None:
        if max_cache_entries_per_kind is not None:
            if (
                type(max_cache_entries_per_kind) is not int
                or max_cache_entries_per_kind < 0
            ):
                raise ValueError(
                    "set_cache_limits: max_cache_entries_per_kind must be int >= 0."
                )
            self.max_cache_entries = int(max_cache_entries_per_kind)

        if max_cache_bytes_per_tensor is not None:
            if (
                type(max_cache_bytes_per_tensor) is not int
                or max_cache_bytes_per_tensor < 0
            ):
                raise ValueError(
                    "set_cache_limits: max_cache_bytes_per_tensor must be int >= 0."
                )
            self.max_cache_bytes_per_tensor = int(max_cache_bytes_per_tensor)

        if drop_excess_now:
            while len(self._cache_window) > self.max_cache_entries:
                self._cache_window.popitem(last=False)
            while len(self._cache_shell) > self.max_cache_entries:
                self._cache_shell.popitem(last=False)
