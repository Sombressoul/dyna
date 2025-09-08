import torch

from collections import OrderedDict
from typing import (
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
    MutableMapping,
    OrderedDict as TypingOrderedDict,
)

from dyna.lib.cpsf.structures import CPSFPeriodizationKind

TypeLRUCacheKey = Tuple[int, int, torch.device]
TypeLRUCacheValue = torch.Tensor
TypeLRUCache = TypingOrderedDict[TypeLRUCacheKey, TypeLRUCacheValue]


class CPSFPeriodization:
    def __init__(
        self,
        kind: CPSFPeriodizationKind,
        window: Optional[Union[int, torch.LongTensor]] = None,
        max_radius: Optional[Union[int, torch.LongTensor]] = None,
        cache_active: bool = True,
        cache_limit: int = 32,
        cache_soft_limit_bytes: int = 128 * 1024 * 1024,
    ):
        if not isinstance(kind, CPSFPeriodizationKind):
            raise TypeError(
                f"CPSFPeriodization: 'kind' must be CPSFPeriodizationKind, got {type(kind)}"
            )

        def _raise_tinfo(x, name: str, target_type: str):
            if isinstance(x, torch.Tensor):
                tinfo = f"type={type(x)}, dtype={x.dtype}, shape={tuple(x.shape)}, numel={x.numel()}"
            else:
                tinfo = f"type={type(x)}"
            raise TypeError(
                "".join(
                    [
                        f"CPSFPeriodization: '{name}' must be '{target_type}' scalar.",
                        f"Got: {tinfo}",
                    ]
                )
            )

        def _to_int_scalar(x, name: str):
            if x is None:
                return None
            if isinstance(x, int) and not isinstance(x, bool):
                return int(x)
            if (
                isinstance(x, torch.Tensor)
                and x.dtype != torch.bool
                and x.numel() == 1
                and not torch.is_floating_point(x)
                and not torch.is_complex(x)
            ):
                return int(x.item())
            _raise_tinfo(x, name, "int")

        normal_window = _to_int_scalar(window, "window")
        normal_max_radius = _to_int_scalar(max_radius, "max_radius")

        if kind is CPSFPeriodizationKind.WINDOW:
            if normal_window is None:
                raise ValueError("CPSFPeriodization(WINDOW): 'window' is required")
            if normal_window < 1:
                raise ValueError("CPSFPeriodization(WINDOW): 'window' must be >= 1")
            if normal_max_radius is not None:
                raise ValueError("CPSFPeriodization(WINDOW): 'max_radius' must be None")
        elif kind is CPSFPeriodizationKind.FULL:
            if normal_window is not None:
                raise ValueError("CPSFPeriodization(FULL): 'window' must be None")
            if normal_max_radius is not None and normal_max_radius < 1:
                raise ValueError("CPSFPeriodization(FULL): 'max_radius' must be >= 1")
        else:
            raise ValueError(f"CPSFPeriodization: unsupported kind={kind}")

        self.kind: CPSFPeriodizationKind = kind
        self.window: Union[int, None] = normal_window
        self.max_radius: Union[int, None] = normal_max_radius
        self._cache_active: bool = bool(cache_active)
        self._cache_limit: int = int(cache_limit)
        self._cache_soft_limit_bytes: int = int(cache_soft_limit_bytes)
        self._cache_shell: TypeLRUCache = OrderedDict()
        self._cache_window: TypeLRUCache = OrderedDict()

    @staticmethod
    def _canon_device(
        dev: Union[torch.device, str],
    ) -> torch.device:
        d = torch.device(dev)
        if d.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CPSFPeriodization: CUDA device requested but torch.cuda.is_available() is False"
                )
            if d.index is None:
                return torch.device("cuda", torch.cuda.current_device())
            return d
        return d

    def _lru_get(
        self,
        cache: MutableMapping[TypeLRUCacheKey, TypeLRUCacheValue],
        key: TypeLRUCacheKey,
    ) -> Optional[TypeLRUCacheValue]:
        if not self._cache_active:
            return None

        t = cache.get(key)
        if t is not None and isinstance(cache, OrderedDict):
            cache.move_to_end(key)

        return t

    def _lru_put(
        self,
        cache: MutableMapping[TypeLRUCacheKey, TypeLRUCacheValue],
        key: TypeLRUCacheKey,
        value: TypeLRUCacheValue,
    ) -> None:
        if not self._cache_active:
            return None

        cache[key] = value
        if isinstance(cache, OrderedDict):
            cache.move_to_end(key)
            while len(cache) > self._cache_limit:
                cache.popitem(last=False)

    def _should_cache(
        self,
        num_points: int,
        N: int,
    ) -> bool:
        if not self._cache_active:
            return False

        est_bytes = num_points * N * 8
        return est_bytes <= self._cache_soft_limit_bytes

    def _cartesian_window_points(
        self,
        N: int,
        W: int,
        device: Union[torch.device, str],
    ) -> torch.Tensor:
        if W < 0:
            raise ValueError(f"_cartesian_window_points: W must be >= 0, got {W}")
        if N < 2:
            raise ValueError(f"_cartesian_window_points: N must be >= 2, got {N}")

        dev = self._canon_device(device)
        key = (N, W, dev)
        cached = self._lru_get(self._cache_window, key)
        if cached is not None:
            return cached

        if W == 0:
            grid = torch.zeros(1, N, dtype=torch.long, device=dev)
        else:
            axis = torch.arange(-W, W + 1, device=dev, dtype=torch.long)
            axes = [axis] * N
            grid = torch.cartesian_prod(*axes)

        M = (2 * W + 1) ** N
        if self._should_cache(M, N):
            self._lru_put(self._cache_window, key, grid)

        return grid

    def _shell_points(
        self,
        N: int,
        W: int,
        device: Union[torch.device, str],
    ) -> torch.Tensor:
        if W < 0:
            raise ValueError(f"_shell_points: W must be >= 0, got {W}")
        if N < 2:
            raise ValueError(f"_shell_points: N must be >= 2, got {N}")

        dev = self._canon_device(device)
        key = (N, W, dev)
        cached = self._lru_get(self._cache_shell, key)
        if cached is not None:
            return cached

        if W == 0:
            shell = torch.zeros(1, N, dtype=torch.long, device=dev)
        else:
            full = torch.arange(-W, W + 1, device=dev, dtype=torch.long)
            interior = torch.arange(-W + 1, W, device=dev, dtype=torch.long)

            def _multi_cartesian(rng: torch.Tensor, repeat: int) -> torch.Tensor:
                if repeat <= 0:
                    return torch.zeros(1, 0, dtype=torch.long, device=dev)
                axes = [rng] * repeat
                return torch.cartesian_prod(*axes)

            parts = []
            for j in range(N):
                pre = _multi_cartesian(interior, j)
                post = _multi_cartesian(full, N - j - 1)

                P = pre.shape[0]
                Q = post.shape[0]
                count = P * Q

                if j > 0:
                    left = pre.repeat_interleave(Q, dim=0)
                else:
                    left = torch.zeros(count, 0, dtype=torch.long, device=dev)

                if N - j - 1 > 0:
                    right = post.repeat(P, 1)
                else:
                    right = torch.zeros(count, 0, dtype=torch.long, device=dev)

                col_neg = torch.full((count, 1), -W, dtype=torch.long, device=dev)
                col_pos = torch.full((count, 1), W, dtype=torch.long, device=dev)

                parts.append(torch.cat([left, col_neg, right], dim=1))
                parts.append(torch.cat([left, col_pos, right], dim=1))

            shell = torch.cat(parts, dim=0)

        M_w = (2 * W + 1) ** N - (2 * W - 1) ** N if W >= 1 else 1

        if self._should_cache(M_w, N):
            self._lru_put(self._cache_shell, key, shell)

        return shell

    def iter_offsets(
        self,
        N: int,
        device: Union[torch.device, str],
    ) -> Iterator[torch.Tensor]:
        if not isinstance(N, int) or isinstance(N, bool) or N < 2:
            raise ValueError(
                f"CPSFPeriodization.iter_offsets: N must be an integer >= 2, got {N}"
            )

        device_c = self._canon_device(device)

        if self.kind is CPSFPeriodizationKind.WINDOW:
            yield self._cartesian_window_points(N, self.window, device=device_c)
            return

        if self.kind is CPSFPeriodizationKind.FULL:
            W = 0
            while True:
                yield self._shell_points(N, W, device=device_c)
                W += 1
                if self.max_radius is not None and W > self.max_radius:
                    break
            return

        raise ValueError(
            f"CPSFPeriodization.iter_offsets: unsupported kind={self.kind}"
        )

    def pack_offsets(
        self,
        N: int,
        device: Union[torch.device, str],
        radii: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(N, int) or isinstance(N, bool) or N < 2:
            raise ValueError(
                f"CPSFPeriodization.pack_offsets: N must be an integer >= 2, got {N}"
            )

        device_c = self._canon_device(device)
        lens: list[int] = []
        chunks: list[torch.Tensor] = []

        for W in radii:
            if not isinstance(W, int) or W < 0:
                raise ValueError(
                    f"pack_offsets: radii must be non-negative ints, got {W}"
                )
            sh = self._shell_points(N, W, device=device_c)
            lens.append(int(sh.shape[0]))
            chunks.append(sh)

        if not chunks:
            return (
                torch.empty(0, N, dtype=torch.long, device=device_c),
                torch.empty(0, dtype=torch.long, device=device_c),
            )

        offsets = torch.cat(chunks, dim=0)
        lengths = torch.as_tensor(lens, dtype=torch.long, device=device_c)

        return offsets, lengths

    def iter_packed_offsets(
        self,
        N: int,
        device: Union[torch.device, str],
        pack_size: int,
        start_radius: int = 0,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
        if not isinstance(N, int) or isinstance(N, bool) or N < 2:
            raise ValueError(
                f"CPSFPeriodization.iter_packed_offsets: N must be an integer >= 2, got {N}"
            )
        if not isinstance(pack_size, int) or pack_size <= 0:
            raise ValueError("iter_packed_offsets: pack_size must be positive int")
        if not isinstance(start_radius, int) or start_radius < 0:
            raise ValueError(
                "iter_packed_offsets: start_radius must be non-negative int"
            )

        device_c = self._canon_device(device)
        W = start_radius
        while True:
            if self.max_radius is not None and W > self.max_radius:
                break

            radii: list[int] = []

            for _ in range(pack_size):
                if self.max_radius is not None and W > self.max_radius:
                    break
                radii.append(W)
                W += 1

            if not radii:
                break

            offsets, lengths = self.pack_offsets(N, device_c, radii)
            yield offsets, lengths, radii[0]

            if self.max_radius is not None and W > self.max_radius:
                break
