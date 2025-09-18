import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class LocalImageDataset(Dataset):
    def __init__(
        self,
        root: str | os.PathLike,
        size: Tuple[int, int],
        device_target: Optional[torch.device | str] = None,
        device_cache: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"LocalImageDataset: root is not a directory: {self.root}")

        self.size = tuple(size)
        if len(self.size) != 2 or not all(
            isinstance(x, int) and x > 0 for x in self.size
        ):
            raise ValueError(
                f"LocalImageDataset: size must be (H, W) with positive ints, got {self.size}"
            )

        self.device_target = torch.device(device_target) if device_target is not None else None
        self.device_cache = torch.device(device_cache) if device_cache is not None else None
        self.dtype = dtype

        files = [
            p
            for p in sorted(self.root.iterdir())
            if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()
        ]
        if not files:
            raise ValueError(
                f"LocalImageDataset: no supported images found in {self.root}"
            )
        self.files: List[Path] = files

        self._cache: List[Optional[torch.Tensor]] = [None] * len(self.files)

    def __len__(
        self,
    ) -> int:
        return len(self.files)

    def _load_to_uint8tensor(
        self,
        path: Path,
    ) -> torch.Tensor:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            H, W = self.size
            im = im.resize((W, H), resample=Image.BICUBIC)
            arr = np.asarray(im, dtype=np.uint8)

        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        t = t.to(torch.uint8)
        if self.device_cache is not None:
            t = t.to(self.device_cache, non_blocking=True)

        return t

    def __getitem__(
        self,
        idx: int,
    ) -> torch.Tensor:
        if idx < 0 or idx >= len(self.files):
            raise IndexError(idx)
        
        return_t = lambda t: t.to(device=self.device_target, dtype=self.dtype).div(255.0)

        cached = self._cache[idx]
        if cached is not None:
            return return_t(cached)

        t = self._load_to_uint8tensor(self.files[idx])
        self._cache[idx] = t

        return return_t(t)

    def clear_cache(
        self,
    ) -> None:
        self._cache = [None] * len(self.files)

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> DataLoader:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=0,
            pin_memory=False,
        )

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        loader = self.get_dataloader(
            batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
        for batch in loader:
            yield batch


# -----------------------------
# Quick sanity test
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Directory with images")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], help="H W")
    parser.add_argument(
        "--device", type=str, default=None, help="cpu | cuda | cuda:0 etc."
    )
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    ds = LocalImageDataset(args.root, size=tuple(args.size), device_target=args.device)
    print(f"Found {len(ds)} images in {args.root}")

    x0 = ds[0]
    print("single sample:", tuple(x0.shape), x0.dtype, x0.device)

    for i, batch in enumerate(ds.iter_batches(args.batch)):
        print(f"batch {i}:", tuple(batch.shape), batch.dtype, batch.device)
        if i >= 1:
            break
