import torch
import math


class MemoryField:
    def __init__(
        self,
        shape: list[int],
        channels: int,
        voxel_size: float = 1.0,
        init_mean: float = 0.0,
        init_std: float = 1.0e-2,
        device: torch.device = "cpu",
        dtype_weights: torch.dtype = torch.float16,
    ) -> None:

        self.shape = shape
        self.channels = channels
        self.voxel_size = voxel_size
        self.init_mean = init_mean
        self.init_std = init_std
        self.device = device
        self.dtype_weights = dtype_weights

        self.n_dim = len(self.shape)
        self.id_max = math.prod(self.shape) - 1

        self.data = torch.empty(self.shape, device=self.device, dtype=self.dtype_weights)
        self.data = self.reset()

        pass

    def get(
        self,
        coords: torch.Tensor,
        normalized: bool = False,
    ) -> torch.Tensor:
        ids = self.c2i_f(coords) if normalized else self.c2i_i(coords)
        return self.data.view(-1)[ids]

    def set(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        normalized: bool = False,
    ) -> None:
        ids = self.c2i_f(coords) if normalized else self.c2i_i(coords)
        self.data.view(-1)[ids] = values
        return

    def get_all(
        self,
    ) -> torch.Tensor:
        return self.data

    def zero_(
        self,
    ) -> torch.Tensor:
        self.data.zero_()
        return self.data

    def reset(self) -> torch.Tensor:
        torch.nn.init.normal_(self.data, mean=self.init_mean, std=self.init_std)
        return self.data

    def c2i_i(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        assert coords.shape[1] == self.n_dim

        strides = torch.tensor(
            [math.prod(self.shape[i + 1:]) for i in range(self.n_dim)],
            device=coords.device
        )
        ids = (coords * strides).sum(dim=-1)

        return ids

    def i2c_i(
        self,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        assert torch.all(ids.le(self.id_max)), f"Expected max id to be <= {self.id_max}, got {ids.max().item()}"
        assert torch.all(ids.ge(0)), f"Expected ids to be >= 0, got {ids.min().item()}"

        coords = torch.empty([ids.shape[0], self.n_dim], dtype=torch.long, device=ids.device)
        remaining = ids.clone()

        for i in range(self.n_dim):
            div = math.prod(self.shape[i + 1:]) if i + 1 < self.n_dim else 1
            coords[:, i] = remaining // div
            remaining = remaining % div

        return coords

    def c2i_f(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        assert coords.shape[1] == self.n_dim
        assert torch.all((coords >= 0.0) & (coords <= 1.0)), "Coordinates must be in [0.0, 1.0]"

        scaled = torch.clamp(
            coords * torch.tensor(self.shape, device=coords.device),
            max=torch.tensor(self.shape, device=coords.device) - 1,
        )
        discrete = scaled.long()
        coords = self.c2i_i(discrete)

        return coords

    def i2c_f(
        self,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        discrete = self.i2c_i(ids)
        shape_tensor = torch.tensor(self.shape, dtype=torch.float32, device=discrete.device)
        coords_f = (discrete + 0.5) / shape_tensor
        return coords_f
