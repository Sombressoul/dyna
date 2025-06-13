import torch

class MinimalHPM(torch.nn.Module):
    def __init__(
        self, 
        shape: list[int] = [128, 128, 128],
        channels: int = 16,
        tau: float = 2.0,
        sigma: float = 0.5,
    ) -> None:
        super().__init__()
        self.memory = torch.nn.Parameter(torch.zeros(*shape, channels))
        self.sigma = sigma
        self.tau = tau

        grid = torch.stack(torch.meshgrid(
            torch.arange(shape[0]),
            torch.arange(shape[1]),
            torch.arange(shape[2]),
            indexing='ij'
        ), dim=-1).float()
        self.register_buffer("grid", grid)

    def kernel(self, x, ray_origin, ray_dir):
        dx = x - ray_origin
        t = (dx * ray_dir).sum(-1)
        x_proj = ray_origin + t[..., None] * ray_dir
        r2 = ((x - x_proj) ** 2).sum(-1)
        kernel = torch.exp(-r2 / (2 * self.sigma ** 2)) * torch.exp(-t.clamp(min=0) / self.tau)
        return kernel

    def read(self, ray_origin, ray_dir):
        B = ray_origin.shape[0]
        flat_grid = self.grid.view(-1, 3)
        mem = self.memory.view(-1, self.memory.shape[-1])
        out = []
        for b in range(B):
            k = self.kernel(flat_grid, ray_origin[b], ray_dir[b])
            t = (mem * k[:, None]).sum(0)
            out.append(t)
        return torch.stack(out, dim=0)

    def write(self, ray_origin, ray_dir, delta, alpha=1.0):
        B = ray_origin.shape[0]
        flat_grid = self.grid.view(-1, 3)
        mem = self.memory.view(-1, self.memory.shape[-1])
        for b in range(B):
            k = self.kernel(flat_grid, ray_origin[b], ray_dir[b])
            update = alpha * delta[b][None, :] * k[:, None]
            mem.data += update
