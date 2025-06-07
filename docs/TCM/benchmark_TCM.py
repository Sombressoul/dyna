import torch
import time
import os
import sys

# ========================================================= #

# Experiment params
seed = 1337
target_count = 256
steps = 1000
batch_size = 32
dtype = torch.float32

# Model(/data) params
output_shape = [512, 512]
context_length = 256
n_subspaces = 32
rank_subspace = 32
rank_transformations = 32

# Data generation params
x_mean = 0.0
x_std = 1.0

# Training params
log_step = 10
lr = 1.0e-3
wd = 1.0e-2
device = "cuda"

# ========================================================= #

dir_script = os.path.dirname(os.path.abspath(__file__))
dir_docs = os.path.dirname(dir_script)
dir_project = os.path.dirname(dir_docs)
sys.path.append(dir_project)

from dyna.lib.tensor_composer_mobius import TensorComposerMobius

torch.manual_seed(seed)

class Net(torch.nn.Module):
    def __init__(
        self,
        output_shape: list[int],
        context_length: int,
        n_subspaces: int,
        rank_subspace: int,
        rank_transformations: int,
        dtype_weights: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        
        self.wl = TensorComposerMobius(
            output_shape=output_shape,
            context_length=context_length,
            n_subspaces=n_subspaces,
            rank_subspace=rank_subspace,
            rank_transformations=rank_transformations,
            dtype_weights=dtype_weights,
        )

        pass

    def forward(self, x) -> torch.Tensor:
        return self.wl(x)

model_cfg = dict(
    output_shape = output_shape,
    context_length = context_length,
    n_subspaces = n_subspaces,
    rank_subspace = rank_subspace,
    rank_transformations = rank_transformations,
    dtype_weights = dtype,
)
model = Net(**model_cfg)
model = model.to(device=device)
loss_fn = torch.nn.functional.mse_loss
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, amsgrad=False)
targets_x = torch.nn.init.normal_(tensor=torch.empty([target_count, context_length], dtype=dtype, device=device), mean=x_mean, std=x_std)
targets_y = torch.randn([target_count, *output_shape], dtype=dtype, device=device, requires_grad=False)

start = time.perf_counter()

accum_steps = target_count // batch_size
assert target_count % batch_size == 0, "target_count must be divisible by batch_size"

for step in range(steps):
    optim.zero_grad()
    total_loss = 0.0

    for i in range(accum_steps):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        batch_x = targets_x[start_idx:end_idx]
        batch_y = targets_y[start_idx:end_idx]

        loss = loss_fn(model(batch_x), batch_y) / accum_steps  # scale loss for accumulation
        loss.backward()
        total_loss += loss.item()

    optim.step()

    if step % log_step == 0:
        grads = [p.grad.abs().max() if p.grad is not None else torch.tensor(0.0) for p in model.parameters()]
        print(f"Step #{step:0>6d} loss: {total_loss:.7f}, max grad: {max(grads).item():.7f}")

        with torch.no_grad():
            out_weights = model(targets_x[0:batch_size])  # [B, H, W]
            out_weights = out_weights.cpu()
            rank = torch.linalg.matrix_rank(out_weights.reshape(target_count, -1).to(torch.float32))
            std = out_weights.std().item()
            sim = torch.nn.functional.cosine_similarity(out_weights[0].flatten(), out_weights[1].flatten(), dim=0)
            print(f"Similarity between w[0] and w[1]: {sim.item():.4f}")
            print(f"Step #{step:0>6d} rank: {rank}, std: {std:.4f}")

        print("\n---\n")

end = time.perf_counter()
elapsed_time = end - start
print(f"Execution time: {elapsed_time} seconds")
