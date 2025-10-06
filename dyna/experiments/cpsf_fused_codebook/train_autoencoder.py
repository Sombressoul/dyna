# example run:
# > python -m dyna.experiments.cpsf_fused_codebook.train_autoencoder --data-root "e:\Datasets\Images_512x512\dataset_01\" --size 256 256 --epochs 1000 --batch 8 --grad_acc 1 --lr 1.0e-4 --device_target cuda --device_cache cpu --log-every 100 --out-dir ./temp

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import (
    Image,
    ImageDraw,
)
from pathlib import Path
from typing import Tuple
from torchvision.models import (
    vgg19,
    VGG19_Weights,
)

from dyna.experiments.cpsf_fused_codebook.cpsf_spectral_autoencoder import (
    CPSFSpectralAutoencoder,
)
from dyna.experiments.cpsf_fused_codebook.cpsf_memcell_autoencoder import (
    CPSFMemcellAutoencoder,
)
from dyna.experiments.cpsf_fused_codebook.local_image_dataset import LocalImageDataset


# -----------------------------
# Utilities
# -----------------------------
@torch.no_grad()
def save_side_by_side(
    x: torch.Tensor,
    y: torch.Tensor,
    save_path: Path,
    epoch: int,
    step: int,
    loss_val: float,
) -> None:
    x = x.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu()
    y = y.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu()

    x_img = Image.fromarray(x.permute(1, 2, 0).numpy(), mode="RGB")
    y_img = Image.fromarray(y.permute(1, 2, 0).numpy(), mode="RGB")

    H, W = x.shape[1], x.shape[2]
    canvas = Image.new("RGB", (W * 2, H), color=(0, 0, 0))
    canvas.paste(x_img, (0, 0))
    canvas.paste(y_img, (W, 0))

    draw = ImageDraw.Draw(canvas)
    bar_h = max(18, H // 24)
    draw.rectangle([(0, 0), (W * 2, bar_h)], fill=(0, 0, 0))
    text = f"epoch {epoch} | step {step} | loss {loss_val:.6f}"
    draw.text((6, 2), text, fill=(255, 255, 255))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, format="PNG")


class PerceptualLoss(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()

        weights = VGG19_Weights.IMAGENET1K_V1
        model = vgg19(weights=weights)

        vgg_features = model.features
        # conv_index == '22' > 8
        # conv_index == '54' > 35
        self.vgg = nn.Sequential(*list(vgg_features)[:35]).to(device).eval()

        for p in self.vgg.parameters():
            p.requires_grad_(False)

        mean, std = None, None
        w_t = weights.transforms()
        mean = getattr(w_t, "mean", None)
        std = getattr(w_t, "std", None)

        mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

        self._ema_alpha = 0.1
        self._scale_p = None
        self._scale_k = None

    def _pre(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return (x - self._mean) / self._std

    def _kl_flat(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "batchmean",
    ) -> torch.Tensor:
        B = pred.shape[0]
        log_q = pred.reshape(B, -1).log_softmax(dim=1)
        log_p = target.reshape(B, -1).log_softmax(dim=1)
        return F.kl_div(log_q, log_p, log_target=True, reduction=reduction)

    def _kl_per_channel(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "batchmean",
    ) -> torch.Tensor:
        B, C, H, W = pred.shape
        log_q = pred.permute(0, 2, 3, 1).reshape(B * H * W, C).log_softmax(dim=1)
        log_p = target.permute(0, 2, 3, 1).reshape(B * H * W, C).log_softmax(dim=1)
        return F.kl_div(log_q, log_p, log_target=True, reduction=reduction)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "batchmean",
        per_channel: bool = False,
    ) -> torch.Tensor:
        x = self.vgg(self._pre(pred))
        with torch.no_grad():
            y = self.vgg(self._pre(target))
        if per_channel:
            return self._kl_per_channel(x, y, reduction)
        else:
            return self._kl_flat(x, y, reduction)

    def combined_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        p_k_alpha: float = 0.5,
    ) -> torch.Tensor:
        eps = torch.finfo(pred.dtype).eps

        p_loss = self.forward(pred, target, reduction="batchmean", per_channel=True)
        k_loss = self._kl_flat(pred, target, reduction="batchmean")

        with torch.no_grad():
            target_scale = torch.maximum(p_loss.detach(), k_loss.detach())

            inst_scale_p = target_scale / (p_loss.detach() + eps)
            inst_scale_k = target_scale / (k_loss.detach() + eps)

            if self._scale_p is None:
                self._scale_p = inst_scale_p
                self._scale_k = inst_scale_k
            else:
                a = self._ema_alpha
                self._scale_p = (1 - a) * self._scale_p + a * inst_scale_p
                self._scale_k = (1 - a) * self._scale_k + a * inst_scale_k

        p_adj = p_loss * self._scale_p
        k_adj = k_loss * self._scale_k

        return torch.lerp(p_adj, k_adj, p_k_alpha)


# -----------------------------
# Training
# -----------------------------
def train(
    data_root: Path,
    out_dir: Path,
    size: Tuple[int, int],
    epochs: int,
    batch_size: int,
    lr: float,
    device_target: str,
    device_cache: str,
    log_every: int,
    model_type: str,
    grad_accumulation_steps: int = 1,
) -> None:
    device = torch.device(device_target)

    ds = LocalImageDataset(
        str(data_root), size=size, device_target=device, device_cache=device_cache
    )
    loader = ds.get_dataloader(batch_size=batch_size, shuffle=True, drop_last=True)

    if model_type == "spectral":
        model = CPSFSpectralAutoencoder().to(device)
    elif model_type == "memcell":
        model = CPSFMemcellAutoencoder().to(device)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    ploss = PerceptualLoss(device=device)
    loss_fn = lambda x, y: ploss.combined_loss(pred=x, target=y, p_k_alpha=0.75)

    out_dir = Path(out_dir)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)

    accum = max(1, int(grad_accumulation_steps))
    global_step = 0
    micro_count = 0
    loss_accum = 0.0
    opt.zero_grad(set_to_none=True)

    x_vis = None
    y_vis = None

    for epoch in range(1, epochs + 1):
        micro_count = 0
        loss_accum = 0.0
        for i, batch in enumerate(loader, start=1):
            x = batch
            y = model(x)
            loss_raw = loss_fn(y, x)
            loss = loss_raw / accum

            loss.backward()

            # print("\n\n========================\n\n")

            # def dbg_c_val(x: torch.Tensor, name: str):
            #     print(f"DEBUG '{name}':")
            #     print(f"\t{x.real.std()=}")
            #     print(f"\t{x.real.mean()=}")
            #     print(f"\t{x.real.min()=}")
            #     print(f"\t{x.real.max()=}")
            #     if x.grad is not None:
            #         if x.grad.data is not None:
            #             print(f"\tDEBUG '{name}' - GRAD:")
            #             print(f"\t\t{x.grad.data.std()=}")
            #             print(f"\t\t{x.grad.data.mean()=}")
            #             print(f"\t\t{x.grad.data.min()=}")
            #             print(f"\t\t{x.grad.data.max()=}")
            #     else:
            #         print(f"\tDEBUG '{name}' - !NO_GRAD!")

            # dbg_c_val(model.cell_0.alpha, "alpha")
            # dbg_c_val(model.cell_0.store.z_j, "z_j")
            # dbg_c_val(model.cell_0.store.vec_d_j, "vec_d_j")
            # dbg_c_val(model.cell_0.store.T_hat_j, "T_hat_j")
            # dbg_c_val(model.cell_0.store.alpha_j, "alpha_j")
            # dbg_c_val(model.cell_0.store.sigma_par, "sigma_par")
            # dbg_c_val(model.cell_0.store.sigma_perp, "sigma_perp")

            # exit()

            micro_count += 1
            loss_accum += float(loss_raw.detach().item())

            with torch.no_grad():
                x_vis = x[0].detach()
                model.train(False)
                y_vis = model(x[0].unsqueeze(0))[0].detach()
                model.train(True)

            if micro_count % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if log_every > 0 and (global_step % log_every == 0):
                    loss_avg = loss_accum / accum
                    preview_path = (
                        out_dir
                        / "previews"
                        / f"ep{epoch:03d}_step{global_step:06d}_loss{loss_avg:.4f}.png"
                    )
                    save_side_by_side(
                        x_vis, y_vis, preview_path, epoch, global_step, float(loss_avg)
                    )
                    print(
                        f"[ep {epoch}/{epochs} | step {global_step}] loss={loss_avg:.6f} → {preview_path.name}"
                    )
                loss_accum = 0.0

        remainder = micro_count % accum
        if remainder != 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            if log_every > 0 and (global_step % log_every == 0):
                loss_avg = loss_accum / remainder
                preview_path = (
                    out_dir
                    / "previews"
                    / f"ep{epoch:03d}_step{global_step:06d}_loss{loss_avg:.4f}.png"
                )
                save_side_by_side(
                    x_vis, y_vis, preview_path, epoch, global_step, float(loss_avg)
                )
                print(
                    f"[ep {epoch}/{epochs} | step {global_step}] loss={loss_avg:.6f} → {preview_path.name}"
                )
            loss_accum = 0.0

    with torch.no_grad():
        for batch in ds.iter_batches(batch_size):
            x = batch
            model.train(False)
            y = model(x)
            model.train(True)
            preview_path = (
                out_dir
                / "previews"
                / f"final_ep{epochs:03d}_step{global_step:06d}_loss{loss_fn(y,x).item():.4f}.png"
            )
            save_side_by_side(
                x[0],
                y[0],
                preview_path,
                epochs,
                global_step,
                float(loss_fn(y, x).item()),
            )
            break


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train classic autoencoder with step-wise previews.",
    )
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Directory with images",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="./runs/ae_baseline",
        help="Output directory for previews",
    )
    p.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Target size H W",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1,
    )
    p.add_argument(
        "--grad_acc",
        type=int,
        default=16,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    p.add_argument(
        "--device_target",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    p.add_argument(
        "--device_cache",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Log/save preview every N optimizer steps",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default=None,
    )
    args = p.parse_args()

    train(
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        size=(args.size[0], args.size[1]),
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device_target=args.device_target,
        device_cache=args.device_cache,
        log_every=args.log_every,
        grad_accumulation_steps=args.grad_acc,
        model_type=args.model_type,
    )
