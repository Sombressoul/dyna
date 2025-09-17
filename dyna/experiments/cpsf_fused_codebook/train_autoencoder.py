# run:
# > python -m dyna.experiments.cpsf_fused_codebook.train_autoencoder --data-root "..\!datasets\Img_512-512_4096_01\" --size 256 256 --epochs 50 --batch 1 --grad_acc 8 --lr 1e-3 --device cuda --log-every 10 --out-dir ./temp

from pathlib import Path
import argparse
from typing import Tuple

import torch
import torch.nn as nn

from PIL import Image, ImageDraw

from dyna.experiments.cpsf_fused_codebook.cpsf_spectral_autoencoder import (
    CPSFSpectralAutoencoder,
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
    device_str: str,
    log_every: int,
    grad_accumulation_steps: int = 1,
) -> None:
    device = torch.device(device_str)

    ds = LocalImageDataset(str(data_root), size=size, device=device)
    loader = ds.get_dataloader(batch_size=batch_size, shuffle=True, drop_last=True)

    model = CPSFSpectralAutoencoder().to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

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
            micro_count += 1
            loss_accum += float(loss_raw.detach().item())
            x_vis = x[0].detach()
            y_vis = y[0].detach()

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
            y = model(x)
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
        default=[256, 256],
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
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log/save preview every N optimizer steps",
    )
    args = p.parse_args()

    train(
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        size=(args.size[0], args.size[1]),
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device_str=args.device,
        log_every=args.log_every,
        grad_accumulation_steps=args.grad_acc,
    )
