# Originated from: https://github.com/pytorch/examples/blob/main/mnist/main.py
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from models import CIFAR100DyNAActivationPerFeature

data_path = f"{project_dir}/data"


####
def train(model, device, train_loader, optimizer, epoch, args):
    loss_accumulator = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        loss_accumulator.append(loss.item())
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return loss_accumulator


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-3,
        metavar="LR",
        help="learning rate (default: 1.0e-3)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1.0e-2,
        metavar="WD",
        help="Weight decay (default: 1.0e-2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--plot-loss",
        default=False,
        action="store_true",
        help="plot training loss (default: False) (requires self-projection package)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.CIFAR100(
        data_path,
        train=True,
        download=True,
        transform=transform,
    )
    dataset_test = datasets.CIFAR100(
        data_path,
        train=False,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = CIFAR100DyNAActivationPerFeature().to(device)

    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("\n")
    print("# ================= MODEL INFO ================= #")
    print(f"Model class name: {model.__class__.__name__}")
    print(f"Total number of trainable parameters: {total_trainable_params}")
    print("# =================############================= #")
    print("\n")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    loss_accumulator = []
    for epoch in range(1, args.epochs + 1):
        loss_accumulator = loss_accumulator + train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            args,
        )
        test(model, device, test_loader)

    if args.plot_loss:
        try:
            from self_projection.utils.functional import plot_loss

            plot_loss(loss_accumulator)
        except ImportError:
            print(
                "Could not import plot_loss. Please install self_projection to plot loss."
            )
            pass


if __name__ == "__main__":
    main()
