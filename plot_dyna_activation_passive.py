import torch
import matplotlib.pyplot as plt
import argparse

from dyna import ModulatedActivation

parser = argparse.ArgumentParser(description="evaluation")
parser.add_argument(
    "--count-modes",
    type=int,
    default=7,
    metavar="N",
    help="wave modes count (default: 7)",
)
parser.add_argument(
    "--dyn-range",
    type=float,
    default=7.5,
    metavar="N",
    help="dynamic range (default: 7.5)",
)
args = parser.parse_args()

x = torch.linspace(-10, 10, 1000).unsqueeze(-1)

signal = ModulatedActivation(
    passive=True,
    count_modes=args.count_modes,
    features=1,
    theta_dynamic_range=args.dyn_range,
)(x)

components = signal.components.permute([1, 0, 2])
plt.figure(figsize=(10, 10))
for i, component in enumerate(components):
    plt.plot(
        x.squeeze().numpy(),
        component.detach().squeeze().numpy(),
        label=f"Set {i}",
    )
plt.title("DyNA Components")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(
    x.squeeze().numpy(),
    signal.nonlinearity.detach().squeeze().numpy(),
    label="Resulting waveform",
)
plt.title("DyNA Nonlinearity")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(
    x.squeeze().numpy(),
    x.squeeze().numpy(),
    label="Original x",
)
plt.plot(
    x.squeeze().numpy(),
    (x * signal.nonlinearity).detach().squeeze().numpy(),
    label="Transformed x",
)
plt.title("DyNA Transformation")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()