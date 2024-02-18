import torch
import matplotlib.pyplot as plt

from dyna import ModulatedActivationSine, SignalModular


x = torch.linspace(-10, 10, 1000).unsqueeze(-1)
params = (
    torch.tensor(
        [
            (+1.50, +1.00, +1.00, -1.00),
            (+1.00, +1.00, +0.50, -0.50),
            (+0.50, +1.00, +0.00, +0.00),
            (+0.25, +1.00, -0.50, +0.50),
            (+0.10, +1.00, -1.00, +1.00),
        ]
    )
    .unsqueeze(-1)
    .unsqueeze(0)
)

signal = SignalModular(x=x, modes=params)
signal = ModulatedActivationSine(passive=True)(signal)

components = signal.components.permute([1, 0, 2])
plt.figure(figsize=(10, 10))
for i, component in enumerate(components):
    plt.plot(
        x.squeeze().numpy(),
        component.squeeze().numpy(),
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
    signal.nonlinearity.squeeze().numpy(),
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
    signal.x.detach().squeeze().numpy(),
    label="Transformed x",
)
plt.title("DyNA Transformation")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
