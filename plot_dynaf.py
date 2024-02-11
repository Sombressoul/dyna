import torch
import matplotlib.pyplot as plt

from dynaf import DyNAFActivation

dynaf_activation = DyNAFActivation(passive=False)
x = torch.linspace(-10, 10, 1000).unsqueeze(-1)  # Adding feature dimension

params = torch.tensor(
    [
        (+1.00, +0.50, +1.00, +0.00),
        (+0.15, +3.00, +5.50, +0.00),
        (+0.20, +6.50, +0.25, +4.50),
        (-1.55, +4.50, +0.15, -2.50),
    ]
).unsqueeze(-1)

_, nonlinearity, components = dynaf_activation.forward(
    x, modes=params, return_components=True, return_nonlinearity=True
)

plt.figure(figsize=(10, 10))
for i, component in enumerate(components):
    plt.plot(
        x.squeeze().numpy(),
        component.squeeze().numpy(),
        label=f"Set {i}",
    )
plt.title("DyNAF Components")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(
    x.squeeze().numpy(),
    nonlinearity.squeeze().numpy(),
    label="Resulting waveform",
)
plt.title("DyNAF Nonlinearity")
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
    (x * nonlinearity).squeeze().numpy(),
    label="Transformed x",
)
plt.title("DyNAF Transformation")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
