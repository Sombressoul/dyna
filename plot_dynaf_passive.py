import torch
import matplotlib.pyplot as plt

from dynaf import DyNAFActivation

dynaf_activation = DyNAFActivation(
    passive=True,
    count_modes=7,
    features=1,
    expected_input_min=-10.0,
    expected_input_max=+10.0,
)
x = torch.linspace(-10, 10, 1000).unsqueeze(-1)

_, nonlinearity, components = dynaf_activation.forward(
    x, return_components=True, return_nonlinearity=True
)

plt.figure(figsize=(10, 10))
for i, component in enumerate(components):
    plt.plot(
        x.squeeze().numpy(),
        component.detach().squeeze().numpy(),
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
    nonlinearity.detach().squeeze().numpy(),
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
    (x * nonlinearity).detach().squeeze().numpy(),
    label="Transformed x",
)
plt.title("DyNAF Transformation")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
