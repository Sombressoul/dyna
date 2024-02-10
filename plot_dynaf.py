import torch
import numpy as np
import math
import matplotlib.pyplot as plt


def DyNAF(x, a, b, g, d):
    return a * (
        (1.0 / (1 + math.e ** (math.fabs(b) * (x - d - math.fabs(g)))))
        - (1.0 / (1 + math.e ** (math.fabs(b) * (x - d + math.fabs(g)))))
    )


x = torch.linspace(-10, 10, 1000)

params = [
    (+1.00, +0.50, +1.00, +0.00),
    (+0.15, +3.00, +5.50, +0.00),
    (+0.20, +6.50, +0.25, +4.50),
    (-1.55, +4.50, +0.15, -2.50),
]

plt.figure(figsize=(10, 10))
for i, (a, b, g, d) in enumerate(params, start=1):
    y = DyNAF(x, a, b, g, d)
    plt.plot(x.numpy(), y.numpy(), label=f"Set {i}: a={a}, b={b}, g={g}, d={d}")

plt.title("DyNAF Components")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

y = torch.zeros_like(x)
for i, (a, b, g, d) in enumerate(params, start=1):
    y = y + DyNAF(x, a, b, g, d)
y = y + 1.0

plt.figure(figsize=(10, 10))
plt.plot(x.numpy(), y.numpy(), label=f"Resulting waveform")
plt.title("DyNAF Nonlinearity")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 10))
plt.plot(x.numpy(), x.numpy(), label=f"Original x")
plt.plot(x.numpy(), (x * y).numpy(), label=f"Transformed x")
plt.title("DyNAF Transformation")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
