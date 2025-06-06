import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#
# The following initial params:
#
#   K = 4
#   d_k = 32
#   d_prime = 2048
#   n_trials = 10000
#   xy_correlation = 0.5
#   torch.manual_seed(42)
#
# Should give the following result:
#   CSP (K=4) — Canonical
#   RMSE=14323.492584319598, Bias=15.375518473648606, Pearson=0.9487576105841762
#   NRMSE=0.21759386061901784, Avg. bias=1.0002335756034362
#
# Note: canonical CSP produces high-variance estimates when inputs are weakly correlated,
#       due to multiplicative amplification of unaligned spectral noise.
#

K = 4
d_k = 32
d_prime = 2048
n_trials = 10000
xy_correlation = 0.5
torch.manual_seed(42)

xs = [torch.randn(n_trials, d_k) for _ in range(K)]
ys = [torch.lerp(torch.randn_like(x), x, xy_correlation) for x in xs]

def generate_hash_and_sign(d_k, d_prime):
    h = torch.randint(0, d_prime, (d_k,))
    s = torch.randint(0, 2, (d_k,)) * 2 - 1
    return h, s

def count_sketch(x, h, s, d_prime):
    sketch = torch.zeros(d_prime)
    for i in range(len(x)):
        sketch[h[i]] += s[i] * x[i]
    return sketch

def unitary_fft(u):
    return torch.fft.fft(u, norm='ortho')

def unitary_ifft(u):
    return torch.fft.ifft(u, norm='ortho')

dot_estimates = []
ground_truths = []

for n in range(n_trials):
    Gx = torch.ones(d_prime, dtype=torch.cfloat)
    Gy = torch.ones(d_prime, dtype=torch.cfloat)

    for k in range(K):
        h, s = generate_hash_and_sign(d_k, d_prime)
        cs_x = count_sketch(xs[k][n], h, s, d_prime)
        cs_y = count_sketch(ys[k][n], h, s, d_prime)

        fx = unitary_fft(cs_x)
        fy = unitary_fft(cs_y)

        Gx *= fx
        Gy *= fy

    phi_x = unitary_ifft(Gx)
    phi_y = unitary_ifft(Gy)
    dot = (phi_x.conj() * phi_y).sum().real * (d_prime ** (K-1))
    dot_estimates.append(dot.item())

    truth = 1.0
    for k in range(K):
        truth *= torch.dot(xs[k][n], ys[k][n]).item()
    ground_truths.append(truth)

# Анализ
dot_estimates = np.array(dot_estimates)
ground_truths = np.array(ground_truths)
rmse = np.sqrt(np.mean((dot_estimates - ground_truths)**2))
bias = np.mean(dot_estimates - ground_truths)
pearson = pearsonr(dot_estimates, ground_truths)[0]
avg = np.mean(dot_estimates)
true_avg = np.mean(ground_truths)
avg_bias = avg / true_avg
nrmse = rmse / np.mean(np.abs(ground_truths))

# График
plt.scatter(ground_truths, dot_estimates, alpha=0.5, s=10)
plt.xlabel("True ∏⟨xₖ, yₖ⟩")
plt.ylabel("CSP Estimate (complex)")
plt.title(f"CSP (K={K}) — Canonical\nRMSE={rmse:.4f}, Bias={bias:.4f}, Pearson={pearson:.4f}\nNRMSE={nrmse:.12f}, Avg. bias={avg_bias:.12f}")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"CSP (K={K}) — Canonical\nRMSE={rmse}, Bias={bias}, Pearson={pearson}\nNRMSE={nrmse}, Avg. bias={avg_bias}")
