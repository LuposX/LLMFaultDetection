import torch

g = torch.Generator().manual_seed(0)
n = 1_000_000_000
u = torch.rand(n, generator=g)
for t in [1e-9, 1e-8, 1e-7, 1e-6]:
    hits = int((u < t).sum())
    print(f"P(u<{t:g}): {hits} hits, expected {n*t:.1f}")
print("min u:", u.min().item())
