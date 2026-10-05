import torch
from inject import flip_bits, unique_parameters
from run_experiment import load_model

m, _ = load_model("Qwen/Qwen3-0.6B", "cpu")  # CPU to rule out GPU copy effects
total_bytes = 0
for rho in [1e-9, 1e-7]:
    total_mask = 0
    total_diff = 0
    gen = torch.Generator().manual_seed(0)
    for name, p in unique_parameters(m):
        flat = p.detach().view(-1)
        n_bytes = flat.numel() * 2
        total_bytes = flat.numel()
        # count mask directly, byte-level
        out = flip_bits(p, rho, gen)
        a = p.detach().view(torch.uint8).flatten()
        b = out.view(torch.uint8).flatten()
        total_diff += int((a != b).sum())
        # expected from bernoulli over bits
    print(f"rho={rho:g}: differing bytes={total_diff}, "
          f"model bytes={sum(p.numel()*2 for _, p in unique_parameters(m))}, "
          f"expected diffs ~{rho*8*sum(p.numel()*2 for _, p in unique_parameters(m)):.1f}")
