import torch
from transformers import AutoModelForCausalLM
from inject import inject_model, snapshot_weights, restore_weights, unique_parameters
from run_experiment import load_model

m, _ = load_model("Qwen/Qwen3-0.6B", "cuda")
snap = snapshot_weights(m)

# bf16 layout: bit 15 sign, bits 14..7 exponent, bits 6..0 mantissa.
def count_flips(t, t0):
    a = t.view(torch.uint8).flatten()
    b = t0.view(torch.uint8).flatten()
    d = a ^ b
    # map byte index -> element index (2 bytes per element, little endian)
    bit_positions = torch.nonzero(d != 0).flatten()
    n = bit_positions.numel()
    # classify: for each differing byte, is it the high byte (exponent/sign) or low byte (mantissa)?
    hi = int((bit_positions % 2 == 1).sum())
    lo = n - hi
    return n, hi, lo

for rho, seed in [(1e-9, 0), (1e-7, 0)]:
    inject_model(m, rho, seed)
    total = 0
    inf_nan = 0
    exp_region = 0
    for name, p in unique_parameters(m):
        t = p.detach().cpu()
        t0 = snap[name]
        n, hi, lo = count_flips(t, t0)
        total += n
        exp_region += hi  # high byte contains sign + exponent
        inf_nan += int(torch.isinf(t).sum() + torch.isnan(t).sum())
    print(f"rho={rho:g}: flipped bits={total}, in high byte (sign|exp)={exp_region}, inf/nan weights={inf_nan}", flush=True)
    restore_weights(m, snap)
