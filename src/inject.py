"""Bit-flip fault injection for HuggingFace causal language models.

Implements the f_bitflip fault model from "Functional Test Generation for
In-Field Testing of Deep Learning Models with Test Storage Constraints"
(Li et al.): every bit of every parameter is flipped independently with
probability rho. rho parameterizes a distribution over the fault space,
which is tractable for LLM-scale networks where enumerating individual
faults is not.

Injection happens on the bf16 byte representation, so a faulty model here
is exactly what a bf16 deployment would run.
"""

import torch


def _popcount_u8(x: torch.Tensor) -> torch.Tensor:
    """Set bits per element of a uint8 tensor (SWAR population count)."""
    x = x - ((x >> 1) & 0x55)          # pairs
    x = (x & 0x33) + ((x >> 2) & 0x33) # nibbles
    return (x + (x >> 4)) & 0x0F       # bytes


def flip_bits(tensor: torch.Tensor, rho: float, generator: torch.Generator,
              chunk_bits: int = 1 << 24) -> torch.Tensor:
    """Return a copy of `tensor` with each underlying bit flipped with prob rho.

    The tensor is reinterpreted as raw bytes and processed in chunks so the
    Bernoulli masks never exceed a few hundred MB even for the embedding
    matrix (~300 MB as bf16). Per-bit independence is preserved across
    chunks because each mask draw is independent.

    Mask draws are float64: torch.rand's float32 grid is 2^-24 (~6e-8), so
    at rho below that, exact-zero draws would flip bits at a ~6e-8 floor
    regardless of rho.
    """
    if rho <= 0:
        return tensor.clone()

    orig_dtype = tensor.dtype
    orig_shape = tensor.shape
    # view(-1) flattens; view(torch.uint8) reinterprets bf16 (2 B/elem) as bytes.
    flat_u8 = tensor.detach().cpu().contiguous().view(-1).view(torch.uint8)
    out_u8 = flat_u8.clone()
    n = out_u8.numel()

    for start in range(0, n, chunk_bits):
        end = min(start + chunk_bits, n)
        chunk = out_u8[start:end]
        flip = torch.zeros(end - start, dtype=torch.uint8)
        for bit in range(8):
            mask = torch.rand(end - start, dtype=torch.float64,
                              generator=generator) < rho
            flip |= mask.to(torch.uint8) << bit
        chunk ^= flip  # in-place on the slice view

    return out_u8.view(orig_dtype).reshape(orig_shape)


def unique_parameters(model: torch.nn.Module):
    """Yield (name, param) with tied weights deduplicated by storage.

    Qwen3-0.6B ties embed_tokens and lm_head; they share one storage and must
    receive one fault draw, not two. Buffers (e.g. rotary inv_freq) are
    excluded: named_parameters() covers only trainable parameters.
    """
    seen = set()
    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        yield name, param


def inject_model(model: torch.nn.Module, rho: float, seed: int) -> None:
    """Apply f_bitflip(rho) in place to all unique parameter storages of model.

    Each (rho, seed) pair is a deterministic, reproducible fault draw from
    the distribution over the fault space.
    """
    generator = torch.Generator().manual_seed(seed)
    for name, param in unique_parameters(model):
        injected = flip_bits(param, rho, generator)
        param.data.copy_(injected.to(device=param.device, dtype=param.dtype))


def snapshot_weights(model: torch.nn.Module) -> dict:
    """Copy all unique parameter storages to CPU (bf16, ~1.2 GB for 0.6B)."""
    return {name: param.detach().cpu().clone()
            for name, param in unique_parameters(model)}


def restore_weights(model: torch.nn.Module, snapshot: dict) -> None:
    for name, param in unique_parameters(model):
        param.data.copy_(snapshot[name].to(device=param.device, dtype=param.dtype))


def verify_restore(model: torch.nn.Module, snapshot: dict) -> bool:
    """True iff every parameter is bitwise identical to the snapshot."""
    for name, param in unique_parameters(model):
        if not torch.equal(param.detach().cpu(), snapshot[name]):
            return False
    return True


if __name__ == "__main__":
    # Self-test of the injection statistics: flipped bit counts must match
    # rho, and repeated draws with the same seed must be BITWISE identical.
    # Comparisons go through the uint8 view — flipping exponent bits
    # produces NaN/inf values in bf16, and torch.equal on NaN always fails.
    for rho in [0.5, 1e-6, 1e-9]:
        gen = torch.Generator().manual_seed(42)
        t = torch.randn(1000, 1000, dtype=torch.bfloat16)
        out = flip_bits(t, rho, gen)
        a = t.view(torch.uint8)
        b = out.view(torch.uint8)
        diff = _popcount_u8(a ^ b).sum().item()
        expected = rho * t.numel() * 16  # 16 bits per bf16 element
        print(f"rho={rho:g}: flipped {diff} bits, expected ~{expected:.1f}")
        assert torch.equal(a, b.view(a.shape) ^ (a ^ b)), "xor not idempotent"
        gen2 = torch.Generator().manual_seed(42)
        out2 = flip_bits(t, rho, gen2)
        assert torch.equal(b, out2.view(torch.uint8)), "not reproducible"
    print("inject self-test passed")
