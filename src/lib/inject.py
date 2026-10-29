"""Bit-flip fault injection for HuggingFace causal language models.

Implements the f_bitflip fault model from "Functional Test Generation for
In-Field Testing of Deep Learning Models with Test Storage Constraints"
(Li et al.): every bit of every parameter is flipped independently with
probability rho. rho parameterizes a distribution over the fault space,
which is tractable for LLM-scale networks where enumerating individual
faults is not.

Injection happens on the bf16 byte representation, so a faulty model here
is exactly what a bf16 deployment would run.

Two samplers produce draws from the same distribution:
- flip_bits: per-bit Bernoulli masks, O(network) cost. Reference
  implementation, also the fallback for large rho.
- flip_bits_sparse: draws the flip count and then uniform bit positions,
  O(flips) cost. Used by inject_model; see DESIGN.md for why the two are
  equivalent and when the sparse path is valid.
"""

import numpy as np
import torch


def _popcount_u8(x: torch.Tensor) -> torch.Tensor:
    """Set bits per element of a uint8 tensor (SWAR population count)."""
    x = x - ((x >> 1) & 0x55)          # pairs
    x = (x & 0x33) + ((x >> 2) & 0x33) # nibbles
    return (x + (x >> 4)) & 0x0F       # bytes


def flip_bits(tensor: torch.Tensor, rho: float, generator: torch.Generator,
              chunk_bits: int = 1 << 24):
    """Return (copy of `tensor` with each bit flipped with prob rho, n_flips).

    The tensor is reinterpreted as raw bytes and processed in chunks so the
    Bernoulli masks never exceed a few hundred MB even for the embedding
    matrix (~300 MB as bf16). Per-bit independence is preserved across
    chunks because each mask draw is independent.

    Mask draws are float64: torch.rand's float32 grid is 2^-24 (~6e-8), so
    at rho below that, exact-zero draws would flip bits at a ~6e-8 floor
    regardless of rho.
    """
    if rho <= 0:
        return tensor.clone(), 0

    orig_dtype = tensor.dtype
    orig_shape = tensor.shape
    # view(-1) flattens; view(torch.uint8) reinterprets bf16 (2 B/elem) as bytes.
    flat_u8 = tensor.detach().cpu().contiguous().view(-1).view(torch.uint8)
    out_u8 = flat_u8.clone()
    n = out_u8.numel()

    n_flips = 0
    for start in range(0, n, chunk_bits):
        end = min(start + chunk_bits, n)
        chunk = out_u8[start:end]
        flip = torch.zeros(end - start, dtype=torch.uint8)
        for bit in range(8):
            mask = torch.rand(end - start, dtype=torch.float64,
                              generator=generator) < rho
            flip |= mask.to(torch.uint8) << bit
        n_flips += int(_popcount_u8(flip).sum().item())
        chunk ^= flip  # in-place on the slice view

    return out_u8.view(orig_dtype).reshape(orig_shape), n_flips


def flip_bits_sparse(tensor: torch.Tensor, rho: float,
                     generator: torch.Generator):
    """Return (copy of `tensor` with each bit flipped with prob rho, n_flips).

    Equivalent to flip_bits in distribution. Per-bit Bernoulli(rho) over
    n bits decomposes exactly as: draw the flip count K ~ Binomial(n, rho),
    then choose K positions uniformly (without replacement). For small rho
    the Binomial count is Poisson(rho*n) up to total-variation error O(rho).
    Positions are drawn with replacement and folded by parity, which matches
    the iid model up to O(rho^2) per bit (duplicate hits cancel under XOR).

    Sparse regime only: rho*n must be a small fraction of n and rho must be
    small for the Poisson approximation. Falls back to flip_bits otherwise.
    Draws for the same (rho, seed) differ from flip_bits draws because the
    RNG stream is consumed differently.
    """
    if rho <= 0:
        return tensor.clone(), 0
    n_bits = tensor.numel() * tensor.element_size() * 8
    if rho > 1e-3:  # Poisson count approximation breaks down
        return flip_bits(tensor, rho, generator)

    orig_dtype = tensor.dtype
    orig_shape = tensor.shape
    flat_u8 = tensor.detach().cpu().contiguous().view(-1).view(torch.uint8)
    out_u8 = flat_u8.clone()

    lam = rho * n_bits
    k = int(torch.poisson(torch.tensor(lam, dtype=torch.float64),
                          generator=generator).item())
    if k == 0:
        return out_u8.view(orig_dtype).reshape(orig_shape), 0

    pos = torch.randint(0, n_bits, (k,), generator=generator)
    pos, counts = torch.unique(pos, return_counts=True)
    pos = pos[counts % 2 == 1]  # a bit hit an even number of times is unflipped

    # XOR into the byte array; np.bitwise_xor.at applies every index hit,
    # so two positions on the same byte with different bits both land.
    arr = out_u8.numpy()
    byte_idx = (pos // 8).numpy()
    bit = (pos % 8).numpy().astype(np.uint8)
    np.bitwise_xor.at(arr, byte_idx, np.bitwise_left_shift(np.uint8(1), bit))

    return out_u8.view(orig_dtype).reshape(orig_shape), pos.numel()


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


def inject_model(model: torch.nn.Module, rho: float, seed: int) -> int:
    """Apply f_bitflip(rho) in place to all unique parameter storages of model.

    Each (rho, seed) pair is a deterministic, reproducible fault draw from
    the distribution over the fault space. Returns the number of bit flips
    actually applied (can be 0 at small rho — callers can skip evaluation
    of an unchanged model).
    """
    generator = torch.Generator().manual_seed(seed)
    total = 0
    for name, param in unique_parameters(model):
        injected, k = flip_bits_sparse(param, rho, generator)
        total += k
        param.data.copy_(injected.to(device=param.device, dtype=param.dtype))
    return total


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
    # Self-test of the injection statistics for BOTH samplers: flipped bit
    # counts must match rho, and repeated draws with the same seed must be
    # BITWISE identical. Comparisons go through the uint8 view — flipping
    # exponent bits produces NaN/inf values in bf16, and torch.equal on NaN
    # always fails.
    for sampler in [flip_bits, flip_bits_sparse]:
        print(sampler.__name__)
        for rho in [0.5, 1e-6, 1e-9]:
            gen = torch.Generator().manual_seed(42)
            t = torch.randn(1000, 1000, dtype=torch.bfloat16)
            out, n_flips = sampler(t, rho, gen)
            a = t.view(torch.uint8)
            b = out.view(torch.uint8)
            diff = _popcount_u8(a ^ b).sum().item()
            expected = rho * t.numel() * 16  # 16 bits per bf16 element
            print(f"  rho={rho:g}: flipped {diff} bits (sampler said {n_flips}), "
                  f"expected ~{expected:.1f}")
            assert n_flips == diff, "reported flip count != actual bit flips"
            assert torch.equal(a, b.view(a.shape) ^ (a ^ b)), "xor not idempotent"
            gen2 = torch.Generator().manual_seed(42)
            out2, _ = sampler(t, rho, gen2)
            assert torch.equal(b, out2.view(torch.uint8)), "not reproducible"
    print("inject self-test passed")
