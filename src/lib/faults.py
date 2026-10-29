"""Explicit, named fault configurations for ATPG-style detection.

inject.py implements the *distribution* f_bitflip(rho): every trial is one
random draw and only the flip count is recorded. The detection pipeline
needs faults as *entities*: exactly which bits flipped, so faults can be
split into train/validation/held-out sets and coverage can be reported per
fault ("fault X detected by patterns {p3, p7}").

A Fault is a set of (parameter name, element index, bit index) triples plus
metadata. XOR is involutive, so apply() and undo() share one code path;
undo() is only valid while the weights are otherwise unchanged since the
apply (use inject.snapshot/restore_weights for the general case).

Bit numbering: bf16, little-endian. Within one 16-bit element, bits 0-6
are the mantissa, 7-14 the exponent, 15 the sign. The scope strata are
defined per element bit index, not per byte (byte 0 mixes mantissa-low and
exponent bit 0, so byte-level strata would be meaningless).

Scopes:
- "all":      uniform over all bit positions (same position distribution
              as f_bitflip, but as an exact-count draw instead of Poisson)
- "exponent": only bits 7-14. A hit here rescales a weight by up to 2^+-128;
              the degradation sweep showed this stratum causes the
              catastrophic, trivially-detected faults.
- "mantissa": only bits 0-6. Perturbs a weight within its normal range;
              the hard-to-detect stratum that ATPG must prove itself on.
- "sign":     bit 15 only (weight sign flips).
"""

import hashlib
import json

import torch

from .inject import unique_parameters

_SCOPES = {
    "all": list(range(16)),
    "mantissa": list(range(0, 7)),
    "exponent": list(range(7, 15)),
    "sign": [15],
}


def _fault_id(flips):
    payload = json.dumps(sorted(flips)).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


class Fault:
    """A recorded set of bit flips, replayable on any model with the same
    parameter names and shapes."""

    def __init__(self, k, scope, seed, flips):
        self.k = k            # requested flip count
        self.scope = scope
        self.seed = seed
        self.flips = sorted(flips)   # [[name, elem_idx, bit], ...]
        self.id = _fault_id(self.flips)

    @property
    def n_flips(self):
        """Actual flips after duplicate-position removal (== k in practice)."""
        return len(self.flips)

    def to_dict(self):
        return {"id": self.id, "k": self.k, "scope": self.scope,
                "seed": self.seed, "flips": self.flips}

    @classmethod
    def from_dict(cls, d):
        f = cls(d["k"], d["scope"], d["seed"], [tuple(x) for x in d["flips"]])
        assert f.id == d["id"], "fault id mismatch — corrupted record?"
        return f

    def _xor(self, model):
        """XOR every recorded bit in place. apply() and undo() both call
        this: flipping the same bits twice returns the original weights."""
        by_name = {name: p for name, p in unique_parameters(model)}
        for name, elem, bit in self.flips:
            param = by_name[name]
            assert param.data.is_contiguous(), f"{name} not contiguous"
            flat_u8 = param.data.view(-1).view(torch.uint8)
            byte_idx = elem * param.data.element_size() + bit // 8
            flat_u8[byte_idx] ^= 1 << (bit % 8)

    def apply(self, model):
        """Corrupt the model's weights at the recorded positions."""
        self._xor(model)

    def undo(self, model):
        """Restore the weights. Only valid if nothing but apply() touched
        the parameters since."""
        self._xor(model)


def sample_fault(model, k, seed, scope="all"):
    """Draw exactly k bit positions uniformly over the eligible positions.

    Eligible = every (element, bit) with bit in the scope's bit set. The
    draw is over elements uniformly and bits uniformly within the scope,
    which is uniform over eligible positions because every element has the
    same 16-bit layout. Duplicate positions are removed (a bit hit twice
    is unflipped); with k <= 8 over ~4.8G elements this is a probability-
    zero event, handled for correctness only.
    """
    assert scope in _SCOPES, f"unknown scope {scope}"
    params = [(name, p) for name, p in unique_parameters(model)]
    assert all(p.element_size() == 2 for _, p in params), \
        "bit layout assumes 16-bit elements (bf16)"
    sizes = torch.tensor([p.numel() for _, p in params])
    starts = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])
    total_elems = int(sizes.sum())

    gen = torch.Generator().manual_seed(seed)
    elems = torch.randint(0, total_elems, (k,), generator=gen)
    allowed = torch.tensor(_SCOPES[scope], dtype=torch.long)
    bits = allowed[torch.randint(0, len(allowed), (k,), generator=gen)]

    param_idx = torch.searchsorted(starts[1:], elems, right=True)
    flips = set()
    for e, b, pi in zip(elems.tolist(), bits.tolist(), param_idx.tolist()):
        name, _ = params[pi]
        flips.add((name, e - int(starts[pi]), b))
    return Fault(k, scope, seed, flips)


def sample_faults(model, n, k, scope, seed0):
    """n independent faults with consecutive seeds (seed0 .. seed0+n-1)."""
    return [sample_fault(model, k, seed0 + i, scope) for i in range(n)]


if __name__ == "__main__":
    import torch.nn as nn

    class _Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(1000, dtype=torch.bfloat16))
            self.b = nn.Parameter(torch.randn(100, dtype=torch.bfloat16))
            self.dup = self.a  # tied storage: must never receive its own draws

    m = _Dummy()

    # 1. reproducibility: same seed -> identical fault (same id)
    f1 = sample_fault(m, 4, seed=7)
    f2 = sample_fault(m, 4, seed=7)
    assert f1.flips == f2.flips and f1.id == f2.id, "not reproducible"
    print(f"reproducible: id={f1.id}, flips={f1.flips}")

    # 2. scopes respected: k=1 draws land only in the scope's bits
    for scope, expected in [("mantissa", range(0, 7)),
                            ("exponent", range(7, 15)),
                            ("sign", [15])]:
        bits = [sample_fault(m, 1, seed=i, scope=scope).flips[0][2]
                for i in range(200)]
        assert all(b in expected for b in bits), f"scope {scope} violated"
        print(f"scope {scope}: 200 draws, bits within {list(expected)}")

    # 3. scope 'all' is uniform over bits: ~8/16 of single flips are
    #    exponent hits (loose 0.4..0.6 band)
    bits = [sample_fault(m, 1, seed=1000 + i).flips[0][2] for i in range(2000)]
    frac_exp = sum(7 <= b <= 14 for b in bits) / len(bits)
    assert 0.4 < frac_exp < 0.6, f"exponent fraction {frac_exp} off"
    print(f"scope all: exponent-bit fraction {frac_exp:.3f} (expect ~0.5)")

    # 4. apply/undo round trip, bitwise, through the uint8 view (NaN-safe)
    before = {n: p.detach().view(torch.uint8).clone()
              for n, p in m.named_parameters()}
    f = sample_fault(m, 6, seed=42)
    f.apply(m)
    changed = sum(int((p.detach().view(torch.uint8) != before[n]).any())
                  for n, p in m.named_parameters())
    assert changed == len({name for name, _, _ in f.flips}), \
        "apply did not hit exactly the recorded parameters"
    f.undo(m)
    assert all(torch.equal(p.detach().view(torch.uint8), before[n])
               for n, p in m.named_parameters()), "undo not bitwise"
    print(f"apply/undo round trip bitwise-identical ({f.n_flips} flips)")

    # 5. tied weights: a fault on 'a' is visible through 'dup', and undo
    #    still restores both (they share one storage). named_parameters()
    #    drops duplicate names, so capture the tensors directly.
    f = sample_fault(m, 3, seed=99)
    a_before = m.a.detach().view(torch.uint8).clone()
    f.apply(m)
    a_changed = not torch.equal(m.a.detach().view(torch.uint8), a_before)
    dup_changed = not torch.equal(m.dup.detach().view(torch.uint8), a_before)
    assert a_changed and dup_changed and m.a.data_ptr() == m.dup.data_ptr()
    f.undo(m)
    assert torch.equal(m.a.detach().view(torch.uint8), a_before)
    print("tied storage: fault on 'a' propagates to 'dup', undo restores both")

    # 6. JSON round trip preserves the fault exactly
    f3 = Fault.from_dict(json.loads(json.dumps(f1.to_dict())))
    assert f3.flips == f1.flips and f3.id == f1.id, "JSON round trip broken"
    print("JSON round trip ok")

    print("faults self-test passed")
