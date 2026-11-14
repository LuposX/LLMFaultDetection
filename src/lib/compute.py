"""Compute accounting for method comparisons.

Methods must be compared at matched compute, so every experiment script
records exactly how much compute it spent. A forward pre-hook on the
model counts every forward call — including the calls inside generate()
and inside helper search loops — by rows (batch size: one forward of one
sequence is one row) and context tokens. Backward passes are counted
manually where the script controls them (a backward costs roughly two
forwards; row_equiv = rows + 2*backwards).

The budget unit for cross-method matching is row-equivalents. Example
(measured, Qwen3-0.6B bf16, 100 probes): the random baseline spends
~6,400 row-equivalents per fault — one faulty free-run per probe, 100
probes decoded in batches of 8 (the healthy reference is decoded once
and shared across faults, so it is not part of the per-fault cost). A
HotFlip step costs ~2 ranking rows + 1 backward (≈4 row-equiv) + W
candidate verifications x 2 rows.

FLOPs accounting: alongside row-equivalents, the counter estimates FLOPs
analytically. Forward passes cost 2*P FLOPs per token, where P is the
number of non-embedding weight elements (Qwen3-0.6B: P = 596,049,920); a
backward pass costs twice the forward cost of the computation graph it
traverses, so callers pass the graph's forward token count to
add_backward(). Attention and embedding lookups are neglected (at the
64-token contexts used here they are below 1 percent of the total).
"""

import torch

# non-embedding weight elements of Qwen3-0.6B (embedding lookup performs no
# matmul, so embedding weights are excluded from the FLOPs estimate)
QWEN3_0P6B_PARAMS = 596_049_920


class ComputeCounter:
    def __init__(self, n_params: int = QWEN3_0P6B_PARAMS):
        self.calls = 0
        self.rows = 0
        self.tokens = 0
        self.backwards = 0
        self.n_params = n_params
        self.bw_tokens = 0          # forward tokens in traversed graphs
        self._bw_tokens_unknown = 0  # backwards without a token count
        self._handles = []

    def attach(self, model):
        """Count all forward passes through model (root module)."""
        def pre(module, f_args, kwargs):
            self.calls += 1
            t = kwargs.get("input_ids")
            if t is None:
                t = kwargs.get("inputs_embeds")
            if t is None and f_args:
                t = f_args[0]
            if t is not None and torch.is_tensor(t) and t.dim() == 2:
                self.rows += t.shape[0]
                self.tokens += t.shape[0] * t.shape[1]
        self._handles.append(
            model.register_forward_pre_hook(pre, with_kwargs=True))

    def add_backward(self, n=1, tokens=None):
        """Count n backward passes. tokens: forward token count of the
        computation graph the backward traverses (2 forwards in the
        HotFlip ranking step -> pass 2*L). Without a token count, the
        average forward-pass token count is used as an approximation."""
        self.backwards += n
        if tokens is None:
            self._bw_tokens_unknown += n
        else:
            self.bw_tokens += n * tokens

    def row_equiv(self):
        return self.rows + 2 * self.backwards

    def flops(self):
        p = self.n_params
        fwd = 2 * p * self.tokens
        if self._bw_tokens_unknown:
            avg = self.tokens / max(self.calls, 1)
            bw = 4 * p * (self.bw_tokens + self._bw_tokens_unknown * avg)
        else:
            bw = 4 * p * self.bw_tokens
        return fwd + bw

    def as_dict(self):
        return {"forward_calls": self.calls, "rows": self.rows,
                "context_tokens": self.tokens, "backwards": self.backwards,
                "row_equiv": self.row_equiv(), "flops": self.flops()}
