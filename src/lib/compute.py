"""Compute accounting for method comparisons.

Methods must be compared at matched compute, so every experiment script
records exactly how much compute it spent. A forward pre-hook on the
model counts every forward call — including the calls inside generate()
and inside helper search loops — by rows (batch size: one forward of one
sequence is one row) and context tokens. Backward passes are counted
manually where the script controls them (a backward costs roughly two
forwards; row_equiv = rows + 2*backwards).

The budget unit for cross-method matching is row-equivalents. Example:
the random baseline per fault spends 100 probes x 2 free-runs of ~65
forward steps ≈ 13,000 row-equivalents; a HotFlip step costs ~2 ranking
rows + 1 backward (≈4 row-equiv) + W candidate verifications x 2 rows.
"""

import torch


class ComputeCounter:
    def __init__(self):
        self.calls = 0
        self.rows = 0
        self.tokens = 0
        self.backwards = 0
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

    def add_backward(self, n=1):
        self.backwards += n

    def row_equiv(self):
        return self.rows + 2 * self.backwards

    def as_dict(self):
        return {"forward_calls": self.calls, "rows": self.rows,
                "context_tokens": self.tokens, "backwards": self.backwards,
                "row_equiv": self.row_equiv()}
