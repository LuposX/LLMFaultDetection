"""Soft Distribution over Tokens (SDoT): differentiable stand-in for the
hard token embedding.

The hard path maps a token id to one row of the embedding matrix — exact
but non-differentiable in the id. The soft path maps a per-position
probability distribution over the vocabulary to the weighted sum of all
embedding rows — differentiable in the underlying logits Z. ATPG
optimization runs on the soft path; deployment runs the hard path on the
argmax-discretized tokens. checks/check_sdot.py verifies that the two
paths agree as the distributions approach one-hot (tau -> 0), which is
what makes optimizing the soft path meaningful for the hard deployment.

Dtype discipline: the embedding mixture is accumulated in fp32 and cast
to the model's dtype before entering the transformer, mirroring how a
hard embedding row (bf16) enters. The cast is differentiable, so
gradients still reach Z. (The fp32 embedding table copy costs ~600 MB
transiently for Qwen3-0.6B; fine at this scale, worth caching if the
optimization loop ever becomes memory-bound.)
"""

import math

import torch

LOG2 = math.log(2.0)


def softmax_tau(Z: torch.Tensor, tau: float) -> torch.Tensor:
    """Tempered softmax over the last (vocabulary) dimension.

    tau < 1 sharpens toward one-hot (refinement toward real tokens),
    tau > 1 flattens (exploration across tokens).
    """
    return torch.softmax(Z / tau, dim=-1)


def soft_embeddings(dists: torch.Tensor, model) -> torch.Tensor:
    """Map [N, V] token distributions to [N, d] input embeddings.

    Uses the model's own embedding weight (via get_input_embeddings()), so
    the soft path consumes the same table the hard path looks up.
    """
    W = model.get_input_embeddings().weight
    return (dists @ W.to(dists.dtype)).to(model.dtype)


def soft_forward(model, Z: torch.Tensor, tau: float) -> torch.Tensor:
    """Forward pass on soft inputs. Z: [N, V] logits for one sequence.

    Returns logits [N, V]: the next-token distribution inputs at every
    position, exactly what the hard path would produce for the same
    positions. inputs_embeds bypasses the hard lookup; the transformer
    body, rotary positions and output head run as usual.

    Differentiable in Z (everything along the way is), so autograd.grad
    w.r.t. Z works; wrap in torch.no_grad() when gradients are unwanted.
    """
    dists = softmax_tau(Z, tau)
    emb = soft_embeddings(dists, model)
    out = model(inputs_embeds=emb.unsqueeze(0))
    return out.logits[0]


def soft_forward_prefix(model, Z: torch.Tensor, tau: float,
                        prefix_ids: torch.Tensor) -> torch.Tensor:
    """Teacher-forced forward: soft prompt (N positions) followed by a
    hard, fixed continuation of M prefix tokens.

    Returns logits [N+M, V]. Positions N-1 .. N+M-2 hold the next-token
    predictions FOR the prefix tokens y1 .. yM — the positions at which
    healthy and faulty model are compared. Because the prefix is fixed
    and identical for both passes, any difference there is attributable
    to the fault, not to diverging self-generated histories (and no
    sampling/argmax occurs on the output side, so gradients stay intact).
    """
    dists = softmax_tau(Z, tau)
    emb_prompt = soft_embeddings(dists, model)
    emb_prefix = model.get_input_embeddings()(prefix_ids)
    emb = torch.cat([emb_prompt, emb_prefix], dim=0)
    out = model(inputs_embeds=emb.unsqueeze(0))
    return out.logits[0]


def js_positions(logits_a: torch.Tensor,
                 logits_b: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence per position between the softmax
    distributions of two logit tensors [T, V]. Returns [T], each in
    [0, log 2]. Differentiable in both inputs; this is the ATPG objective
    (maximize to make a fault visible) and the same quantity family the
    probe metrics use passively.

    torch.where guards the p == 0 terms: log p = -inf there and p * -inf
    would be NaN; the contribution to JS is exactly 0.
    """
    la = torch.log_softmax(logits_a.float(), dim=-1)
    lb = torch.log_softmax(logits_b.float(), dim=-1)
    log_m = torch.logaddexp(la, lb) - LOG2      # log((a + b) / 2)
    p, q = la.exp(), lb.exp()
    term_a = torch.where(p > 0, p * (la - log_m), torch.zeros_like(la))
    term_b = torch.where(q > 0, q * (lb - log_m), torch.zeros_like(lb))
    return 0.5 * term_a.sum(-1) + 0.5 * term_b.sum(-1)


def one_hot_logits(ids: torch.Tensor, vocab_size: int,
                   amplitude: float = 30.0) -> torch.Tensor:
    """Z that puts `amplitude` of logit mass on the given token ids.

    This is the bridge between the two paths: at tau = 1 the softmax of
    such a Z is nearly (fp32: within ~1e-9 mass) one-hot on `ids`; smaller
    tau drives it further toward exact one-hot. Used by the equivalence
    check to feed the soft path the same 'tokens' as the hard path.
    """
    Z = torch.zeros(ids.numel(), vocab_size, device=ids.device)
    Z[torch.arange(ids.numel(), device=ids.device), ids] = amplitude
    return Z
