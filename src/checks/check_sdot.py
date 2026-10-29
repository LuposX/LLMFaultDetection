"""Soft-vs-hard equivalence check for the SDoT path: does the differentiable
path the optimizer runs on compute the same thing as the hard token path
deployment uses?

For real prompts, the next-token logits of the same sequence are compared
under two computations:
  (a) hard path: token ids -> embedding row lookup (deployment),
  (b) soft path: near-one-hot SDoT distributions -> embedding mixtures
      (what gradient-based optimization differentiates through).

The sweep runs a range of temperatures tau. As tau shrinks the soft
distributions approach one-hot, so the two paths should converge — at the
sharpest temperatures to bitwise-identical embeddings and hence
bitwise-identical logits. At flat tau the paths SHOULD diverge: a mixture
of all tokens is a genuinely different input than any single token. The
tau where divergence starts bounds the temperature range in which
optimization stays faithful to the discrete deployment, i.e. it is direct
input for the annealing schedule. The sweep is recorded to
results/sdot_tau_sweep.json; the figure is produced separately by
src/plots/plot_sdot_sweep.py (checks produce data, plots produce figures).

Acceptance (exit 0): at the two sharpest taus, zero argmax mismatches and
max |logit difference| < 1.0 across all prompts, plus one differentiability
check (backward from soft logits to Z yields finite, nonzero gradients).
Residual logit differences at sharp tau come from double rounding
(fp32 mixture -> bf16 embedding) and are reported, not failed, as long as
the argmax tokens agree.
"""

import argparse
import json
import os
import sys

import torch

# one level below src/: add src/ to the import path for the lib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.loading import load_model  # noqa: E402
from lib.sdot import soft_forward, one_hot_logits  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"
ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

# the interesting band: the equivalence is bitwise-exact at tau <= ~1.15,
# logits start to deviate around 1.2, argmax decisions break between 1.2
# and 1.5. Above 1.5 everything is fully broken — uninteresting.
TAUS = [1.5, 1.4, 1.3, 1.25, 1.2, 1.18, 1.17, 1.16, 1.15, 1.1, 1.05, 1.0]

# logit amplitude of the one-hot encoding fed to the soft path; larger
# amplitude pushes the faithfulness boundary to larger tau
AMPLITUDE = 30.0


def run_sweep(model, tokenizer, prompts, vocab, taus, device):
    """Per prompt: token count and, per tau, the max/mean logit difference
    and the number of positions where the argmax token differs."""
    rows = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
        with torch.no_grad():
            hard = model(ids.unsqueeze(0)).logits[0].float()
            Z = one_hot_logits(ids, vocab, amplitude=AMPLITUDE)
            maxs, means, mism = [], [], []
            print(f"prompt with {ids.numel()} tokens", flush=True)
            print(f"  {'tau':>5s}  {'max|dlogit|':>12s}  {'mean|dlogit|':>13s}"
                  f"  {'argmax mism.':>13s}", flush=True)
            for tau in taus:
                soft = soft_forward(model, Z, tau).float()
                diff = (soft - hard).abs()
                mx = diff.max().item()
                mn = diff.mean().item()
                mi = int((soft.argmax(-1) != hard.argmax(-1)).sum())
                maxs.append(mx)
                means.append(mn)
                mism.append(mi)
                print(f"  {tau:5.2f}  {mx:12.3e}  {mn:13.3e}  {mi:13d}",
                      flush=True)
        rows.append({"n": ids.numel(), "max": maxs, "mean": means,
                     "mism": mism})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/sdot_tau_sweep.json"),
                    help="sweep record; the figure is made separately by "
                         "src/plots/plot_sdot_sweep.py")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(args.model, device)
    vocab = model.get_input_embeddings().num_embeddings

    prompts = [json.loads(line)["prompt"]
               for line in open(os.path.join(ROOT, "data/prompts.jsonl"))
               if line.strip()][: args.n_prompts]
    print(f"{len(prompts)} prompts, vocab {vocab}, device={device}\n", flush=True)

    rows = run_sweep(model, tokenizer, prompts, vocab, TAUS, device)

    # averaged summary (the table the per-prompt print omits)
    print("\naveraged over prompts:", flush=True)
    print(f"  {'tau':>5s}  {'max|dlogit|':>12s}  {'mean|dlogit|':>13s}"
          f"  {'mism (total)':>13s}", flush=True)
    for i, tau in enumerate(TAUS):
        mx = max(r["max"][i] for r in rows)
        mn = sum(r["mean"][i] for r in rows) / len(rows)
        mi = sum(r["mism"][i] for r in rows)
        print(f"  {tau:5.2f}  {mx:12.3e}  {mn:13.3e}  {mi:13d}", flush=True)

    record = {"model": args.model, "vocab": vocab, "amplitude": AMPLITUDE,
              "taus": TAUS, "rows": rows}
    with open(args.out, "w") as f:
        json.dump(record, f)
    print(f"wrote {args.out} (figure: python src/plots/plot_sdot_sweep.py)",
          flush=True)

    # differentiability: backward from soft logits to Z must work
    ids = tokenizer(prompts[0], return_tensors="pt").input_ids[0].to(device)
    Z = one_hot_logits(ids, vocab, amplitude=AMPLITUDE).requires_grad_(True)
    loss = soft_forward(model, Z, tau=1.0).pow(2).mean()
    (grad,) = torch.autograd.grad(loss, Z)
    ok_grad = bool(torch.isfinite(grad).all() and grad.abs().sum() > 0)
    print(f"\ndifferentiability: grad finite and nonzero: {ok_grad} "
          f"(|grad| sum {grad.abs().sum().item():.3e})", flush=True)

    # acceptance at the two sharpest taus
    sharp = [(r["max"][i], r["mism"][i])
             for i in (-2, -1) for r in rows]
    ok_sharp = all(m == 0 and mx < 1.0 for mx, m in sharp)
    print(f"acceptance at tau in {TAUS[-2:]}: "
          f"{'PASS' if ok_sharp and ok_grad else 'FAIL'}", flush=True)
    sys.exit(0 if (ok_sharp and ok_grad) else 1)


if __name__ == "__main__":
    main()
