# Design notes

Decisions here shape the results, so they are written down. Everything else
is in code comments.

## Attention on RDNA4: eager only, per-task batch sizes

The nixpkgs torch 2.9.1/ROCm 6.4.3 build segfaults in `scaled_dot_product_attention`
on gfx1201: the FLASH and mem-efficient backends crash outright, only the
MATH backend survives. Transformers is therefore loaded with
`attn_implementation="eager"` (equivalent to MATH), which materializes the
full attention matrix — memory scales as batch × heads × seq². Consequence:
batch size is set per task in benchmarks.py. Hellaswag/MMLU/GSM8K contexts
are short (~0.5–2k tokens) and run at batch 8; wikitext requests are single
whole documents (~10k tokens) and must run at batch 1. This is what keeps
the 16 GB card from OOM-ing, not an efficiency choice.

The same crash extends beyond the LLM: transformers defaults BERT-family
encoders to SDPA, so the MiniLM embedder in probes.py also segfaults
unless loaded with `attn_implementation="eager"`. Any new model in this
repo must be checked for its attention implementation before it runs.

## Everything from nixpkgs stable

The environment is fully declarative: `python3.withPackages` over
nixpkgs **nixos-25.11** with `torchWithRocm`, transformers, lm-eval,
matplotlib. The stable branch is deliberate, not conservative: on unstable,
`torchWithRocm` 2.13.0 (ROCm 7.2) is not in the binary cache and
`nix build --dry-run` shows the whole ROCm closure compiling from source
(hours). On 25.11, torch 2.9.1 with ROCm 6.4.3 is fully cached — verified
with `--dry-run` (130 paths fetched, 0 built). ROCm 6.4 is also the first
series with RDNA4 (gfx1201, the RX 9070 XT) support, so stable is exactly
the right branch, not a compromise.

## bf16 bit flips

Faults are injected into the bf16 representation of the weights, matching
what a bf16 deployment runs. Consequence: a flip in an exponent byte can
turn a weight into a huge value (or inf/NaN-like outlier), so degradation
may appear as cliffs rather than smooth curves. Casting to fp32 first would
give smoother curves but test a model representation that is not deployed.

## Uniform fault distribution over all parameters

f_bitflip(rho) is applied to every unique parameter storage (embedding,
lm_head, biases included), deduplicated because Qwen3-0.6B ties
embed_tokens and lm_head. Buffers are excluded. This matches the proposal's
fault model ("fault injection is performed exclusively on the network
weights and biases"). Per-layer or sensitivity-weighted distributions come
later, when ATPG optimization needs them.

## Poisson-position sampler for bit flips (instead of per-bit masks)

inject_model draws faults with `flip_bits_sparse`, not the mask-based
`flip_bits`. The reasoning that this substitution is sound:

The fault model is: every bit flipped independently with probability rho.
That model decomposes exactly. If n bits are each flipped iid Bernoulli(rho),
then the total flip count K follows Binomial(n, rho), and conditioned on
K = k, the set of flipped bits is uniform over all k-subsets. So
"roll one die per bit" and "draw the count, then pick that many positions
uniformly" sample the identical distribution over corrupted tensors.

Two approximations sit on top, both negligible on the experiment grid:

1. Count: K ~ Binomial(n, rho) is replaced by Poisson(rho*n). The
   total-variation error of this approximation is O(rho). The grid stops
   at rho = 1e-5, so the count distribution deviates by at most ~1e-5
   relative — far below the variance between independent draws.
2. Positions: positions are drawn with replacement (cheap randint), not
   without. This is the "flip events" model: a bit hit twice is unflipped,
   which np.bitwise_xor.at plus odd-parity folding (`torch.unique` with
   return_counts, keep odd counts) implements exactly. It differs from iid
   Bernoulli only at O(rho^2) per bit, and the expected number of duplicate
   hits is k^2/2n ~ 0.5 even at the largest grid point (rho = 1e-5, k ~
   96k flips over 9.6 Gbit).

Validity is the sparse regime. `flip_bits_sparse` falls back to the mask
path for rho > 1e-3, where the Poisson count approximation breaks down.
On the grid (rho <= 1e-5) the cost is O(flips) instead of O(network):
~10 flips at 1e-9 cost milliseconds instead of the ~70 s per draw the
mask path needs for 9.6 Gbit. Memory drops from chunked masks (hundreds
of MB) to O(k). The float64-mask note above becomes moot: there is no
per-bit `rand < rho` comparison, so the 2^-24 float32 floor cannot occur.

Reproducibility caveat: for the same (rho, seed), sparse and mask draws
are NOT bitwise identical — the RNG stream is consumed differently. Trials
in results/trials.jsonl produced by the mask injector remain valid draws
from f_bitflip(rho); they just cannot be regenerated bit-for-bit.

## The probe set

data/prompts.jsonl (built by src/make_prompts.py, deterministic, rerun-safe)
holds 100 prompts in three strata sampled from the TRAINING splits of public
datasets, independent of the scored benchmarks: 40 wikitext-2 train passage
continuations (free prose drift), 40 SciQ train questions 2-shot (short
factual answers), 20 GSM8K train word problems 2-shot with worked
exemplars (chain-of-thought amplification). Fixed exemplars inside the qa
and math prompts steer the base model to well-formed outputs — without
them the 0.6B model rambles and the strata lose their intended sensitivity.
The set is frozen once the experiment series starts: the probe set compares
every trial against a cached healthy reference for exactly these prompts,
so any change to the file invalidates all recorded curves.

## The probe metrics

Per trial, four fault-effect metrics on the same prompts (see probes.py):
token match and first-divergence position against the healthy greedy output
(when damage becomes visible in text), teacher-forced Jensen-Shannon
divergence of next-token distributions along the healthy trajectory (dense
signal, sensitive from the first flip, same quantity family as the paper's
d(x, f)), the Laya semantic judge (P(same meaning), see below), and 1 -
cosine between MiniLM embeddings of both outputs (in-field observable,
robust to surface divergence). Aggregated per stratum. JS and token match
are complementary by construction: JS moves as soon as logits shift, token
match only when the argmax flips, so the gap between the two curves
quantifies 'internal damage precedes observable damage'.

KL was tried and dropped: it is not monotone in damage. It explodes when a
partially damaged model assigns ~0 probability to tokens the healthy model
certain (measured 3.75e36 at 13 flips) and dips for fully collapsed models
whose output distribution spreads mass everywhere (measured 10.4 at 100
flips) — more damage, lower KL. The bounded Jensen-Shannon divergence
saturates at log 2 instead and rises monotonically.

## The Laya semantic judge

The `laya` package (pkgs/laya.nix, built from the PyPI sdist 0.4.7 — not in
nixpkgs; its core deps are exactly the environment's torch/transformers/
safetensors/huggingface-hub/numpy and it rides the ROCm torch override).
ModernBERT-large decision model, question type noul, state = {reference:
healthy output, observed: faulty output}, instructions: 'Do the two
responses in the state have the same meaning?'. Metric: mean P(same) over
prompts (laya_same). Runs on CPU: the GPU path segfaults on this ROCm
build, and forcing eager attention does not fix it. Sanity check on hand
pairs: identical 0.918, paraphrase 0.944, unrelated 0.027. Known risk
(their own docs): the shipped checkpoint is calibrated on support-ticket
style states, not free-form LLM output pairs, and the base checkpoint is
near chance zero-shot on out-of-domain decisions — so laya_same is a
prototype signal; validate against the JS/embedding curves before
drawing conclusions from it alone. Their documented noul label-following
bug (issue #156) did not appear in our sanity check; if values look stuck,
same escape hatch as theirs: two-option choice with neutral keys.

## rho grid and the bf16 cliff

Default grid: rho in {0, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5}, 3
independent draws per rho. At 0.6B params (~9.6 Gbit of weights), expected
flips per draw are: 1e-11 -> ~0.1, 1e-10 -> ~1, 1e-9 -> ~10, 1e-8 -> ~96.

Smoke runs showed why the grid reaches down to 1e-11: with all bf16 bits
eligible, roughly half the flips land in the exponent byte and scale a
weight by up to 2^±128. The model is statistically intact at ~1–10 flips
(1e-11 … 1e-9) and completely destroyed at ~100 flips (1e-8: hellaswag
acc_norm 49.7% -> 30.7%, wikitext ppl 27.5 -> 1.3e6) — a cliff, not a
smooth curve. In the saturated region, metrics of different draws coincide
(the corrupted model collapses to the same degenerate output regardless of
which weights were hit), so error bars there are not meaningful. The
detection-relevant regime for the ATPG work is the transition band around
1e-9 … 1e-8.

Related fix worth remembering: torch.rand is float32 (2^-24 grid); with
float32 mask draws, exact-zero draws flip bits at a ~6e-8 floor, burying
the true rate below rho ~ 1e-8. Mask draws use float64.

## Benchmarks

## Custom injector instead of LLMFI

LLMFI (github.com/hyfshishen/LLMFI, "Not All Errors Are Equal") targets
single, precisely located fault trials via config files — good for
characterizing where errors come from, but not built for sampling thousands
of independent Bernoulli draws over the whole fault space. The custom
injector is ~60 lines, exactly implements f_bitflip(rho), and is
deterministic per seed. Cross-checking individual draws against LLMFI can
be done later if needed.

## Benchmarks

lm-evaluation-harness runs hellaswag (0-shot), wikitext-2 perplexity, mmlu
(5-shot) and gsm8k (5-shot, limited to 200 samples). MMLU is evaluated
5-shot on purpose: a 0.6B model scores near random chance at 0-shot, which
compresses the measurable degradation range. GSM8K is limited for speed;
with ~16 trials in the grid it would otherwise dominate the runtime.

## Weight restoration

After every trial the weights are restored from a CPU snapshot and verified
bitwise, so a failed restore aborts the experiment instead of silently
corrupting all following trials.
