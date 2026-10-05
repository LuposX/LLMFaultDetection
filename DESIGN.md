# Design notes

Decisions here shape the results, so they are written down. Everything else
is in code comments.

## Attention on RDNA4: eager only, per-task batch sizes

The nixpkgs torch 2.9.1/ROCm 6.4.3 build segfaults in `scaled_dot_product_attention`
on gfx1201: the FLASH and mem-efficient backends crash outright, only the
MATH backend survives. Transformers is therefore loaded with
`attn_implementation="eager"` (equivalent to MATH), which materializes the
full attention matrix — memory scales as batch × heads × seq². Consequence:
batch size is set per task in evaluate.py. Hellaswag/MMLU/GSM8K contexts
are short (~0.5–2k tokens) and run at batch 8; wikitext requests are single
whole documents (~10k tokens) and must run at batch 1. This is what keeps
the 16 GB card from OOM-ing, not an efficiency choice.

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

## rho grid

Default grid: rho in {0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5}, 3 independent draws
per rho. At 0.6B params (~9.6 Gbit of weights), rho = 1e-6 flips ~9600 bits
per draw; random single-bit-flip studies on similar models show that even
thousands of random flips can leave accuracy nearly intact, so 1e-5
(~96k flips, ~0.1% of all bits) is included as the upper end to make sure
the sweep actually shows signal. The grid is cheap to change via --rhos.

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
