"""Benchmark evaluation wrapper around lm-evaluation-harness.

Tasks follow the research proposal:
  - hellaswag        (0-shot, loglikelihood, acc_norm)
  - wikitext         (WikiText-2 perplexity)
  - mmlu             (5-shot, loglikelihood, acc)
  - gsm8k            (5-shot, generative, exact_match; limited for speed)

Evaluation runs on the model object as it currently sits in memory, so
injected weights are benchmarked without reloading anything from disk.
"""

import lm_eval
from lm_eval.models.huggingface import HFLM

# Preferred metric key per task; the caller falls back to whatever keys the
# harness reports if a name changes between lm-eval versions.
PREFERRED_METRIC = {
    "hellaswag": "acc_norm",
    "mmlu": "acc",
    "gsm8k": "exact_match",
}


def evaluate(model, tasks, gsm8k_limit=200, batch_size=8, limit=None):
    """Run the requested tasks and return {task: {metric: value}}.

    `tasks` is a subset of {"hellaswag", "wikitext", "mmlu", "gsm8k"}.
    Loglikelihood tasks and generative tasks need different few-shot
    settings, so the harness is invoked in up to four batches over the same
    HFLM instance (no model reload between them).

    With eager attention, memory scales with batch * heads * seq^2, so the
    batch is set per task: wikitext requests are single whole documents
    (~10k tokens) and must run at batch 1; the other tasks are
    short-context and tolerate larger batches.
    """
    lm = HFLM(pretrained=model, batch_size=batch_size)
    merged = {}

    runs = []
    if "hellaswag" in tasks:
        runs.append(("hellaswag", None, limit, batch_size))
    if "wikitext" in tasks:
        runs.append(("wikitext", None, limit, 1))
    if "mmlu" in tasks:
        runs.append(("mmlu", 5, limit, batch_size))
    if "gsm8k" in tasks:
        runs.append(("gsm8k", 5, gsm8k_limit, batch_size))

    for task, fewshot, task_limit, bs in runs:
        # simple_evaluate ignores batch_size for a prebuilt LM, so the
        # per-task batch is set on the HFLM instance directly.
        lm.batch_size_per_gpu = bs
        result = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=fewshot,   # None -> task defaults (hellaswag: 0)
            batch_size=bs,
            limit=task_limit,
        )
        for t, metrics in result["results"].items():
            merged[t] = {k: v for k, v in metrics.items()
                         if isinstance(v, (int, float))}

    return merged
