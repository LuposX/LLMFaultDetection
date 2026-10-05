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


def evaluate(model, tasks, gsm8k_limit=200, batch_size="auto"):
    """Run the requested tasks and return {task: {metric: value}}.

    `tasks` is a subset of {"hellaswag", "wikitext", "mmlu", "gsm8k"}.
    Loglikelihood tasks and generative tasks need different few-shot
    settings, so the harness is invoked in up to three batches over the
    same HFLM instance (no model reload between them).
    """
    lm = HFLM(pretrained=model, batch_size=batch_size)
    merged = {}

    runs = []
    if "hellaswag" in tasks or "wikitext" in tasks:
        runs.append(([t for t in ["hellaswag", "wikitext"] if t in tasks], None, None))
    if "mmlu" in tasks:
        runs.append((["mmlu"], 5, None))
    if "gsm8k" in tasks:
        runs.append((["gsm8k"], 5, gsm8k_limit))

    for task_list, fewshot, limit in runs:
        result = lm_eval.simple_evaluate(
            model=lm,
            tasks=task_list,
            num_fewshot=fewshot,   # None -> task defaults (hellaswag/wikitext: 0)
            batch_size=batch_size,
            limit=limit,
        )
        for task, metrics in result["results"].items():
            merged[task] = {k: v for k, v in metrics.items()
                            if isinstance(v, (int, float))}

    return merged
