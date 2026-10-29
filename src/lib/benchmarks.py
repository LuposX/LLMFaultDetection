"""Benchmark evaluation wrapper around lm-evaluation-harness.

Tasks follow the research proposal:
  - hellaswag        (0-shot, loglikelihood, acc_norm)
  - wikitext         (WikiText-2 perplexity)
  - mmlu             (5-shot, loglikelihood, acc)
  - gsm8k            (5-shot, generative, exact_match; limited for speed)

Evaluation runs on the model object as it currently sits in memory, so
injected weights are benchmarked without reloading anything from disk.
"""

import logging
import os
import time

# Kill the HuggingFace datasets/transformers tqdm bars (dataset downloads and
# split generation spam one finished bar per mmlu subtask). The harness's
# own per-request bars stay visible.
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

import torch
import datasets
import lm_eval
import tqdm as tqdm_mod
from lm_eval.models.huggingface import HFLM
from transformers.utils import logging as hf_logging

datasets.disable_progress_bars()
hf_logging.disable_progress_bar()
# lm-eval logs benign per-task metric warnings ([Task: wikitext] metric ...
# using default aggregation) on every run; they carry no information.
# Child loggers (lm_eval.api.task etc.) inherit this level.
logging.getLogger("lm_eval").setLevel(logging.ERROR)
# lm_eval.api.task.build_all_requests prints one instant, finished bar per
# task (it just tokenizes the docs). Point task.py's tqdm import at a
# permanently disabled wrapper.
_orig_tqdm = tqdm_mod.tqdm


def _quiet_tqdm(*a, **kw):
    kw.setdefault("disable", True)
    return _orig_tqdm(*a, **kw)


import lm_eval.api.task as _lm_task
_lm_task.tqdm = _quiet_tqdm

# Preferred metric key per task; the caller falls back to whatever keys the
# harness reports if a name changes between lm-eval versions.
PREFERRED_METRIC = {
    "hellaswag": "acc_norm",
    "mmlu": "acc",
    "gsm8k": "exact_match",
    "winogrande": "acc",
    "arc_easy": "acc",
    "boolq": "acc",
}


def _quiet_micro_bars(lm, threshold=32):
    """Silence lm-eval progress bars that only ever show a handful of items.

    loglikelihood_rolling (wikitext) feeds _loglikelihood_tokens one window
    at a time with disable_tqdm hardcoded to False, so a finished "1/1" bar
    line is printed per document. Bar instances with fewer than `threshold`
    requests are instant or near-instant and carry no information, so they
    are disabled; long bars (hellaswag, mmlu, gsm8k request batches) stay.
    """
    orig_tokens = lm._loglikelihood_tokens

    def tokens(requests, disable_tqdm=False, override_bs=None, **kw):
        if len(requests) < threshold:
            disable_tqdm = True
        return orig_tokens(requests, disable_tqdm=disable_tqdm,
                           override_bs=override_bs, **kw)

    lm._loglikelihood_tokens = tokens

    orig_rolling = lm.loglikelihood_rolling

    def rolling(requests, disable_tqdm=False):
        # outer per-document tokenization bar; instant, so always quiet
        return orig_rolling(requests, disable_tqdm=True)

    lm.loglikelihood_rolling = rolling


def evaluate(model, tasks, gsm8k_limit=200, batch_size=8, limit=None,
             on_task=None, mmlu_limit=None):
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
    # Cap the evaluation context. Eager attention needs batch*heads*seq^2
    # memory (fp32 softmax peak); uncapped wikitext docs (~10k tokens) OOM
    # a 16 GB card at batch 1. 4096 tokens truncates the tail of long docs
    # but applies identically to every trial, so the comparison stays fair.
    lm = HFLM(pretrained=model, batch_size=batch_size, max_length=4096)
    _quiet_micro_bars(lm)
    merged = {}

    # Per-task batch caps. _model_call materializes full logits
    # (batch * seq * 152k vocab); mmlu's 5-shot contexts reach ~2k tokens,
    # so batch 8 peaks at several GB per forward and OOMs a 16 GB card.
    # batch 2 fits with headroom; hellaswag/gsm8k contexts are short enough
    # to keep the caller's batch size.
    task_batch_cap = {"mmlu": 2}

    runs = []
    # (task, fewshot (None -> task default), limit, batch size). All
    # loglikelihood tasks; short contexts, so the base batch size applies.
    for task in tasks:
        if task in ("hellaswag", "wikitext", "winogrande", "arc_easy", "boolq"):
            bs = 1 if task == "wikitext" else batch_size
            runs.append((task, None, limit, bs))
    if "mmlu" in tasks:
        # mmlu's limit is per SUBTASK (57 subtasks), so a uniform limit of
        # 500 still evaluates ~14k docs (~11 min). A dedicated smaller cap
        # keeps the per-trial budget sane.
        runs.append(("mmlu", 5, mmlu_limit if mmlu_limit is not None else limit,
                     min(batch_size, task_batch_cap.get("mmlu", batch_size))))
    if "gsm8k" in tasks:
        runs.append(("gsm8k", 5, gsm8k_limit, batch_size))

    for i, (task, fewshot, task_limit, bs) in enumerate(runs, 1):
        if on_task:
            on_task("start", i, len(runs), task,
                    {"fewshot": fewshot, "limit": task_limit, "batch": bs})
        t0 = time.time()
        # simple_evaluate ignores batch_size for a prebuilt LM, so the
        # per-task batch is set on the HFLM instance directly.
        lm.batch_size_per_gpu = bs
        result = None
        while True:
            try:
                result = lm_eval.simple_evaluate(
                    model=lm,
                    tasks=[task],
                    num_fewshot=fewshot,   # None -> task defaults (hellaswag: 0)
                    batch_size=bs,
                    limit=task_limit,
                )
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                lm.batch_size_per_gpu = bs
                print(f"   {task}: OOM, retrying at batch {bs}", flush=True)
            except Exception as e:
                # Fault draws can destroy the model to the point where
                # aggregation itself fails (wikitext: math.exp of the
                # weighted log-likelihood overflows float range). A
                # collapsed model is a data point, not a crash: record the
                # task as errored and move on to the next task.
                torch.cuda.empty_cache()
                merged[task] = {"error": f"{type(e).__name__}: {e}"}
                if on_task:
                    on_task("error", i, len(runs), task,
                            {"seconds": time.time() - t0, "error": str(e)})
                break
        dt = time.time() - t0
        if result is not None:
            for t, metrics in result["results"].items():
                merged[t] = {k: v for k, v in metrics.items()
                             if isinstance(v, (int, float))}
        if on_task:
            on_task("done", i, len(runs), task,
                    {"seconds": dt, "metrics": merged.get(task, {})})
        # Eager attention peaks are spiky; without this the reserved-but-
        # unallocated pool from the previous task's batch shapes OOMs the
        # next task's first long request.
        torch.cuda.empty_cache()

    return merged
