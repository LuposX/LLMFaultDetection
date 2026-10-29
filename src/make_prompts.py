"""Build data/prompts.jsonl: the fixed prompt set for the fault-effect
metrics (probes.py).

Three strata, chosen to probe different failure modes with the same fault
draw. Every stratum is sampled deterministically (fixed seed) from the
TRAINING split of a public dataset, so the set is reproducible, citable,
and independent of the scored benchmarks (hellaswag, mmlu — plus wikitext
test, which is only used for perplexity):
- wiki: wikitext-2 train passages truncated mid-prose. Open-ended factual
  continuation; measures drift in free prose. Zero-shot (natural format).
- qa: SciQ train questions with their gold answers, 2-shot. Short factual
  answers; sensitive to argmax flips at the answer position.
- math: GSM8K train word problems, 2-shot with worked solutions.
  Chain-of-thought style amplification: a flipped logit early in the
  derivation derails everything after it. Restricted to items with short
  reference solutions so the generation budget covers the full derivation.

Rerunning this script reproduces the identical file.
"""

import json
import os
import random

from datasets import load_dataset

N_WIKI, N_QA, N_MATH = 40, 40, 20
SEED = 42


def wiki_prompts(n):
    """Passage prefixes from wikitext-2 train; gold = the removed tail."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    rng = random.Random(SEED)
    # candidate paragraphs: real prose, neither headers nor empty lines,
    # long enough to cut a prompt off and leave a gold continuation.
    # raw wikitext lines carry a leading space, hence the strip
    cand = [t.strip() for t in ds["text"]
            if len(t.strip()) > 300 and not t.strip().startswith("=")
            and t.strip()[0].isupper()]
    rng.shuffle(cand)
    out = []
    for t in cand:
        cut = t.rfind(" ", 150, 220)  # break inside a word boundary
        if cut <= 0:
            continue
        prompt, gold = t[:cut].rstrip(), t[cut:].strip()
        if len(gold) < 40:  # keep the continuation meaningful
            continue
        out.append({"stratum": "wiki", "prompt": prompt, "gold": gold})
        if len(out) == n:
            return out
    raise RuntimeError("not enough wikitext candidates")


def qa_prompts(n):
    """SciQ train questions; gold = the crowdsourced correct answer.

    Restricted to short, self-contained questions with short answers so the
    greedy generation has a well-defined endpoint. Two fixed in-context
    exemplars steer the base model to short answers — without them it
    rambles and the 'short factual answer' probe is lost.
    """
    ds = load_dataset("sciq", split="train")
    rng = random.Random(SEED + 2)
    cand = []
    for r in ds:
        q, a = r["question"].strip(), r["correct_answer"].strip()
        if not (40 <= len(q) <= 300 and q.endswith("?")):
            continue
        if not a or len(a.split()) > 5 or "\n" in a:
            continue
        cand.append((q, a))
    rng.shuffle(cand)
    header = ("Answer each question with a short phrase.\n\n"
              "Question: What is the largest ocean on Earth?\n"
              "Answer: Pacific Ocean\n\n"
              "Question: What is the chemical symbol for gold?\n"
              "Answer: Au\n\n")
    return [{"stratum": "qa",
             "prompt": f"{header}Question: {q}\nAnswer:", "gold": a}
            for q, a in cand[:n]]


def math_prompts(n):
    """GSM8K train word problems; gold = the final numeric answer.

    Restricted to items whose reference solution is short, so the chain of
    thought fits into the generation budget. Two fixed exemplars (the
    canonical first two GSM8K train items) establish the step-by-step
    answer format.
    """
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(SEED + 3)
    cand = []
    for r in ds:
        q = r["question"].strip()
        solution = r["answer"]
        if "####" not in solution:
            continue
        gold = solution.split("####")[-1].strip().replace(",", "")
        if not gold.replace("-", "").isdigit():
            continue
        if len(q) > 400 or len(solution) > 400:
            continue
        cand.append((q, gold))
    rng.shuffle(cand)
    ex1 = ("Natalia sold clips to 48 of her friends in April, and then she "
           "sold half as many clips in May. How many clips did Natalia sell "
           "altogether in April and May?",
           "In April Natalia sold 48 clips. In May she sold half as many, "
           "so 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. "
           "The answer is 72.")
    ex2 = ("Weng earns $12 an hour for babysitting. Yesterday, she just did "
           "50 minutes of babysitting. How much did she earn?",
           "Weng earns 12 / 60 = $0.2 per minute. Working 50 minutes, she "
           "earned 0.2 x 50 = $10. The answer is 10.")
    header = (f"Q: {ex1[0]}\nA: {ex1[1]}\n\n"
              f"Q: {ex2[0]}\nA: {ex2[1]}\n\n")
    return [{"stratum": "math",
             "prompt": f"{header}Q: {q}\nA:", "gold": gold}
            for q, gold in cand[:n]]


def main():
    os.makedirs("data", exist_ok=True)
    rows = wiki_prompts(N_WIKI) + qa_prompts(N_QA) + math_prompts(N_MATH)
    with open("data/prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote data/prompts.jsonl with {len(rows)} prompts")


if __name__ == "__main__":
    main()