import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM
from evalbench import evaluate
from run_experiment import load_model

m, _ = load_model("Qwen/Qwen3-0.6B", "cuda")
print("=== wikitext only, batch should be 1, limit 5", flush=True)
res = evaluate(m, ["wikitext"], limit=5, batch_size=8)
print("RESULT:", res, flush=True)
