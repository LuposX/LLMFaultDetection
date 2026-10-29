"""Model loading shared by every entry point.

Central place for the load-time decisions that shape results (see
DESIGN.md): bf16 weights matching a bf16 deployment, eager attention
(SDPA segfaults on this torch/ROCm/RDNA4 build), eval mode.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
        # SDPA segfaults on RDNA4 (gfx1201) with the nixpkgs torch 2.9.1
        # build; plain eager attention works and is fast enough for 0.6B.
        attn_implementation="eager",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
