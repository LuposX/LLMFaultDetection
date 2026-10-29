"""Shared greedy generation: one decode path for the whole project.

probes (fault-effect metrics), check_determinism and the later
detection validator all need the same greedy decoding. When the detection
criterion is exact token matching, two generation implementations with
subtly different padding or trimming behavior would be a correctness
hazard, so there is exactly one.

Left padding + attention mask, prompts sorted by length so short prompts
don't pad out long ones, generated ids trimmed at the first EOS/pad.
"""

import torch


@torch.no_grad()
def generate_greedy(model, tokenizer, prompts, max_new_tokens, batch_size,
                    device):
    """Greedy generation for a list of prompts. Returns (texts, token_ids)."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [""] * len(prompts)
    ids_out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]),
                   reverse=True)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt",
                        padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
        for row, i in enumerate(idx):
            ids = gen[row].tolist()
            for j, t in enumerate(ids):
                if t == eos or t == pad:
                    ids = ids[:j]
                    break
            ids_out[i] = ids
            texts[i] = tokenizer.decode(ids, skip_special_tokens=True)
    return texts, ids_out
