import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inject import inject_model, snapshot_weights, restore_weights, unique_parameters

m, _ = None, None
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
                                         device_map="cuda", attn_implementation="eager").eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
x = tok("The weather today is", return_tensors="pt").to("cuda")

def n_diff(model, snap):
    n = 0
    for name, p in unique_parameters(model):
        n += int((p.detach().cpu() != snap[name]).sum())
    return n

def gen():
    with torch.no_grad():
        return tok.decode(m.generate(**x, max_new_tokens=8, do_sample=False)[0].tolist())

snap = snapshot_weights(m)
print("clean gen:", repr(gen()), flush=True)

inject_model(m, 1e-7, 0)
print("rho=1e-7: params changed:", n_diff(m, snap), flush=True)
print("rho=1e-7 gen:", repr(gen()), flush=True)

restore_weights(m, snap)
print("restored equal:", n_diff(m, snap) == 0, flush=True)

inject_model(m, 1e-5, 0)
print("rho=1e-5: params changed:", n_diff(m, snap), flush=True)
print("rho=1e-5 gen:", repr(gen()), flush=True)
