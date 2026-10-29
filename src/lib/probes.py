"""Generation-based fault-effect metrics: token match, KL divergence,
embedding distance.

For the fixed prompt set (data/prompts.jsonl), the faulty model is compared
against the healthy model on three levels:

- token match: fraction of the healthy model's generated token ids that the
  faulty model reproduces (greedy decoding), plus the first-divergence
  position. Measures when damage becomes *visible* in the output; mostly
  tracks the position of the first argmax flip.
- Jensen-Shannon divergence: teacher-forced on the healthy model's own
  output, divergence of next-token distributions over the generated
  positions. Dense signal: the distribution moves long before the argmax
  does, so this is sensitive from the first flip. Same quantity family as
  the paper's d(x, f) between softmax outputs. Bounded by log 2, hence
  monotone in damage; KL was tried and dropped — it explodes when a damaged
  model assigns ~0 probability to healthy-certain tokens (KL -> 3e36 at 13
  flips) and dips for collapsed flat models (10.4 at 100 flips), so it is
  NOT monotone in damage.
- Laya semantic judge (laya_same): P(same meaning) from the laya
  decision model (convaiinnovations/laya, noul question) given the healthy
  reference and faulty output as state. Runs on CPU — the GPU path
  segfaults on this ROCm/RDNA4 build (attn_implementation eager does not
  fix it). Sanity-checked: identical 0.918, paraphrase 0.944, unrelated
  0.027.
- embedding distance: 1 - cosine between sentence embeddings of the two
  generated texts. In-field observable (needs only generated text) and
  robust to surface divergence, but has a dead zone while greedy outputs
  are still identical.

The healthy reference (generated outputs, token ids, log-softmax over the
generated positions, embeddings) is computed once on the clean model and
cached in CPU RAM (~2 MB per prompt for the log-softmax). Each trial then
costs one generation pass plus one teacher-forced forward per prompt.
"""

import json
import math
import os

import torch

from .generation import generate_greedy

LOG2 = math.log(2.0)


class ProbeSet:
    def __init__(self, model, tokenizer, prompt_file="data/prompts.jsonl",
                 max_new_tokens=96, gen_batch=8,
                 embed_model="all-MiniLM-L6-v2"):
        self.prompts = [json.loads(line) for line in open(prompt_file)
                        if line.strip()]
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.gen_batch = gen_batch
        self.device = next(model.parameters()).device
        self._model = model
        self.embed_model_name = embed_model
        self.embedder = None
        # healthy reference, filled by compute_baseline()
        self.healthy_texts = None
        self.healthy_ids = None
        self.healthy_logp = None  # fp16 CPU tensors, [glen, vocab]
        self.healthy_emb = None
        self.laya_agent = None  # semantic judge, lazy-loaded on CPU
        self.laya_identical = None  # judge P(same) on identical pairs

    @property
    def n_prompts(self):
        return len(self.prompts)

    @torch.no_grad()
    def compute_baseline(self):
        """Run the clean model once and cache the healthy reference.

        Must be called while the model is still untouched (before any
        injection).
        """
        prompts = [p["prompt"] for p in self.prompts]
        self.healthy_texts, self.healthy_ids = generate_greedy(
            self._model, self.tokenizer, prompts,
            self.max_new_tokens, self.gen_batch, self.device)
        # Cache the healthy next-token log-distribution over the generated
        # positions, teacher-forced on prompt + healthy output. Storing fp16
        # on CPU: ~64 x 152k x 2 B ~ 2 MB per prompt. Small probabilities
        # underflow to zero in fp16, which is harmless here: they are
        # multiplied by P_healthy ~ 0 in the KL sum.
        self.healthy_logp = []
        for p, gen_ids in zip(prompts, self.healthy_ids):
            if not gen_ids:
                self.healthy_logp.append(None)
                continue
            seq = self.tokenizer(p, return_tensors="pt").input_ids[0] \
                       .to(self.device)
            full = torch.cat([seq, torch.tensor(gen_ids, device=self.device)])
            plen, glen = seq.numel(), len(gen_ids)
            logits = self._model(full.unsqueeze(0)).logits[0].float()
            # logits[t] predicts token t+1; generated token j (full[plen+j])
            # is predicted by logits[plen-1+j]
            lp = torch.log_softmax(logits[plen - 1:plen - 1 + glen], dim=-1)
            self.healthy_logp.append(lp.to(torch.float16).cpu())
        from sentence_transformers import SentenceTransformer
        # The SDPA segfault on this torch/ROCm/gfx1201 build is not limited
        # to the LLM: transformers defaults BERT-family encoders (MiniLM)
        # to SDPA attention, which crashes the same way. Force eager.
        self.embedder = SentenceTransformer(
            self.embed_model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"attn_implementation": "eager"})
        self.healthy_emb = self.embedder.encode(
            self.healthy_texts, normalize_embeddings=True)
        # semantic judge on CPU: laya's GPU path segfaults on this ROCm
        # build even with eager attention (see module docstring)
        os.environ.setdefault("USE_TF", "0")  # transformers/TF import deadlock guard
        import laya
        self.laya_agent = laya.load("convaiinnovations/laya", device="cpu")
        self.laya_q = {"same_meaning": {
            "type": "noul",
            "instructions": "Do the two responses in the state have the same meaning?"}}
        # the judge's own ceiling: P(same) on identical pairs — the value
        # laya_same would have at rho=0 (the judge is imperfect, so this is
        # < 1 and measured on the real prompt set, not assumed)
        vals = []
        for t in self.healthy_texts:
            r = self.laya_agent.predict({"reference": t, "observed": t},
                                        self.laya_q)
            vals.append(r["answers"]["same_meaning"]["noul"])
        self.laya_identical = sum(vals) / len(vals)
        return self.healthy_texts

    @torch.no_grad()
    def evaluate(self, model):
        """Fault-effect metrics for the model as currently in memory."""
        self._model = model
        prompts = [p["prompt"] for p in self.prompts]
        texts, ids = generate_greedy(self._model, self.tokenizer, prompts,
                               self.max_new_tokens, self.gen_batch,
                               self.device)
        rows = []
        for i, p in enumerate(self.prompts):
            row = {"stratum": p["stratum"], "match": None, "first_div": None,
                   "js": None, "cos": None, "laya_same": None}
            h_ids, f_ids = self.healthy_ids[i], ids[i]
            # token match over the healthy generation
            if h_ids:
                n = min(len(h_ids), len(f_ids))
                if n:
                    eq = [a == b for a, b in zip(h_ids[:n], f_ids[:n])]
                    row["match"] = sum(eq) / len(h_ids)
                    row["first_div"] = next(
                        (j for j, e in enumerate(eq) if not e), None)
            # KL on the teacher-forced healthy trajectory
            if self.healthy_logp[i] is not None:
                gen_ids = self.healthy_ids[i]
                seq = self.tokenizer(p["prompt"], return_tensors="pt") \
                          .input_ids[0].to(self.device)
                full = torch.cat(
                    [seq, torch.tensor(gen_ids, device=self.device)])
                plen, glen = seq.numel(), len(gen_ids)
                logits = self._model(full.unsqueeze(0)).logits[0].float()
                lp_f = torch.log_softmax(
                    logits[plen - 1:plen - 1 + glen], dim=-1)
                lp_h = self.healthy_logp[i].float().to(lp_f.device)
                # Jensen-Shannon: bounded by log 2, monotone damage signal.
                p = lp_h.exp()
                q = lp_f.exp()
                m = 0.5 * (p + q)
                js = (0.5 * (p * (lp_h - (m + 1e-12).log())).sum(-1)
                      + 0.5 * (q * (lp_f - (m + 1e-12).log())).sum(-1)
                      ).mean().item()
                row["js"] = max(0.0, min(js, LOG2)) if math.isfinite(js) else LOG2
            # Laya semantic judge: P(same meaning) for (healthy, faulty) text
            judge = self.laya_agent.predict(
                {"reference": self.healthy_texts[i], "observed": texts[i]},
                self.laya_q)
            row["laya_same"] = judge["answers"]["same_meaning"]["noul"]
            rows.append(row)
        # embedding cosine in one batch
        emb = self.embedder.encode(texts, normalize_embeddings=True)
        for i, row in enumerate(rows):
            row["cos"] = float(self.healthy_emb[i] @ emb[i])
        return self._aggregate(rows)

    @staticmethod
    def _aggregate(rows):
        def mean(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        def agg(rs):
            if not rs:
                return None
            div = [r["first_div"] for r in rs if r["first_div"] is not None]
            return {
                "token_match": mean(r["match"] for r in rs),
                "frac_diverged": len(div) / len(rs),
                "first_div_mean": mean(div),
                "js": mean(r["js"] for r in rs),
                "laya_same": mean(r["laya_same"] for r in rs),
                "embed_cos": mean(r["cos"] for r in rs),
                "n": len(rs),
            }

        out = {"overall": agg(rows)}
        for stratum in sorted({r["stratum"] for r in rows}):
            out[stratum] = agg([r for r in rows if r["stratum"] == stratum])
        return out
