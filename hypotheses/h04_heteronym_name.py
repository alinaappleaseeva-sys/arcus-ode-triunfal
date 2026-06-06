"""
H04: Flag is the missing heteronym name
========================================
Hypothesis: The poem "Ode Triunfal" is by Álvaro de Campos —
a heteronym of Pessoa that is NOT in the model's vocab.
The flag is the name of this missing heteronym.

Candidates to test on SSH server:
  - alvaro_de_campos
  - álvaro_de_campos
  - alvaro de campos
  - Álvaro de Campos
  - campos

This script validates the hypothesis by checking:
1. Is "álvaro" or "campos" generatable by the model?
2. Does the model assign high probability to these tokens in context?
3. What does the model say is the "author" of the poem stanza?

Status: TODO — needs SSH server test
"""

import math
import torch
import torch.nn.functional as F
from model import load_model, log_probability

POEM = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "        Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "    Só porque houve outra e foram humanos Virgílio e Platão"
)

CANDIDATES = [
    "alvaro_de_campos",
    "álvaro_de_campos",
    "alvaro de campos",
    "Álvaro de Campos",
    "campos",
    "alvaro",
]


@torch.no_grad()
def run():
    model, special, cfg = load_model('ode.pt')
    poem_ids = list(POEM.encode('utf-8'))

    print("=== H04: Missing heteronym — Álvaro de Campos ===\n")
    print("The poem 'Ode Triunfal' is by Álvaro de Campos.")
    print("He is NOT in the model vocabulary (only 4 heteronyms present).")
    print("Hypothesis: the flag is his name in some form.\n")

    # 1. Conditional log-probability of each candidate string given the poem.
    #    We use log_probability(poem + candidate) - log_probability(poem)
    #    to isolate P(candidate | poem_context).
    #
    #    Note: log_probability() returns a SUM over tokens (not average).
    #    Perplexity = exp(-sum / n_tokens) — division by len() is done here.
    print("1. Conditional log-prob of candidates given poem (pessoa prefix):")
    print(f"   {'Candidate':<25} {'cond_log_prob':>14} {'ppl':>10}")
    print("   " + "-"*52)

    prefix_id = special['<|fernando_pessoa|>']
    lp_poem   = log_probability(model, prefix_id, poem_ids)

    for cand in CANDIDATES:
        try:
            cand_ids = list(cand.encode('utf-8'))
            lp_full  = log_probability(model, prefix_id, poem_ids + cand_ids)
            lp_cond  = lp_full - lp_poem            # sum over cand tokens only
            ppl      = math.exp(-lp_cond / len(cand_ids))
            print(f"   {cand:<25} {lp_cond:>14.2f} {ppl:>10.4f}")
        except Exception as e:
            print(f"   {cand:<25} ERROR: {e}")

    # 2. Top-20 most likely next tokens after the poem ends
    print("\n2. Top-20 tokens after poem ends (pessoa prefix):")
    seed = [prefix_id] + poem_ids
    x    = torch.tensor([seed], dtype=torch.long)
    logits = model(x)[0, -1]
    id_to_special = {v: k for k, v in special.items()}
    top  = logits.topk(20)
    for val, idx in zip(top.values.tolist(), top.indices.tolist()):
        idx = int(idx)
        if idx < 256:
            ch = repr(chr(idx)) if 32 <= idx < 127 else f'0x{idx:02x}'
        else:
            ch = id_to_special.get(idx, f'special_{idx}')
        print(f"   id={idx:3d}  {ch:<25}  logit={val:+.3f}")

    # 3. SSH test instructions
    print("\n3. Candidates to try on SSH server (in priority order):")
    for i, cand in enumerate(CANDIDATES, 1):
        print(f"   {i}. {cand}")
    print("\n   → ssh augustalabs.ai → enter at 'flag:' prompt")
