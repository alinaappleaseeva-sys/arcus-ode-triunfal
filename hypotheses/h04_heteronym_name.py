"""
H04: Heteronym name candidates as the flag
==========================================

Hypothesis
----------
The SSH challenge shows the Ode Triunfal poem stanza and asks for a flag.
The poem's author is Álvaro de Campos — a heteronym deliberately absent
from the model's special-token vocabulary. The flag may simply be his name
(or a canonical variant) rather than a byte sequence hidden inside the model.

This script does three things:

  1. Ranks ALL candidate strings by their conditional log-probability under
     the model given the poem as context.  Higher log-prob = more "natural"
     completion for this model.  Top candidates → best SSH submissions.

  2. Records the top-20 next-token predictions immediately after the poem,
     to check whether the model spontaneously heads toward "Álvaro" or any
     other name.

  3. Generates (temperature=0.3) from the poem prefix and checks if a
     recognisable name appears in the output.

Candidates tested
-----------------
  Tier 1 — direct name variants (Álvaro de Campos):
    alvaro_de_campos, álvaro_de_campos, alvaro de campos,
    Álvaro de Campos, Álvaro De Campos, ALVARO DE CAMPOS

  Tier 2 — other Pessoa heteronyms / personas:
    fernando pessoa, Fernando Pessoa,
    alberto caeiro, Alberto Caeiro,
    ricardo reis, Ricardo Reis,
    bernardo soares, Bernardo Soares

  Tier 3 — meta / challenge-specific strings:
    ode_triunfal, ode triunfal, Ode Triunfal,
    pessoa, campos, heteronym, heterónimo

  Tier 4 — known flag-format guesses:
    flag{alvaro_de_campos}, flag{álvaro_de_campos},
    flag{ode_triunfal}, flag{campos}

Results are sorted descending by log-probability.
Top-8 are printed as ACTION ITEMS to type into the SSH prompt.

Status: TODO
"""

import math
import torch
import torch.nn.functional as F
from model import load_model, generate, decode


POEM = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "        Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "    Só porque houve outra e foram humanos Virgílio e Platão"
)

CANDIDATES = [
    # Tier 1 — Álvaro de Campos variants
    "alvaro_de_campos",
    "álvaro_de_campos",
    "alvaro de campos",
    "Álvaro de Campos",
    "Álvaro De Campos",
    "ALVARO DE CAMPOS",
    "alvaro",
    "campos",
    # Tier 2 — other heteronyms
    "fernando pessoa",
    "Fernando Pessoa",
    "alberto caeiro",
    "Alberto Caeiro",
    "ricardo reis",
    "Ricardo Reis",
    "bernardo soares",
    "Bernardo Soares",
    # Tier 3 — meta
    "ode_triunfal",
    "ode triunfal",
    "Ode Triunfal",
    "pessoa",
    "heteronym",
    "heterónimo",
    # Tier 4 — explicit flag-format guesses
    "flag{alvaro_de_campos}",
    "flag{álvaro_de_campos}",
    "flag{ode_triunfal}",
    "flag{campos}",
    "flag{heteronym}",
    "flag{pessoa}",
]


def run():
    model, special, cfg = load_model('ode.pt')
    poem_ids = list(POEM.encode('utf-8'))
    id_to_special = {v: k for k, v in special.items()}

    # ── Part 1: rank candidates by conditional log-prob given poem ────────────
    print("=" * 70)
    print("  H04 — Candidate ranking by P(candidate | poem)")
    print("=" * 70)

    scores = []
    for cand in CANDIDATES:
        cand_ids = list(cand.encode('utf-8'))
        # Score: sum_{i} log P(cand[i] | poem + cand[:i])
        context = poem_ids + cand_ids
        x = torch.tensor([context], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        logprob = 0.0
        for i, target in enumerate(cand_ids):
            pos = len(poem_ids) + i - 1   # predicts token at pos+1
            logprob += F.log_softmax(logits[0, pos], dim=-1)[target].item()
        ppl = math.exp(-logprob / len(cand_ids))
        scores.append((cand, logprob, ppl))

    scores.sort(key=lambda t: t[1], reverse=True)

    print(f"\n{'Rank':<5} {'LogProb':>10} {'PPL':>8}  Candidate")
    print("-" * 62)
    for rank, (cand, lp, ppl) in enumerate(scores, 1):
        marker = "  ← TRY ON SSH" if rank <= 5 else ""
        print(f"  {rank:<3} {lp:>10.2f} {ppl:>8.1f}  {cand!r}{marker}")

    # ── Part 2: top-20 next tokens after poem (no prefix) ────────────────────
    print("\n" + "=" * 70)
    print("  Top-20 next tokens immediately after poem (no heteronym prefix)")
    print("=" * 70)

    x = torch.tensor([poem_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    last_logits = logits[0, -1]
    probs = F.softmax(last_logits, dim=-1)
    top20_probs, top20_ids = torch.topk(probs, 20)

    print(f"\n  {'ID':>5}  {'Prob':>8}  Token")
    print("  " + "-" * 42)
    for tok_id, prob in zip(top20_ids.tolist(), top20_probs.tolist()):
        if tok_id < 256:
            display = repr(chr(tok_id)) if 32 <= tok_id < 127 else f'0x{tok_id:02x}'
        else:
            display = id_to_special.get(tok_id, f'special_{tok_id}')
        print(f"  {tok_id:>5}  {prob:>8.4f}  {display}")

    # ── Part 3: generation from poem at low temperature ───────────────────────
    print("\n" + "=" * 70)
    print("  Generation from poem  (temperature=0.3, max_new=80, seed=42)")
    print("=" * 70)

    torch.manual_seed(42)
    gen_ids = generate(model, poem_ids, max_new=80,
                       temperature=0.3, greedy=False, stop_at_close=False)
    print(f"\n  [no prefix]      {decode(gen_ids)!r}")

    torch.manual_seed(42)
    gen_ids2 = generate(model,
                        [special['<|fernando_pessoa|>']] + poem_ids,
                        max_new=80, temperature=0.3, greedy=False,
                        stop_at_close=False)
    print(f"  [pessoa prefix]  {decode(gen_ids2)!r}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ACTION ITEMS — try these on SSH server (ranked by model log-prob):")
    print("=" * 70)
    for rank, (cand, lp, ppl) in enumerate(scores[:8], 1):
        print(f"  {rank}. {cand}")
    print()
