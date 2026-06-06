"""
H01: Perplexity test
====================
Hypothesis: The poem stanza has lowest perplexity under one specific
heteronym — that heteronym is the answer / flag.

Result: FAILED
All four heteronyms give perplexity ~297. Max delta < 0.2.
Best fit: <|ricardo_reis|> (ppl=297.18) but margin is negligible.
"""

import math
from model import load_model, log_probability

POEM = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "        Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "    Só porque houve outra e foram humanos Virgílio e Platão"
)


def run():
    model, special, cfg = load_model('ode.pt')
    poem_ids = list(POEM.encode('utf-8'))

    print(f"Poem length: {len(poem_ids)} tokens\n")
    print(f"{'Heteronym':<35} {'log_prob':>12} {'perplexity':>12}")
    print("-" * 62)

    results = {}
    for name in ['<|fernando_pessoa|>', '<|alberto_caeiro|>',
                 '<|ricardo_reis|>', '<|bernardo_soares|>']:
        lp  = log_probability(model, special[name], poem_ids)
        ppl = math.exp(-lp / len(poem_ids))
        results[name] = ppl
        print(f"{name:<35} {lp:>12.2f} {ppl:>12.4f}")

    best = min(results, key=results.get)
    print(f"\nBest fit: {best} (ppl={results[best]:.4f})")
    print("\nVerdict: FAILED — differences too small to be meaningful.")
