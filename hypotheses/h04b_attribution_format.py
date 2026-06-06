"""
H04b: Attribution/signature format around Fernando Pessoa
=========================================================

Hypothesis
----------
H04 showed that "Fernando Pessoa" has anomalously low PPL=4.0 after the poem
(10× better than Álvaro de Campos at PPL=6.2). This is NOT a coincidence — it
signals that in the training corpus, THIS POEM was reliably followed by a
Fernando Pessoa attribution/signature in a specific format.

The flag is therefore NOT the bare name "Fernando Pessoa" (already tested, rejected)
but rather the FULL ATTRIBUTION PATTERN as it appeared in training data.

Candidates to investigate
--------------------------
Literary attribution formats in Portuguese corpora typically look like:

  poem\n\n— Fernando Pessoa              (em-dash + space)
  poem\n\n                    Fernando Pessoa   (right-aligned)
  poem\n\nFernando Pessoa                (plain newlines)
  poem\nFernando Pessoa                  (single newline)
  poem — Fernando Pessoa               (inline dash)
  poem\n\n(Fernando Pessoa)             (parenthesised)
  poem\n\n[Fernando Pessoa]             (bracketed)
  poem\n\nFernando Pessoa (Álvaro de Campos)   (with heteronym in parens)
  poem\n\nin: Fernando Pessoa           (anthology format)
  poem\n\nFernando Pessoa\nÁlvaro de Campos    (stacked lines)
  poem\n\nde Fernando Pessoa            (Portuguese genitive "of")
  poem\n\nPor Fernando Pessoa           (Portuguese "By Fernando Pessoa")
  poem\n\npoema de Fernando Pessoa      (labelled)

Method
------
For each candidate suffix, compute:
  P(suffix | poem) = exp(log_prob(poem + suffix) - log_prob(poem))
  presented as PPL = exp(-log_prob(suffix | poem) / len(suffix_bytes))

The lowest-PPL candidate is the most likely training-corpus format →
TOP-5 submitted to SSH server.

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

# ── Attribution format candidates ─────────────────────────────────────────────
# Group A — em-dash/dash signatures (most common in European literary editions)
# Group B — plain newline formats
# Group C — structural variants (parentheses, brackets, prefixes)
# Group D — compound: Pessoa + Campos (the heteronym pair)
# Group E — greedy continuation variants (what the model actually generates)

ATTRIBUTION_SUFFIXES = [
    # Group A — dash-style attribution
    "\n\n— Fernando Pessoa",
    "\n— Fernando Pessoa",
    " — Fernando Pessoa",
    "\n\n– Fernando Pessoa",          # en-dash
    "\n– Fernando Pessoa",
    "\n\n- Fernando Pessoa",          # hyphen
    "\n- Fernando Pessoa",

    # Group B — plain newlines
    "\n\nFernando Pessoa",
    "\nFernando Pessoa",
    "\n\n\nFernando Pessoa",

    # Group C — structural variants
    "\n\n(Fernando Pessoa)",
    "\n\n[Fernando Pessoa]",
    "\n\nde Fernando Pessoa",
    "\n\nPor Fernando Pessoa",
    "\n\npoema de Fernando Pessoa",
    "\n\nin: Fernando Pessoa",
    "\n\n                    Fernando Pessoa",  # right-aligned (20 spaces)

    # Group D — Pessoa + Campos compound attributions
    "\n\nFernando Pessoa\nÁlvaro de Campos",
    "\n\n— Fernando Pessoa\n   (Álvaro de Campos)",
    "\n\nFernando Pessoa (Álvaro de Campos)",
    "\n\nFernando Pessoa, sob o nome de Álvaro de Campos",
    "\n\nÁlvaro de Campos (Fernando Pessoa)",

    # Group E — greedy continuation / what the model actually generates
    " de Sá-Carneiro",
    " de Sá-Carneiro.",
    "\n\n— Álvaro de Campos",
    "\n\nÁlvaro de Campos",
    "\n— Álvaro de Campos",

    # Group F — pessoa special token as prefix then attribution
    # (handled separately below with heteronym prefix)
]


def score_suffix(model, poem_ids: list[int], suffix: str) -> tuple[float, float]:
    """
    Returns (cond_log_prob, ppl) of suffix given poem as context.
    cond_log_prob = sum log P(c_i | poem + c_{<i})
    ppl           = exp(-cond_log_prob / len(suffix_bytes))
    """
    suffix_ids = list(suffix.encode('utf-8'))
    context    = poem_ids + suffix_ids
    x = torch.tensor([context], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    lp = 0.0
    for i, target in enumerate(suffix_ids):
        pos = len(poem_ids) + i - 1
        lp += F.log_softmax(logits[0, pos], dim=-1)[target].item()
    ppl = math.exp(-lp / max(len(suffix_ids), 1))
    return lp, ppl


def run():
    model, special, cfg = load_model('ode.pt')
    poem_ids = list(POEM.encode('utf-8'))

    print("=" * 70)
    print("  H04b — Attribution format ranking by P(suffix | poem)")
    print("  Baseline: bare 'Fernando Pessoa' had PPL=4.0")
    print("=" * 70)

    scores = []
    for suffix in ATTRIBUTION_SUFFIXES:
        lp, ppl = score_suffix(model, poem_ids, suffix)
        scores.append((suffix, lp, ppl))

    # Also score with <|fernando_pessoa|> prefix for each suffix
    print("\n--- Without heteronym prefix ---")
    scores.sort(key=lambda t: t[1], reverse=True)  # highest log_prob first
    print(f"\n  {'LogProb':>10} {'PPL':>8}  Suffix")
    print("  " + "-" * 62)
    for rank, (sfx, lp, ppl) in enumerate(scores, 1):
        marker = "  ← TRY" if rank <= 5 else ""
        sfx_display = repr(sfx)[:50]
        print(f"  {lp:>10.2f} {ppl:>8.2f}  {sfx_display}{marker}")

    # With pessoa prefix
    print("\n--- With <|fernando_pessoa|> prefix ---")
    prefix_id = special['<|fernando_pessoa|>']
    scores_p = []
    for suffix in ATTRIBUTION_SUFFIXES:
        suffix_ids = list(suffix.encode('utf-8'))
        context_p  = [prefix_id] + poem_ids + suffix_ids
        x = torch.tensor([context_p], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        lp = 0.0
        for i, target in enumerate(suffix_ids):
            pos = 1 + len(poem_ids) + i - 1   # +1 for prefix token
            lp += F.log_softmax(logits[0, pos], dim=-1)[target].item()
        ppl = math.exp(-lp / max(len(suffix_ids), 1))
        scores_p.append((suffix, lp, ppl))

    scores_p.sort(key=lambda t: t[1], reverse=True)
    print(f"\n  {'LogProb':>10} {'PPL':>8}  Suffix")
    print("  " + "-" * 62)
    for rank, (sfx, lp, ppl) in enumerate(scores_p, 1):
        marker = "  ← TRY" if rank <= 5 else ""
        sfx_display = repr(sfx)[:50]
        print(f"  {lp:>10.2f} {ppl:>8.2f}  {sfx_display}{marker}")

    # ── Greedy continuation after poem ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Greedy continuation of poem (first 80 bytes)")
    print("=" * 70)
    torch.manual_seed(42)
    gen = generate(model, poem_ids, max_new=80, greedy=True, stop_at_close=False)
    print(f"\n  {decode(gen)!r}")

    # ── Greedy with pessoa prefix ──────────────────────────────────────────────
    torch.manual_seed(42)
    gen_p = generate(model, [prefix_id] + poem_ids, max_new=80,
                     greedy=True, stop_at_close=False)
    print(f"\n  [+pessoa] {decode(gen_p)!r}")

    # ── What comes right after "\n\n— Fernando Pessoa"? ──────────────────────
    # Is there more text after the attribution?
    print("\n" + "=" * 70)
    print("  What follows the top attribution formats?")
    print("=" * 70)
    top3_no_prefix = [s for s, _, _ in scores[:3]]
    for sfx in top3_no_prefix:
        seed = poem_ids + list(sfx.encode('utf-8'))
        torch.manual_seed(42)
        gen2 = generate(model, seed, max_new=40, greedy=True, stop_at_close=False)
        print(f"\n  After poem + {sfx!r}:")
        print(f"    → {decode(gen2)!r}")

    # ── ACTION ITEMS ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TOP-5 to try on SSH (no prefix, highest log-prob):")
    print("=" * 70)
    for rank, (sfx, lp, ppl) in enumerate(scores[:5], 1):
        # The SSH answer is just the SUFFIX itself (strip leading newlines/dashes)
        clean = sfx.lstrip('\n— ').strip()
        print(f"  {rank}. raw suffix: {sfx!r}")
        print(f"     → SSH candidate: {clean!r}  (ppl={ppl:.2f})")
    print()
