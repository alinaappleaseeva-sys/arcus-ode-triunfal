"""
H07: Beam search over printable ASCII — find highest-probability completions
============================================================================

Hypothesis
----------
All previous experiments used greedy generation (gets stuck in loops) or
scored hand-picked candidate strings. Neither approach answers the question:

  "What is the globally most likely SHORT completion after this poem?"

Beam search explores all paths simultaneously and finds the highest-probability
sequences without requiring us to guess the candidate first.

Key difference from H05a:
  - H05a: scored specific candidates we already thought of
  - H07: discovers the best candidates from scratch via beam search

We run beam search under 4 contexts:
  A. SSH 4-line stanza (what the server shows)
  B. Full poem last stanza
  C. Full poem (last 900 bytes)
  D. SSH stanza + [_special] prefix

For each context, beam search finds:
  - Top-10 completions of length 5–30 printable ASCII chars
  - Top-10 completions containing only [a-zA-Z0-9_{}] (flag-like chars)
  - Lowest-PPL completion overall

If the flag is any printable string, beam search will find it or get very close.

Status: TODO
"""

import math
import heapq
import torch
import torch.nn.functional as F
from model import load_model, decode


# ── Contexts ─────────────────────────────────────────────────────────────────
POEM_SSH = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "        Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "    Só porque houve outra e foram humanos Virgílio e Platão"
)

POEM_LAST_STANZA = """\
Eia comboios, eia pontes, eia hotéis à hora do jantar,
Eia aparelhos de todas as espécies, férreos, brutos, mínimos,
Instrumentos de precisão, aparelhos de triturar, de cavar,
Engenhos brocas, máquinas rotativas!
Eia! eia! eia!
Eia electricidade, nervos doentes da Matéria!
Eia telegrafia-sem-fios, simpatia metálica do Inconsciente!
Eia túneis, eia canais, Panamá, Kiel, Suez!
Eia todo o passado dentro do presente!
Eia todo o futuro já dentro de nós! eia!
Eia! eia! eia!
Frutos de ferro e útil da árvore-fábrica cosmopolita!
Eia! eia! eia! eia-hô-ô-ô!
Nem sei que existo para dentro. Giro, rodeio, engenho-me.
Engatam-me em todos os comboios.
Içam-me em todos os cais.
Giro dentro das hélices de todos os navios.
Eia! eia-hô! eia!
Eia! sou o calor mecânico e a electricidade!
Eia! e os rails e as casas de máquinas e a Europa!
Eia e hurrah por mim-tudo e tudo, máquinas a trabalhar, eia!

Galgar com tudo por cima de tudo! Hup-lá!

Hup-lá, hup-lá, hup-lá-hô, hup-lá!
Hé-la! He-hô! H-o-o-o-o!
Z-z-z-z-z-z-z-z-z-z-z-z!

Ah não ser eu toda a gente e toda a parte!"""

FULL_POEM_PATH = "/tmp/ode_triunfal_full.txt"


# ── Beam search ───────────────────────────────────────────────────────────────
PRINTABLE_ASCII = list(range(32, 127))   # space through ~
FLAG_CHARS      = (
    list(range(ord('a'), ord('z') + 1)) +
    list(range(ord('A'), ord('Z') + 1)) +
    list(range(ord('0'), ord('9') + 1)) +
    [ord('_'), ord('{'), ord('}'), ord('-')]
)


@torch.no_grad()
def beam_search(model, context_ids, max_new=30, beam_width=10,
                allowed_tokens=None, stop_at=125, min_len=4):
    """
    Beam search over `allowed_tokens` (default: printable ASCII).
    Returns list of (log_prob, token_ids, text) sorted best-first.
    Stops a beam when it hits `stop_at` token (default: } = 125).
    Only returns beams that reached stop_at with length >= min_len.
    """
    if allowed_tokens is None:
        allowed_tokens = PRINTABLE_ASCII

    # Each beam: (neg_log_prob, ids_so_far)  — min-heap by neg_lp (best = smallest)
    beams     = [(0.0, context_ids[:])]
    completed = []

    for step in range(max_new):
        new_beams = []
        for neg_lp, ids in beams:
            x = torch.tensor([ids[-1024:]], dtype=torch.long)
            logits   = model(x)
            log_probs = F.log_softmax(logits[0, -1], dim=-1)

            for tok in allowed_tokens + [stop_at]:
                new_neg_lp = neg_lp - log_probs[tok].item()
                new_ids    = ids + [tok]
                if tok == stop_at:
                    n_new = len(new_ids) - len(context_ids)
                    if n_new >= min_len:
                        completed.append((new_neg_lp, new_ids))
                else:
                    new_beams.append((new_neg_lp, new_ids))

        # Keep top beam_width (smallest neg_log_prob = highest prob)
        new_beams.sort(key=lambda t: t[0])
        beams = new_beams[:beam_width]
        if not beams:
            break

    # Sort completed by PPL = neg_lp / n_new_tokens
    results = []
    for neg_lp, ids in completed:
        n_new  = len(ids) - len(context_ids)
        ppl    = math.exp(neg_lp / n_new) if n_new > 0 else float('inf')
        lp     = -neg_lp
        new_ids = ids[len(context_ids):]
        try:
            text = bytes(new_ids).decode('utf-8', errors='replace')
        except Exception:
            text = str(new_ids)
        results.append((ppl, lp, text, new_ids))

    results.sort(key=lambda t: t[0])
    return results


def run():
    model, special, cfg = load_model('ode.pt')
    BLOCK = cfg.get('block_size', 1024)

    ssh_ids  = list(POEM_SSH.encode('utf-8'))
    last_ids = list(POEM_LAST_STANZA.encode('utf-8'))[-1000:]

    try:
        full_text = open(FULL_POEM_PATH).read().strip()
        full_ids  = list(full_text.encode('utf-8'))[-900:]
        print(f"Full poem loaded: {len(full_ids)} bytes (truncated to 900)")
    except FileNotFoundError:
        full_ids  = last_ids
        print("Full poem not found, using last stanza")

    print("=" * 72)
    print("  H07 — Beam search over printable ASCII")
    print(f"  beam_width=10, max_new=30, stop_at='}}'")
    print("=" * 72)

    contexts = [
        ("SSH 4-line stanza",       ssh_ids),
        ("Full poem last 900b",     full_ids),
        ("Last stanza",             last_ids),
        ("SSH + [_special]",        ssh_ids + [special['_']]),
        ("SSH + [{special]",        ssh_ids + [special['{']]),
    ]

    for label, ctx in contexts:
        print(f"\n{'─' * 72}")
        print(f"  Context: {label!r}  ({len(ctx)} tokens)")
        print(f"{'─' * 72}")

        # ── A: Beam over all printable ASCII (stop at }) ──────────────────
        print("\n  [A] Beam search — printable ASCII, stop at '}':")
        results_a = beam_search(
            model, ctx,
            max_new=35, beam_width=10,
            allowed_tokens=PRINTABLE_ASCII, stop_at=125, min_len=3
        )
        if results_a:
            for rank, (ppl, lp, text, ids) in enumerate(results_a[:10], 1):
                print(f"    {rank:2d}. PPL={ppl:.3f}  lp={lp:.2f}  {text!r}")
        else:
            print("    (no completions reached '}')")

        # ── B: Beam over flag-like chars only ─────────────────────────────
        print("\n  [B] Beam search — flag chars [a-zA-Z0-9_{}], stop at '}':")
        results_b = beam_search(
            model, ctx,
            max_new=30, beam_width=10,
            allowed_tokens=[c for c in FLAG_CHARS if c != 125],
            stop_at=125, min_len=3
        )
        if results_b:
            for rank, (ppl, lp, text, ids) in enumerate(results_b[:10], 1):
                print(f"    {rank:2d}. PPL={ppl:.3f}  lp={lp:.2f}  {text!r}")
        else:
            print("    (no completions reached '}')")

        # ── C: Top-5 next tokens (reference) ─────────────────────────────
        x = torch.tensor([ctx[-1024:]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        probs = F.log_softmax(logits[0, -1], dim=-1)
        top5  = torch.topk(probs, 5)
        id_to_s = {v: k for k, v in special.items()}
        top5_str = "  ".join(
            f"{repr(chr(t)) if 32<=t<127 else id_to_s.get(t,f'0x{t:02x}')}:{math.exp(p):.3f}"
            for t, p in zip(top5.indices.tolist(), top5.values.tolist())
        )
        print(f"\n  [C] Top-5 next tokens: {top5_str}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  SUMMARY — best completions found by beam search:")
    print(f"{'=' * 72}")
    print("  (see [A] and [B] results per context above)")
    print("  → try top-3 from each context on SSH server")
    print()
