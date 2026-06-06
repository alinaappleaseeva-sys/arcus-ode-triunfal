"""
H04c: Fernando Pessoa attribution — exhaustive format variants
==============================================================

Hypothesis
----------
H04 found Fernando Pessoa with PPL=4.0 — the strongest signal across all
28 candidates. H04b tested attribution suffixes but focused on the FORMAT
(em-dash, newlines, etc.) without exhausting the NAME itself in all forms.

H04b tested these Pessoa formats:
  '\n\n— Fernando Pessoa'   PPL=2.37  (was rank ~4 in H04b)
  '\n— Fernando Pessoa'     PPL=?
  '\nFernando Pessoa'       PPL=2.82
  '\n\nFernando Pessoa'     PPL=?
  ' — Fernando Pessoa'      PPL=2.37

What H04b did NOT test (and H04c adds):
  — abbreviated: F. Pessoa, Fernando P., F.P.
  — bare surname: Pessoa
  — with trailing whitespace/newline: '— Fernando Pessoa\n'
  — with title: '— Fernando Pessoa, Ode Triunfal'
  — with dates: '— Fernando Pessoa (1888-1935)'
  — with heteronym clarification: '— Fernando Pessoa / Álvaro de Campos'
  — with 'por': 'por Fernando Pessoa'
  — with 'de': 'de Fernando Pessoa'
  — Portuguese sig styles: 'in Fernando Pessoa, Orpheu I (1915)'
  — just the flag{} wrapper: 'flag{fernando_pessoa}'

The aim: find if any Pessoa format has PPL < 2.37 (the H04b best).
Top formats → SSH.

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

# All formats to test — pure Pessoa variants only
SUFFIXES = [
    # ── Bare surname only ─────────────────────────────────────────────────────
    "\n\nPessoa",
    "\nPessoa",
    " Pessoa",
    "\n\n— Pessoa",
    "\n— Pessoa",

    # ── Abbreviated first name ────────────────────────────────────────────────
    "\n\nF. Pessoa",
    "\nF. Pessoa",
    " F. Pessoa",
    "\n\n— F. Pessoa",
    "\n— F. Pessoa",
    "\n\nFernando P.",
    "\n\nF.P.",

    # ── Full name, trailing newline / punctuation ─────────────────────────────
    "\n\n— Fernando Pessoa\n",
    "\n\n— Fernando Pessoa\n\n",
    "\n\n— Fernando Pessoa.",
    "\n\n— Fernando Pessoa,",

    # ── With title / publication ──────────────────────────────────────────────
    "\n\n— Fernando Pessoa, Ode Triunfal",
    "\n\n— Fernando Pessoa, «Ode Triunfal»",
    "\n\n(Fernando Pessoa, Ode Triunfal, 1914)",
    "\n\nin Fernando Pessoa, Orpheu I (1915)",
    "\n\n— Fernando Pessoa, in: Orpheu",
    "\n\n— Fernando Pessoa, Orpheu, 1915",

    # ── With dates ────────────────────────────────────────────────────────────
    "\n\n— Fernando Pessoa (1888-1935)",
    "\n\n— Fernando Pessoa (1888–1935)",   # en-dash
    "\n— Fernando Pessoa (1888-1935)",

    # ── With heteronym attribution ────────────────────────────────────────────
    "\n\n— Fernando Pessoa / Álvaro de Campos",
    "\n\n— Fernando Pessoa (het. Álvaro de Campos)",
    "\n\n— Fernando Pessoa, sob pseudónimo Álvaro de Campos",
    "\n\n[Fernando Pessoa — Álvaro de Campos]",

    # ── Portuguese preposition styles ─────────────────────────────────────────
    "\n\npor Fernando Pessoa",
    "\n\nde Fernando Pessoa",
    "\n\nPor Fernando Pessoa",
    "\n\nDe Fernando Pessoa",

    # ── flag{} format ─────────────────────────────────────────────────────────
    "\n\nflag{fernando_pessoa}",
    "flag{fernando_pessoa}",
    "\nflag{fernando_pessoa}",
    " flag{fernando_pessoa}",

    # ── Reference: best from H04b (PPL=2.37) to compare baseline ─────────────
    "\n\n— Fernando Pessoa",   # H04b rank ~4, PPL=2.37
    "\nFernando Pessoa",       # H04b PPL=2.82
]


def score_suffix(model, poem_ids, suffix):
    suffix_ids = list(suffix.encode('utf-8'))
    context = poem_ids + suffix_ids
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
    print("  H04c — Fernando Pessoa format variants (exhaustive)")
    print("  Reference baseline from H04b: '\\n\\n— Fernando Pessoa' PPL=2.37")
    print("=" * 70)

    scores = []
    for sfx in SUFFIXES:
        lp, ppl = score_suffix(model, poem_ids, sfx)
        scores.append((sfx, lp, ppl))

    scores.sort(key=lambda t: t[1], reverse=True)

    print(f"\n  {'LogProb':>10} {'PPL':>8}  Suffix")
    print("  " + "-" * 62)
    for rank, (sfx, lp, ppl) in enumerate(scores, 1):
        marker = "  ← TRY SSH" if rank <= 8 else ""
        beat_baseline = "  *** BEATS H04b BASELINE ***" if ppl < 2.37 else ""
        sfx_display = repr(sfx)[:55]
        print(f"  {lp:>10.2f} {ppl:>8.2f}  {sfx_display}{marker}{beat_baseline}")

    # ── Per-token breakdown of the best candidate ────────────────────────────
    best_sfx, best_lp, best_ppl = scores[0]
    print(f"\n{'=' * 70}")
    print(f"  Per-token probs for best suffix: {best_sfx!r}")
    print(f"{'=' * 70}")
    sfx_ids = list(best_sfx.encode('utf-8'))
    context = poem_ids + sfx_ids
    x = torch.tensor([context], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    for i, tid in enumerate(sfx_ids):
        pos = len(poem_ids) + i - 1
        p = F.softmax(logits[0, pos], dim=-1)[tid].item()
        ch = repr(chr(tid)) if 32 <= tid < 127 else f'0x{tid:02x}'
        print(f"    pos {i:2d}: id={tid:3d} ({ch})  P={p:.4f}")

    # ── ACTION ITEMS ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  TOP-8 SSH CANDIDATES (highest log-prob → most likely to be flag):")
    print(f"{'=' * 70}")
    for rank, (sfx, lp, ppl) in enumerate(scores[:8], 1):
        clean = sfx.lstrip('\n— ').strip()
        print(f"  {rank}. raw: {sfx!r}")
        print(f"     ssh: {clean!r}  (ppl={ppl:.2f})")
    print()
