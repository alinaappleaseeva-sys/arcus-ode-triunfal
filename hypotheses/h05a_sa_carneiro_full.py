"""
H05a: Mário de Sá-Carneiro — full name, flag format, full poem context
======================================================================

Hypothesis
----------
H04b showed ' de Sá-Carneiro' as the greedy continuation with PPL=1.47 —
the strongest signal found. But SSH rejects all ` de Sá-Carneiro` variants.

The missing piece may be one of:

  A. We tried the surname suffix but NOT the full name with first name:
       "Mário de Sá-Carneiro"

  B. The answer needs the CTF flag{} wrapper:
       "flag{mario_de_sa_carneiro}"
       "flag{sa_carneiro}"
       "flag{mario_de_sa_carneiro_1890_1916}"

  C. The poem stanza we see in SSH is only 4 lines; the model was trained on
     the FULL "Ode Triunfal" (hundreds of lines). With full poem as context,
     the most likely continuation might differ → flag lives at end of the poem.

  D. The Sá-Carneiro signal is a red herring from the training corpus
     (contamination). The flag is something the model generates under a
     different protocol we haven't tried.

This script:
  1. Scores "Mário de Sá-Carneiro" and many variants (full name + flag format)
  2. Runs greedy continuation with FULL Ode Triunfal as context
  3. Compares PPL of full vs 4-line poem context for top candidates

Status: TODO
"""

import math
import torch
import torch.nn.functional as F
from model import load_model, generate, decode

# ── 4-line poem (what SSH shows) ─────────────────────────────────────────────
POEM_SHORT = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "        Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "    Só porque houve outra e foram humanos Virgílio e Platão"
)

# ── Full Ode Triunfal (first ~40 lines) ─────────────────────────────────────
# Source: Fernando Pessoa (Álvaro de Campos), Ode Triunfal, 1914
# Orpheu I, 1915
POEM_FULL = """\
À dolorosa luz das grandes lâmpadas eléctricas da fábrica
Tenho febre e escrevo.
Escrevo rangendo os dentes, fera para a beleza disto,
Para a beleza disto totalmente desconhecida dos antigos.
Ó rodas, ó engrenagens, r-r-r-r-r-r-r eterno!
Forte espasmo retido dos maquinismos em fúria!
Em fúria fora e dentro de mim,
Por todos os meus nervos dissecados fora,
Por todas as papilas fora de tudo com que eu sinto!
Tenho os lábios secos, ó grandes ruídos modernos,
De vos ouvir demasiadamente de perto,
E arde-me a cabeça de vos querer cantar com um excesso
De expressão de todas as minhas sensações,
Com um excesso contemporâneo de vós, ó máquinas!

Em febre e olhando os motores como a uma Natureza tropical —
Grandes trópicos humanos de ferro e fogo e força —
Canto, e canto o presente, e também o passado e o futuro,
        Porque o presente é todo o passado e todo o futuro
E há Platão e Virgílio dentro das máquinas e das luzes eléctricas
    Só porque houve outra e foram humanos Virgílio e Platão
E há Platão e Virgílio dentro das máquinas e das luzes eléctricas
E os poemas de Homero e os mares de Ulisses"""

CANDIDATES = [
    # A — Full name variants (what we never tried)
    "Mário de Sá-Carneiro",
    "Mario de Sa-Carneiro",
    "mário de sá-carneiro",
    "mario de sa-carneiro",
    "Mário de Sá Carneiro",
    "Sá-Carneiro",
    "Sa-Carneiro",

    # B — flag{} format
    "flag{mario_de_sa_carneiro}",
    "flag{sa_carneiro}",
    "flag{sá_carneiro}",
    "flag{mario_sa_carneiro}",
    "flag{mario_de_sa_carneiro_1890_1916}",
    "flag{orpheu}",
    "flag{alvaro_de_campos}",
    "flag{álvaro_de_campos}",
    "flag{ode_triunfal}",
    "flag{luso_lit}",
    "flag{luso_lit_lm}",

    # C — Orpheu / publication context
    "Orpheu",
    "Orpheu I",
    "Orpheu 1",
    "orpheu",
    "1914",
    "1915",

    # D — Fernando Pessoa as author of heteronym
    "Fernando Pessoa",
    "Fernando Pessoa (Álvaro de Campos)",
    "Fernando António Nogueira Pessoa",
]


def score(model, poem_ids, text):
    text_ids = list(text.encode('utf-8'))
    context = poem_ids + text_ids
    x = torch.tensor([context], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    lp = 0.0
    for i, target in enumerate(text_ids):
        pos = len(poem_ids) + i - 1
        lp += F.log_softmax(logits[0, pos], dim=-1)[target].item()
    ppl = math.exp(-lp / max(len(text_ids), 1))
    return lp, ppl


def run():
    model, special, cfg = load_model('ode.pt')
    poem_short_ids = list(POEM_SHORT.encode('utf-8'))
    poem_full_ids  = list(POEM_FULL.encode('utf-8'))

    print("=" * 70)
    print("  H05a — Mário de Sá-Carneiro variants + full poem context")
    print("=" * 70)

    # ── Score all candidates with SHORT poem (4 lines) ───────────────────────
    print("\n--- Short poem context (4 lines) ---")
    scores_short = []
    for cand in CANDIDATES:
        try:
            lp, ppl = score(model, poem_short_ids, cand)
            scores_short.append((cand, lp, ppl))
        except Exception as e:
            print(f"  ERROR for {cand!r}: {e}")

    scores_short.sort(key=lambda t: t[1], reverse=True)
    print(f"\n  {'LogProb':>10} {'PPL':>8}  Candidate")
    print("  " + "-" * 62)
    for rank, (cand, lp, ppl) in enumerate(scores_short, 1):
        marker = "  ← TRY SSH" if rank <= 8 else ""
        print(f"  {lp:>10.2f} {ppl:>8.2f}  {cand!r}{marker}")

    # ── Score same candidates with FULL poem ─────────────────────────────────
    print("\n--- Full poem context ---")
    scores_full = []
    for cand in CANDIDATES:
        try:
            lp, ppl = score(model, poem_full_ids, cand)
            scores_full.append((cand, lp, ppl))
        except Exception as e:
            print(f"  ERROR for {cand!r}: {e}")

    scores_full.sort(key=lambda t: t[1], reverse=True)
    print(f"\n  {'LogProb':>10} {'PPL':>8}  Candidate")
    print("  " + "-" * 62)
    for rank, (cand, lp, ppl) in enumerate(scores_full, 1):
        marker = "  ← TRY SSH" if rank <= 8 else ""
        print(f"  {lp:>10.2f} {ppl:>8.2f}  {cand!r}{marker}")

    # ── Greedy from full poem (compare to 4-line greedy) ─────────────────────
    print("\n" + "=" * 70)
    print("  Greedy continuation of FULL poem (first 60 bytes)")
    print("=" * 70)
    torch.manual_seed(42)
    gen = generate(model, poem_full_ids, max_new=60, greedy=True, stop_at_close=False)
    print(f"\n  {decode(gen)!r}")

    # ── Reference: greedy from short poem ────────────────────────────────────
    print("\n--- Reference: greedy from 4-line poem ---")
    torch.manual_seed(42)
    gen2 = generate(model, poem_short_ids, max_new=60, greedy=True, stop_at_close=False)
    print(f"\n  {decode(gen2)!r}")

    # ── Score 'Mário de Sá-Carneiro' per-token probabilities ─────────────────
    print("\n" + "=" * 70)
    print("  Per-token probs: 'Mário de Sá-Carneiro' after short poem")
    print("=" * 70)
    target_str = "Mário de Sá-Carneiro"
    target_ids = list(target_str.encode('utf-8'))
    context = poem_short_ids + target_ids
    x = torch.tensor([context], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    print(f"\n  context_len={len(poem_short_ids)}")
    for i, tid in enumerate(target_ids):
        pos = len(poem_short_ids) + i - 1
        p = F.softmax(logits[0, pos], dim=-1)[tid].item()
        ch = chr(tid) if 32 <= tid < 127 else f'0x{tid:02x}'
        print(f"    pos {i:2d}: id={tid:3d} ({ch!r})  P={p:.4f}")

    # ── ACTION ITEMS ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TOP SSH CANDIDATES (from short poem, highest log-prob):")
    print("=" * 70)
    for rank, (cand, lp, ppl) in enumerate(scores_short[:8], 1):
        print(f"  {rank}. {cand!r}  (ppl={ppl:.2f})")
    print()
