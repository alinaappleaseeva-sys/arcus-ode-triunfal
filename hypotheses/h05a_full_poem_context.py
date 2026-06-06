"""
H05a: Full Ode Triunfal poem as model context
==============================================

Hypothesis
----------
All H04 variants used only the 4-line stanza visible in the SSH TUI. The full
"Ode Triunfal" is ~270 lines / ~11 KB. With the full poem as context:

  1. The greedy continuation may differ significantly (full poem ends with
     "Ah não ser eu toda a gente e toda a parte!" — not the 4-line stanza)
  2. The model may have seen this poem in training with an attribution afterwards
  3. PPL of candidates may change dramatically with full context

Important observations:
  - Block size = 1024 tokens; full poem = 11 KB → truncate to last ~900 bytes
  - SSH stanza uses "outra" but Wikisource uses "outrora" (version difference!)
  - The 4-line poem ends mid-poem; the FULL poem ends differently

Experiments:
  A. Greedy from full poem (last 900 bytes) → what comes after "toda a parte!"?
  B. Score candidates with full poem context
  C. Top-20 next tokens after full poem
  D. Heteronym prefix + full poem → greedy
  E. Score candidates with last-2-stanzas context (practical for block_size)

Status: TODO
"""

import math
import torch
import torch.nn.functional as F
from model import load_model, generate, decode

# ── Full poem (from Wikisource, public domain) ────────────────────────────────
POEM_FULL = open("/tmp/ode_triunfal_full.txt").read().strip()

# ── 4-line SSH stanza (what the TUI shows) ────────────────────────────────────
POEM_SHORT = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "        Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "    Só porque houve outra e foram humanos Virgílio e Platão"
)

# ── Last stanza only (ending of the poem) ─────────────────────────────────────
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

# ── Candidates ────────────────────────────────────────────────────────────────
CANDIDATES = [
    # Best from H04 / H04b with short context
    " de Sá-Carneiro",
    "Mário de Sá-Carneiro",
    "Fernando Pessoa",
    "\nFernando Pessoa",
    "\n\n— Fernando Pessoa",
    "Álvaro de Campos",
    "\nÁlvaro de Campos",
    "\n\n— Álvaro de Campos",

    # Publication attribution
    "\n\nÁlvaro de Campos\nOrpheu I, 1915",
    "\n\nFernando Pessoa\nÁlvaro de Campos",
    "\n\n(Álvaro de Campos)",

    # Continuation of the poem itself
    "\n",
    "\n\n",
    "!",
    "\nAh",
    " Ah",

    # flag{} formats
    "flag{alvaro_de_campos}",
    "flag{mario_de_sa_carneiro}",
    "flag{ode_triunfal}",
    "flag{toda_a_parte}",
    "flag{nao_ser_eu}",
]


def score(model, context_ids, text):
    text_ids = list(text.encode('utf-8'))
    # Truncate context to leave room for text
    max_ctx = 1023 - len(text_ids)
    ctx = context_ids[-max_ctx:] if len(context_ids) > max_ctx else context_ids
    full = ctx + text_ids
    x = torch.tensor([full], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    lp = 0.0
    offset = len(ctx) - 1
    for i, target in enumerate(text_ids):
        lp += F.log_softmax(logits[0, offset + i], dim=-1)[target].item()
    ppl = math.exp(-lp / max(len(text_ids), 1))
    return lp, ppl


def run():
    model, special, cfg = load_model('ode.pt')
    id_to_s = {v: k for k, v in special.items()}

    short_ids     = list(POEM_SHORT.encode('utf-8'))
    full_ids      = list(POEM_FULL.encode('utf-8'))
    last_ids      = list(POEM_LAST_STANZA.encode('utf-8'))

    print("=" * 72)
    print("  H05a — Full Ode Triunfal as context")
    print(f"  Full poem: {len(full_ids)} bytes | Last stanza: {len(last_ids)} bytes")
    print(f"  Short (SSH): {len(short_ids)} bytes | Block size: {cfg.get('block_size',1024)}")
    print("=" * 72)

    # ── A: Greedy from end of full poem ──────────────────────────────────────
    print("\n--- A: Greedy continuation (first 80 bytes) ---")
    for label, ctx in [("full poem (last 900b)", full_ids[-900:]),
                       ("last stanza",           last_ids),
                       ("SSH 4-line",            short_ids)]:
        torch.manual_seed(42)
        gen = generate(model, ctx, max_new=80, greedy=True, stop_at_close=False)
        print(f"  [{label}] → {decode(gen)!r}")

    # ── B: Candidate scoring across 3 contexts ────────────────────────────────
    print("\n--- B: Candidate scoring (3 contexts) ---")
    hdr = f"  {'Candidate':40s} {'SSH PPL':>8}  {'LastS PPL':>9}  {'Full PPL':>8}  Best"
    print(hdr)
    print("  " + "-" * 80)

    rows = []
    for cand in CANDIDATES:
        lp_s, ppl_s = score(model, short_ids, cand)
        lp_l, ppl_l = score(model, last_ids,  cand)
        lp_f, ppl_f = score(model, full_ids,  cand)
        best = min(ppl_s, ppl_l, ppl_f)
        rows.append((cand, ppl_s, ppl_l, ppl_f, best))

    rows.sort(key=lambda r: r[4])  # sort by best PPL
    for cand, ps, pl, pf, best in rows:
        new_flag = " *** NEW BEST ***" if best < 2.0 and cand not in (" de Sá-Carneiro",) else ""
        print(f"  {cand!r:40s} {ps:>8.2f}  {pl:>9.2f}  {pf:>8.2f}  {best:.2f}{new_flag}")

    # ── C: Top-20 next tokens after each context ──────────────────────────────
    print("\n--- C: Top-20 next tokens ---")
    for label, ctx in [("full poem (last 900b)", full_ids[-900:]),
                       ("last stanza",           last_ids),
                       ("SSH 4-line",            short_ids)]:
        x = torch.tensor([ctx], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        probs = F.softmax(logits[0, -1], dim=-1)
        top = torch.topk(probs, 20)
        print(f"\n  [{label}]")
        for tid, p in zip(top.indices.tolist(), top.values.tolist()):
            name = repr(chr(tid)) if 32 <= tid < 127 else (id_to_s.get(tid) or f'0x{tid:02x}')
            print(f"    id={tid:3d}  P={p:.4f}  {name}")

    # ── D: Heteronym prefix + full poem ───────────────────────────────────────
    print("\n--- D: Heteronym prefix + last stanza → greedy ---")
    for het in ['<|fernando_pessoa|>', '<|alberto_caeiro|>',
                '<|ricardo_reis|>', '<|bernardo_soares|>']:
        pid = special[het]
        seed = [pid] + last_ids
        torch.manual_seed(42)
        gen = generate(model, seed, max_new=60, greedy=True, stop_at_close=False)
        print(f"  [{het}] → {decode(gen)!r}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  TOP-8 SSH CANDIDATES (best PPL across all contexts):")
    print(f"{'=' * 72}")
    for rank, (cand, ps, pl, pf, best) in enumerate(rows[:8], 1):
        clean = cand.lstrip('\n— ').strip()
        print(f"  {rank}. {cand!r}  best_ppl={best:.2f}  (ssh={ps:.2f}, last={pl:.2f}, full={pf:.2f})")
        print(f"     → try: {clean!r}")
    print()
