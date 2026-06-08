"""
H14: Generation grid — context × prefix × temperature
=======================================================

HYPOTHESIS
----------
There exists a combination of (poem source / truncation / prefix) that
causes the model to stably generate a phrase of the same format as the
false-positive candidate but with different content — and we have not
tried that combination yet.  The flag lives in generation, not in weights.

CONTRACT / Definition of Done
------------------------------
CONFIRMED    if: a generated phrase survives normalise_flag() AND passes
                 flag_verify.py 4-step single-session protocol
INCONCLUSIVE if: a new stable phrase appears but has not been SSH-verified
REJECTED     if: all 21 cells produce either (a) the false-positive phrase
                 again, (b) degenerate/looping output, or (c) no
                 phrase-shaped string in 120 tokens

STOP CRITERION  ← read before running
--------------------------------------
  • 7 contexts × 3 prefixes = 21 cells; greedy + t=0.8 per cell = 42 rows
  • If no cell produces a NEW phrase: REJECTED → write 1 para in WRITEUP
  • Do NOT add more contexts to escape a null result
  • Do NOT open WRITEUP until this script finishes

FLAG VERIFICATION PROTOCOL (mandatory before any SSH attempt)
--------------------------------------------------------------
  python tools/flag_verify.py --candidate "flag{...}"
  Steps: flag{aaa} open → flag{test} open → CANDIDATE → flag{zzz} open
  Declare CONFIRMED only when steps 1–2 open, step 3 closed, step 4 open.

WHAT WE DO NOT DO TODAY
-----------------------
  ✗ open WRITEUP before a result
  ✗ create a new Hxx without a result from this grid
  ✗ expand the grid beyond 7 contexts (scope creep)
  ✗ treat connection-close without post-control as verification
  ✗ re-interpret Bernardo's hint mid-experiment
"""

import math
import re
import torch
import torch.nn.functional as F
from model import load_model

# ── Poem sources ──────────────────────────────────────────────────────────────

POEM_WIKI_PATH = "/tmp/ode_triunfal_full.txt"

SSH_STANZA = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "        Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "    Só porque houve outra e foram humanos Virgílio e Platão"
)

LAST_STANZA = """\
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise_flag(text: str) -> str:
    import unicodedata
    nfd = unicodedata.normalize("NFD", text.strip())
    ascii_only = "".join(
        c for c in nfd
        if unicodedata.category(c) != "Mn" and ord(c) < 128
    ).lower()
    parts = []
    for ch in ascii_only:
        if ch.isalnum():
            parts.append(ch)
        elif ch in " _-":
            parts.append("_")
    content = "_".join(p for p in "".join(parts).split("_") if p)
    return f"flag{{{content}}}"


def extract_phrase(text: str) -> str | None:
    """
    Pull the first phrase-shaped string from generated text.
    A phrase is: starts after leading whitespace/newlines, ends at
    sentence-final punctuation (. ! ?) or a double newline.
    Returns None if nothing found.
    """
    # Strip leading whitespace / newlines
    stripped = text.lstrip("\n\r ")
    if not stripped:
        return None
    # Match up to first sentence-ending punctuation or double newline
    m = re.match(r"([^\n!?.]{4,}[!?.])", stripped)
    if m:
        return m.group(1).strip()
    # Fallback: first non-empty line of >= 4 chars
    for line in stripped.split("\n"):
        line = line.strip()
        if len(line) >= 4:
            return line
    return None


def generate_greedy(model, ids, max_new, block_size):
    generated = []
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-block_size:]], dtype=torch.long)
            logits = model(x)
            nxt = torch.argmax(logits[0, -1]).item()
            generated.append(nxt)
            ids = ids + [nxt]
    return generated


def generate_sampled(model, ids, max_new, block_size, temperature=0.8, top_k=40):
    generated = []
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-block_size:]], dtype=torch.long)
            logits = model(x)
            logits_last = logits[0, -1] / temperature
            if top_k:
                vals, _ = torch.topk(logits_last, top_k)
                logits_last[logits_last < vals[-1]] = -float("inf")
            probs = F.softmax(logits_last, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            generated.append(nxt)
            ids = ids + [nxt]
    return generated


def decode_bytes(token_ids):
    return bytes([t for t in token_ids if 0 <= t < 256]).decode("utf-8", errors="replace")


# ── False-positive reference ──────────────────────────────────────────────────
FALSE_POS_PHRASE = "Ah! não ser eu toda a gente que me acontece!"
FALSE_POS_FLAG   = "flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}"


def is_false_positive(phrase: str) -> bool:
    if phrase is None:
        return False
    return normalise_flag(phrase) == FALSE_POS_FLAG


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    model, special, cfg = load_model("ode.pt")
    BLOCK = cfg.get("block_size", 1024)

    pid_fp    = special["<|fernando_pessoa|>"]   # 256
    # Álvaro de Campos is absent from the model vocabulary.
    # We simulate it as a text-byte prefix instead.
    ALVARO_TEXT_PREFIX = "Álvaro de Campos —\n"

    # ── Load poem variants ────────────────────────────────────────────────────
    try:
        poem_wiki_full = open(POEM_WIKI_PATH).read().strip()
    except FileNotFoundError:
        print(f"ERROR: {POEM_WIKI_PATH} not found.")
        print("  Run: curl -s 'https://pt.wikisource.org/w/index.php"
              "?title=Ode_Triunfal&action=raw' > /tmp/ode_triunfal_full.txt")
        return

    # Context C5: poem without final stanza (cut before last "Eia comboios" block)
    cutpoint = poem_wiki_full.rfind("Eia comboios")
    poem_wiki_no_final = poem_wiki_full[:cutpoint].rstrip() if cutpoint != -1 else poem_wiki_full

    # ── Context definitions ───────────────────────────────────────────────────
    contexts = [
        # (label,               bytes_to_use)
        ("C1_ssh_stanza",       list(SSH_STANZA.encode("utf-8"))),
        ("C2_wiki_900",         list(poem_wiki_full.encode("utf-8"))[-900:]),
        ("C3_wiki_512",         list(poem_wiki_full.encode("utf-8"))[-512:]),
        ("C4_wiki_256",         list(poem_wiki_full.encode("utf-8"))[-256:]),
        ("C5_wiki_no_final_900",list(poem_wiki_no_final.encode("utf-8"))[-900:]),
        ("C6_last_stanza",      list(LAST_STANZA.encode("utf-8"))),
    ]

    # ── Prefix definitions ────────────────────────────────────────────────────
    alvaro_bytes = list(ALVARO_TEXT_PREFIX.encode("utf-8"))
    prefixes = [
        # (label,            token_prefix,   byte_prepend)
        ("P1_none",          [],              []),
        ("P2_fp",            [pid_fp],        []),
        ("P3_alvaro_text",   [],              alvaro_bytes),
    ]

    # ── Generation modes ─────────────────────────────────────────────────────
    MAX_NEW   = 120
    TEMP      = 0.8
    TOP_K     = 40
    SEED      = 42

    print("=" * 80)
    print("  H14 — Generation grid")
    print(f"  {len(contexts)} contexts × {len(prefixes)} prefixes × 2 modes = "
          f"{len(contexts)*len(prefixes)*2} rows")
    print(f"  false-positive reference: {FALSE_POS_FLAG}")
    print("=" * 80)

    results   = []   # (ctx_label, pfx_label, mode, phrase, flag_candidate, is_fp, is_new)
    new_found = []   # only non-false-positive flag-shaped phrases

    for ctx_label, ctx_bytes in contexts:
        for pfx_label, tok_pfx, byte_pfx in prefixes:

            base_ids = tok_pfx + byte_pfx + ctx_bytes
            # Ensure fits in block
            if len(base_ids) > BLOCK - MAX_NEW:
                base_ids = base_ids[-(BLOCK - MAX_NEW):]

            for mode in ("greedy", f"t{TEMP}_k{TOP_K}"):
                torch.manual_seed(SEED)

                if mode == "greedy":
                    gen = generate_greedy(model, base_ids[:], MAX_NEW, BLOCK)
                else:
                    gen = generate_sampled(model, base_ids[:], MAX_NEW, BLOCK,
                                           temperature=TEMP, top_k=TOP_K)

                gen_text  = decode_bytes(gen)
                phrase    = extract_phrase(gen_text)
                flag_cand = normalise_flag(phrase) if phrase else None
                is_fp     = is_false_positive(phrase)
                is_new    = (flag_cand is not None and not is_fp
                             and len(flag_cand) > len("flag{}") + 4)

                row = (ctx_label, pfx_label, mode, phrase, flag_cand, is_fp, is_new)
                results.append(row)

                if is_new:
                    new_found.append(row)

                marker = "*** NEW ***" if is_new else ("(fp)" if is_fp else "")
                print(f"\n  [{ctx_label}] [{pfx_label}] [{mode}] {marker}")
                print(f"    raw:    {gen_text[:120]!r}")
                print(f"    phrase: {phrase!r}")
                print(f"    flag:   {flag_cand}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"  {'Context':<28} {'Prefix':<18} {'Mode':<14} {'Flag candidate':<50} {'Status'}")
    print("  " + "-" * 120)
    for ctx_label, pfx_label, mode, phrase, flag_cand, is_fp, is_new in results:
        status = "*** NEW ***" if is_new else ("false-pos" if is_fp else "—")
        cand   = (flag_cand or "—")[:48]
        print(f"  {ctx_label:<28} {pfx_label:<18} {mode:<14} {cand:<50} {status}")

    # ── New candidates ────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    if new_found:
        print(f"  NEW CANDIDATES ({len(new_found)}) — run flag_verify.py before SSH:")
        print(f"{'=' * 80}")
        for ctx_label, pfx_label, mode, phrase, flag_cand, _, _ in new_found:
            print(f"\n  Context:  {ctx_label}")
            print(f"  Prefix:   {pfx_label}")
            print(f"  Mode:     {mode}")
            print(f"  Phrase:   {phrase!r}")
            print(f"  Flag:     {flag_cand}")
            print(f"\n  Verify:   python tools/flag_verify.py --candidate \"{flag_cand}\"")
    else:
        print("  NO NEW CANDIDATES — all cells returned false-positive or degenerate output.")
        print("  RESULT: REJECTED")
        print("  Next:   add 1-paragraph note to WRITEUP generation section; close search day.")
    print()


if __name__ == "__main__":
    run()
