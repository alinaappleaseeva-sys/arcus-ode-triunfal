"""
H03+: Systematic byte-sequence analysis with structured output
==============================================================
Extends H03 with three improvements from post-PR review:

1. Machine-readable output: writes results/h03_plus_<ts>.json and .csv
   alongside human-readable stdout. Schema per row:
     label, seed_tokens, full_ids, full_named,
     slice_name, slice_hex, slice_ints,
     best_encoding, best_text, notes

2. Multiple byte-extraction strategies per context:
     "up_to_first_close"   — original H03 strategy: bytes before first }
     "between_open_close"  — bytes between first { (byte 123) and last }
     "first_N_raw"         — first 32 raw byte-tokens, no trimming

3. Special-token context window: for each generation, prints which
   special tokens appear within ±5 positions of the first { occurrence.
   If the flag uses a pattern like special / bytes / special, this shows it.

Status: TODO
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from model import load_model, generate

# ── Constants ────────────────────────────────────────────────────────────────

POEM = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "        Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "    Só porque houve outra e foram humanos Virgílio e Platão"
)

RESULTS_DIR = Path('results')

# ── Helpers ───────────────────────────────────────────────────────────────────

def tok_name(i: int, id_to_special: dict) -> str:
    if i < 256:
        return repr(chr(i)) if 32 <= i < 127 else f'0x{i:02x}'
    return id_to_special.get(i, f'special_{i}')


def extract_slices(ids: list[int]) -> dict[str, list[int]]:
    """
    Return multiple candidate byte-only slices from a token sequence.

    Strategies
    ----------
    up_to_first_close
        Classic H03 strategy. Byte tokens before the first } (id=125).
    between_open_close
        Byte tokens strictly between the first occurrence of { (id=123,
        the raw byte — NOT the special token 261) and the last }.
        Useful when the flag appears as literal {...} text in the output.
    first_N_raw
        First 32 byte tokens (id < 256), no trimming. Preserves context
        even when no } is present.
    """
    byte_ids = [i for i in ids if i < 256]

    # up_to_first_close
    if ord('}') in byte_ids:
        end = byte_ids.index(ord('}'))
        up_to_close = byte_ids[:end]
    else:
        up_to_close = byte_ids

    # between_open_close (raw byte { = 123, NOT special token 261)
    if ord('{') in byte_ids and ord('}') in byte_ids:
        open_pos  = byte_ids.index(ord('{'))
        close_pos = len(byte_ids) - 1 - byte_ids[::-1].index(ord('}'))
        between   = byte_ids[open_pos + 1: close_pos]
    else:
        between = []

    # first_N_raw
    first_n = byte_ids[:32]

    return {
        'up_to_first_close':   up_to_close,
        'between_open_close':  between,
        'first_N_raw':         first_n,
    }


def try_decodings(raw: list[int]) -> tuple[str, str]:
    """
    Try all encodings on raw byte list.
    Returns (best_encoding_name, best_text) — where 'best' means
    fully printable ASCII. Falls back to hex if nothing is clean.
    """
    import base64

    b = bytes(raw)
    candidates = []

    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            text = b.decode(enc)
            printable = all(32 <= ord(c) < 127 or c in '\n\r\t' for c in text)
            candidates.append((enc, text, printable))
        except Exception:
            pass

    # XOR sweep
    for key in [0x42, 0xFF, 0xAA, 0x55, 0x20]:
        xored = bytes(bv ^ key for bv in b)
        try:
            text = xored.decode('latin-1')
            printable = all(32 <= ord(c) < 127 for c in text)
            if printable:
                candidates.append((f'XOR_0x{key:02X}', text, True))
        except Exception:
            pass

    # ASCII shift
    for shift in [32, 64, -32, -64]:
        shifted = bytes(max(0, min(255, bv + shift)) for bv in b)
        text = shifted.decode('ascii', errors='ignore')
        if len(text) == len(b) and text.isprintable():
            candidates.append((f'shift{shift:+d}', text, True))

    # Base64 representation (always valid)
    candidates.append(('base64', base64.b64encode(b).decode(), True))
    candidates.append(('hex',    b.hex(),                       True))

    # Prefer first fully-printable ASCII result
    for enc, text, printable in candidates:
        if printable and enc not in ('base64', 'hex'):
            return enc, text
    return 'hex', b.hex()


def special_token_window(ids: list[int], id_to_special: dict,
                         window: int = 5) -> list[dict]:
    """
    Find positions of special tokens (id >= 256) in ids.
    For each, return a dict with position and ±window neighbours.
    """
    result = []
    for pos, tok in enumerate(ids):
        if tok >= 256:
            lo  = max(0, pos - window)
            hi  = min(len(ids), pos + window + 1)
            ctx = [tok_name(t, id_to_special) for t in ids[lo:hi]]
            result.append({
                'position':  pos,
                'token_id':  tok,
                'token_name': id_to_special.get(tok, f'special_{tok}'),
                'context_window': ctx,
                'window_start':   lo,
            })
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, special, cfg = load_model('ode.pt')
    id_to_special = {v: k for k, v in special.items()}

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = RESULTS_DIR / f'h03_plus_{ts}.json'
    csv_path  = RESULTS_DIR / f'h03_plus_{ts}.csv'

    CSV_FIELDS = [
        'label', 'seed_tokens', 'full_ids', 'full_named',
        'slice_name', 'slice_hex', 'slice_ints',
        'best_encoding', 'best_text', 'notes',
    ]

    all_rows   = []   # for JSON
    csv_rows   = []   # for CSV

    # ── Define contexts ───────────────────────────────────────────────────────
    poem_bytes = list(POEM.encode('utf-8'))
    contexts = {
        'bare_{':           [special['_'], special['{']],
        'bare_{_only':      [special['_']],
        'bare_{brace_only': [special['{']],
        'pessoa+_{':        [special['<|fernando_pessoa|>'], special['_'], special['{']],
        'caeiro+_{':        [special['<|alberto_caeiro|>'],  special['_'], special['{']],
        'reis+_{':          [special['<|ricardo_reis|>'],    special['_'], special['{']],
        'soares+_{':        [special['<|bernardo_soares|>'], special['_'], special['{']],
        'poem_only':        poem_bytes,
        'pessoa+poem':      [special['<|fernando_pessoa|>']] + poem_bytes,
        'pessoa+poem+_{':   [special['<|fernando_pessoa|>']] + poem_bytes
                            + [special['_'], special['{']],
        'caeiro+poem+_{':   [special['<|alberto_caeiro|>']]  + poem_bytes
                            + [special['_'], special['{']],
        'reis+poem+_{':     [special['<|ricardo_reis|>']]    + poem_bytes
                            + [special['_'], special['{']],
        'soares+poem+_{':   [special['<|bernardo_soares|>']] + poem_bytes
                            + [special['_'], special['{']],
    }

    print(f"{'─'*70}")
    print(f"  H03+  |  {len(contexts)} contexts  |  3 slices each  |  seed=42")
    print(f"{'─'*70}")

    for label, seed in contexts.items():
        ids = generate(model, seed, max_new=120, greedy=True,
                       stop_at_close=False)   # don't stop — we want full seq
        named = [tok_name(i, id_to_special) for i in ids]

        # ── Special-token window ──────────────────────────────────────────────
        windows = special_token_window(ids, id_to_special, window=5)

        print(f"\n[{label}]  gen_len={len(ids)}")
        print(f"  full_ids:  {ids[:30]}{'...' if len(ids)>30 else ''}")
        print(f"  full_named:{named[:15]}{'...' if len(named)>15 else ''}")

        if windows:
            print(f"  special tokens in output ({len(windows)} found):")
            for w in windows:
                print(f"    pos={w['position']:3d}  {w['token_name']:<35}"
                      f"  ctx={w['context_window']}")
        else:
            print(f"  special tokens in output: none")

        # ── Slices + decodings ────────────────────────────────────────────────
        slices = extract_slices(ids)
        for slice_name, slice_ids in slices.items():
            if not slice_ids:
                print(f"  slice [{slice_name}]: empty — skipping")
                continue

            best_enc, best_text = try_decodings(slice_ids)
            notes = ''
            if best_enc not in ('hex', 'base64') and len(best_text) >= 4:
                notes = '✅ PRINTABLE — check this'
            elif 'flag' in best_text.lower() or '{' in best_text:
                notes = '🚨 CONTAINS FLAG PATTERN'

            print(f"  slice [{slice_name}]  hex={bytes(slice_ids).hex()!r:30}  "
                  f"{best_enc}: {best_text[:40]!r}  {notes}")

            # Accumulate for export
            row = {
                'label':        label,
                'seed_tokens':  str(seed[:6]) + ('...' if len(seed) > 6 else ''),
                'full_ids':     str(ids),
                'full_named':   str(named),
                'slice_name':   slice_name,
                'slice_hex':    bytes(slice_ids).hex(),
                'slice_ints':   str(slice_ids),
                'best_encoding': best_enc,
                'best_text':    best_text,
                'notes':        notes,
            }
            all_rows.append(row)
            csv_rows.append([row[f] for f in CSV_FIELDS])

    # ── Write structured output ───────────────────────────────────────────────
    json_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        writer.writerows(csv_rows)

    print(f"\n{'─'*70}")
    print(f"  Structured output written:")
    print(f"    {json_path}")
    print(f"    {csv_path}")

    # ── Quick summary: any printable hits? ────────────────────────────────────
    hits = [r for r in all_rows if '✅' in r['notes'] or '🚨' in r['notes']]
    if hits:
        print(f"\n  ⭐  PRINTABLE / FLAG HITS ({len(hits)}):")
        for r in hits:
            print(f"    [{r['label']}] [{r['slice_name']}]  "
                  f"{r['best_encoding']}: {r['best_text'][:60]!r}  {r['notes']}")
    else:
        print(f"\n  No clean printable strings found in any slice.")
        print(f"  → Proceed to H04 (heteronym name) or H05 (broader search).")

    print(f"{'─'*70}\n")
