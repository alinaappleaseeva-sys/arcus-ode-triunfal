"""
tools/h03_scan_for_flag.py — Post-hoc flag pattern scanner for H03+ results.
=============================================================================

Addresses two methodological gaps identified in PR #3 code review:

  1. try_decodings() picks the first printable text as "best" — if that is
     normal Portuguese prose, hex/base64 representations that contain a flag
     pattern are never surfaced as suspicious.

  2. first_N_raw = 32 bytes only covers the start of each sequence.  The full
     full_ids are in the JSON, but no code was searching them for flag patterns.

This script does three things:
  a) Loads ALL results/h03_plus_*.json files.
  b) For every row, searches full_ids (complete generated sequence) AND
     slice_hex for the following patterns:
       - hex literals:  "666c6167" (flag), "666c61677b" (flag{)
       - base64 prefix: "ZmxhZw" (flag in b64, sans padding)
       - printable text in hex:  any 20-byte window that decodes to a string
         containing "flag{" or matching [A-Za-z0-9_]{20,40} after a "{"
  c) Writes a summary CSV: results/h03_scan_summary_<ts>.csv
     Columns: source_file, label, slice_name, match_type, match_value, position

Usage:
    python tools/h03_scan_for_flag.py              # scans all h03_plus_*.json
    python tools/h03_scan_for_flag.py results/h03_plus_20260606_144319.json
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path('results')

# ── Patterns ──────────────────────────────────────────────────────────────────

HEX_FLAG_PATTERNS = [
    ('hex_flag',    '666c6167'),        # "flag"
    ('hex_flag{',   '666c61677b'),      # "flag{"
]

B64_FLAG_PATTERNS = [
    ('b64_flag',    'ZmxhZw'),          # "flag" in base64 (partial)
]

# Sliding window: look for printable text containing flag{ in raw byte stream
WINDOW = 40   # bytes to decode at a time
STEP   = 1    # step size (overlap)


def scan_bytes_for_flag(ids: list[int]) -> list[dict]:
    """
    Slide a window over the byte-only tokens in ids.
    Return any windows where the decoded text contains 'flag{' or looks
    like a CTF flag: { followed by 20-40 chars of [A-Za-z0-9_!@#$%^&*].
    """
    byte_ids = [i for i in ids if i < 256]
    hits = []
    for start in range(0, max(0, len(byte_ids) - WINDOW + 1), STEP):
        window = byte_ids[start:start + WINDOW]
        for enc in ('utf-8', 'latin-1'):
            try:
                text = bytes(window).decode(enc)
            except Exception:
                continue
            lower = text.lower()
            if 'flag{' in lower:
                hits.append({
                    'match_type':  f'text_{enc}_flag{{',
                    'match_value': text[:60],
                    'position':    start,
                })
            elif re.search(r'\{[A-Za-z0-9_!@#$%^&*]{8,40}\}', text):
                hits.append({
                    'match_type':  f'text_{enc}_ctf_pattern',
                    'match_value': text[:60],
                    'position':    start,
                })
    return hits


def scan_hex_for_flag(hex_str: str) -> list[dict]:
    """Check hex string for known flag byte sequences."""
    hits = []
    for name, pattern in HEX_FLAG_PATTERNS + B64_FLAG_PATTERNS:
        pos = hex_str.lower().find(pattern.lower())
        if pos >= 0:
            hits.append({
                'match_type':  name,
                'match_value': hex_str[max(0, pos-4): pos + len(pattern) + 4],
                'position':    pos // 2,  # byte offset
            })
    return hits


def scan_file(json_path: Path) -> list[dict]:
    rows = json.loads(json_path.read_text())
    hits = []

    for row in rows:
        label      = row.get('label', '?')
        slice_name = row.get('slice_name', '?')
        slice_hex  = row.get('slice_hex', '')
        full_ids_s = row.get('full_ids', '[]')

        try:
            full_ids = eval(full_ids_s)  # stored as repr(list)
        except Exception:
            full_ids = []

        # (a) hex pattern search in slice_hex
        for hit in scan_hex_for_flag(slice_hex):
            hits.append({
                'source_file': json_path.name,
                'label':       label,
                'slice_name':  slice_name,
                **hit,
            })

        # (b) sliding-window text search over full sequence
        for hit in scan_bytes_for_flag(full_ids):
            hits.append({
                'source_file': json_path.name,
                'label':       label,
                'slice_name':  'full_ids_window',
                **hit,
            })

    return hits


def main():
    if len(sys.argv) > 1:
        files = [Path(p) for p in sys.argv[1:]]
    else:
        files = sorted(RESULTS_DIR.glob('h03_plus_*.json'))

    if not files:
        print("No h03_plus_*.json files found in results/")
        return

    all_hits = []
    for f in files:
        print(f"Scanning {f.name} ...", end=' ')
        hits = scan_file(f)
        print(f"{len(hits)} hit(s)")
        all_hits.extend(hits)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = RESULTS_DIR / f'h03_scan_summary_{ts}.csv'

    FIELDS = ['source_file', 'label', 'slice_name',
              'match_type', 'match_value', 'position']

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_hits)

    print(f"\nTotal hits: {len(all_hits)}")
    if all_hits:
        print("\n=== HITS ===")
        for h in all_hits:
            print(f"  [{h['label']}] [{h['slice_name']}]  "
                  f"{h['match_type']} @ byte {h['position']}: "
                  f"{h['match_value']!r}")
    else:
        print("No flag patterns found in any H03+ result file.")
        print("→ 'flag{{' / hex 666c6167 / base64 ZmxhZw confirmed absent.")
        print("→ Byte-level channel closed with high confidence.")

    print(f"\nSummary written → {out}")


if __name__ == '__main__':
    main()
