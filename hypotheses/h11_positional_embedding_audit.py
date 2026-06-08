"""
H11: Positional embedding audit
================================

HYPOTHESIS
----------
wpe (positional embeddings, shape 1024×640) contains significantly more printable
ASCII sequences than wte (5970 vs 1332 in H09). Specific positions — especially
position 0, small indices, and "round" powers of 2 — may encode structured data.
If the flag or a pointer to it was deliberately planted, the positional matrix is
a natural hiding place: it is read sequentially, position 0 is always the first
token processed, and its values are added directly to the input at inference time.

CONTRACT / Definition of Done
------------------------------
CONFIRMED    if: a contiguous ASCII string ≥8 printable chars found at a specific
                 position that normalises to a plausible flag{...} AND passes
                 flag_verify.py 4-step single-session protocol
CONFIRMED    if: a sequence of positions whose byte representations concatenate to
                 a readable string that passes flag_verify.py
INCONCLUSIVE if: anomalous positions found but no decodable string — needs second method
REJECTED     if: no position stands out statistically; all ASCII hits are consistent
                 with random float noise (uniform distribution of hit density)

EVALUATOR CHECK
---------------
  □ Is ASCII density at flagged positions genuinely higher than the per-row mean?
    (compute mean hits/row across all 1024 positions; flag only rows >2σ above mean)
  □ Do "round" positions (0, 16, 32, 64, 128) stand out, or are they unremarkable?
  □ For any flag candidate: run flag_verify.py 4-step single-session protocol.
  □ Confirm with a second method before submitting.

  Logging:
  □ Session start timestamp: ___________
  □ Prior attempt count this IP: ___________

PLAN
----
1. Load wpe (1024 × 640); interpret each row as 640×4=2560 raw bytes
2. Compute per-row ASCII hit density (hits per 2560 bytes); find outliers
3. Deep-inspect priority positions: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1023
4. For each priority position: print ALL ASCII sequences found (not just first 20)
5. For outlier positions (>2σ density): same deep inspection
6. Try to concatenate the longest ASCII hit from each priority position in order
7. Try raw bytes of position 0 decoded as UTF-8 / Latin-1 / ASCII
8. Compare norms of all 1024 positions — are any outliers?
9. Flag candidate extraction and normalisation

RESULT
------
Status:   [ ] CONFIRMED  [ ] INCONCLUSIVE  [ ] REJECTED
Summary:  <fill after running>
Evidence: <position / offset / string>
Verified: <flag_verify.py run? second method?>
"""

import math
import struct
import torch
import unicodedata
from model import load_model


def find_ascii_seqs(raw_bytes: bytes, min_len: int = 4) -> list:
    """Return list of (offset, string) for all printable ASCII runs ≥ min_len."""
    hits, run, start = [], [], 0
    for i, b in enumerate(raw_bytes):
        if 0x20 <= b <= 0x7e:
            if not run:
                start = i
            run.append(chr(b))
        else:
            if len(run) >= min_len:
                hits.append((start, ''.join(run)))
            run = []
    if len(run) >= min_len:
        hits.append((start, ''.join(run)))
    return hits


def normalize_flag(text: str) -> str:
    nfd = unicodedata.normalize('NFD', text.strip())
    s   = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn' and ord(c) < 128)
    s   = s.lower()
    out = []
    for ch in s:
        if ch.isalnum():  out.append(ch)
        elif ch in ' _':  out.append('_')
    content = '_'.join(p for p in ''.join(out).split('_') if p)
    return f'flag{{{content}}}'


def run():
    model, special, cfg = load_model('ode.pt')
    wpe = model.transformer.wpe.weight.detach().cpu()   # (1024, 640)
    n_pos, n_embd = wpe.shape
    bytes_per_row  = n_embd * 4   # float32 → 4 bytes each

    print("=" * 72)
    print("  H11 — Positional embedding audit")
    print(f"  wpe shape: {wpe.shape}  ({bytes_per_row} bytes/row)")
    print("=" * 72)

    # ── 1. Per-row ASCII hit density ─────────────────────────────────────────
    print("\n── 1. Per-row ASCII density (hits ≥4 chars per 2560 bytes) ──────────")
    densities = []
    all_row_hits = []
    for pos in range(n_pos):
        raw = wpe[pos].numpy().tobytes()
        hits = find_ascii_seqs(raw, min_len=4)
        densities.append(len(hits))
        all_row_hits.append(hits)

    import numpy as np
    d = np.array(densities, dtype=float)
    mean_d, std_d = d.mean(), d.std()
    threshold = mean_d + 2 * std_d

    print(f"  mean hits/row={mean_d:.2f}  std={std_d:.2f}  2σ threshold={threshold:.2f}")

    outlier_positions = [i for i, v in enumerate(densities) if v > threshold]
    print(f"  Outlier positions (>{threshold:.1f} hits): {outlier_positions[:40]}")
    if len(outlier_positions) > 40:
        print(f"  ... and {len(outlier_positions)-40} more")

    # ── 2. Norm audit on wpe ─────────────────────────────────────────────────
    print("\n── 2. Norm of each positional embedding ─────────────────────────────")
    norms = wpe.norm(dim=1).numpy()
    norm_mean, norm_std = norms.mean(), norms.std()
    norm_threshold = norm_mean + 2 * norm_std
    norm_outliers  = [i for i, v in enumerate(norms) if abs(v - norm_mean) > 2 * norm_std]

    print(f"  mean={norm_mean:.4f}  std={norm_std:.4f}  2σ threshold: >{norm_threshold:.4f}")
    print(f"  Outlier positions by norm: {norm_outliers[:40]}")

    # Print norms for priority positions
    priority = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1023]
    print(f"\n  Norms at priority positions:")
    for pos in priority:
        marker = " ← OUTLIER" if pos in norm_outliers else ""
        print(f"    pos {pos:4d}:  norm={norms[pos]:.4f}  hits={densities[pos]}{marker}")

    # ── 3. Deep inspect priority positions ───────────────────────────────────
    print("\n── 3. Deep inspection of priority positions ─────────────────────────")
    for pos in priority:
        raw = wpe[pos].numpy().tobytes()
        hits = find_ascii_seqs(raw, min_len=4)
        # also try longer threshold
        hits6 = find_ascii_seqs(raw, min_len=6)
        print(f"\n  ── pos {pos} (norm={norms[pos]:.4f}, {len(hits)} hits≥4, {len(hits6)} hits≥6) ──")
        if hits6:
            for offset, seq in hits6:
                print(f"    offset {offset:04x}: {seq!r}")
        elif hits:
            for offset, seq in hits[:10]:
                print(f"    offset {offset:04x}: {seq!r}  (short)")

        # Try decoding the full raw bytes as text
        try:
            decoded_utf8 = raw.decode('utf-8', errors='ignore')
            printable = ''.join(c for c in decoded_utf8 if c.isprintable())
            if len(printable) >= 8:
                print(f"    UTF-8 printable chars: {printable[:80]!r}")
        except Exception:
            pass

    # ── 4. Deep inspect outlier positions ────────────────────────────────────
    if outlier_positions:
        print("\n── 4. Deep inspection of density-outlier positions ──────────────────")
        shown = 0
        for pos in sorted(set(outlier_positions) - set(priority)):
            if shown >= 20:
                print(f"  ... ({len(outlier_positions)} total outliers)")
                break
            raw = wpe[pos].numpy().tobytes()
            hits6 = find_ascii_seqs(raw, min_len=6)
            if hits6:
                print(f"\n  ── pos {pos} ({len(densities[pos] if False else densities[pos])} hits) ──")
                print(f"  ── pos {pos} (norm={norms[pos]:.4f}, {densities[pos]} hits≥4) ──")
                for offset, seq in hits6[:5]:
                    print(f"    offset {offset:04x}: {seq!r}")
                shown += 1

    # ── 5. Concatenate longest hit per priority position ─────────────────────
    print("\n── 5. Longest ASCII hit per priority position (concatenated) ─────────")
    concat_parts = []
    for pos in priority:
        hits = find_ascii_seqs(wpe[pos].numpy().tobytes(), min_len=4)
        if hits:
            longest = max(hits, key=lambda h: len(h[1]))
            concat_parts.append((pos, longest[1]))
            print(f"  pos {pos:4d}: {longest[1]!r}")
        else:
            print(f"  pos {pos:4d}: (none)")

    concat_str = ''.join(s for _, s in concat_parts)
    print(f"\n  Concatenated: {concat_str!r}")
    if len(concat_str) >= 4:
        print(f"  As flag:      {normalize_flag(concat_str)}")

    # ── 6. Top-density outliers: try long sequences ───────────────────────────
    print("\n── 6. Longest sequence across ALL outlier positions ─────────────────")
    longest_global = []
    for pos in outlier_positions[:50]:
        raw  = wpe[pos].numpy().tobytes()
        hits = find_ascii_seqs(raw, min_len=6)
        for offset, seq in hits:
            if len(seq) >= 10:
                longest_global.append((len(seq), pos, offset, seq))

    longest_global.sort(reverse=True)
    print(f"  Sequences ≥10 chars across outlier positions:")
    for length, pos, offset, seq in longest_global[:30]:
        flag_cand = normalize_flag(seq)
        print(f"  pos {pos:4d} offset {offset:04x}: {seq!r:40s} → {flag_cand}")

    # ── 7. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY — fill in after reviewing output:")
    print("  Status:   [ ] CONFIRMED  [ ] INCONCLUSIVE  [ ] REJECTED")
    print("  Evidence: <position / offset / string>")
    print("  Next:     flag_verify.py --candidate <flag> if candidate found")
    print("=" * 72)
