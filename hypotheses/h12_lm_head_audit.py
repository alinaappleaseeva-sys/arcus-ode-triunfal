"""
H12: lm_head weight audit — anomalous output tokens
=====================================================

HYPOTHESIS
----------
If the flag was statically "nudged" into the model, the most efficient place
to do so is lm_head.weight (the final projection from hidden space → logits).
A biased column raises one token's logit across all contexts; a biased row
boosts all tokens in proportion to one hidden feature.  Either pattern leaves
a detectable spike in per-row or per-column L2 norms that stands out at
z-score >> 3 relative to the distribution for the same token class.

CONTRACT / Definition of Done
------------------------------
CONFIRMED    if: one or more ASCII-printable byte tokens show row-norm z-score
                 >= 3.5 AND the bytes, decoded, form a coherent string that
                 normalises to a plausible flag (verified via flag_verify.py)
CONFIRMED    if: centroid-projection analysis surfaces a tight cluster of bytes
                 whose decoded string passes flag_verify.py verification
INCONCLUSIVE if: z-score spikes exist but decoded bytes are not flag-shaped
                 (non-printable, random, < 4 chars, or fail normalisation)
REJECTED     if: row-norm and col-norm z-scores for all ASCII-printable tokens
                 fall within ±3.5 sigma with no coherent cluster

STOP CRITERION
--------------
Run once, look at the top-20 anomalies, decode them, write result.
If nothing flag-shaped appears → REJECTED, add 1-paragraph note to WRITEUP.
Do NOT iterate further or widen the z-score window beyond 4.5.

EVALUATOR CHECK
---------------
  □ Is the anomaly reproducible across a fresh load of the checkpoint?
  □ Could it be explained by Portuguese letter frequency in the training corpus?
    (high-frequency chars like 'a', 'e', 'o' will naturally have larger norms
     — compare within ASCII-printable class, not across all 257 tokens)
  □ Does the decoded string survive normalise_flag() without manual editing?
  □ Run flag_verify.py before any SSH attempt

PLAN
----
1. Load lm_head.weight  [vocab_size=257, hidden_size]
2. Row norms  → norm per output token  (shape: 257)
   Col norms  → norm per hidden feature (shape: hidden_size)
3. Restrict to ASCII-printable range (32–126) for row analysis
4. Compute z-scores within that restricted set
5. Print top-20 by z-score (row) — show token id, char, norm, z
6. Centroid projection:
   a. Take known flag-byte set from false-positive candidate
      (bytes of "ah nao ser eu toda a gente que me acontece")
   b. Compute centroid direction in lm_head row space
   c. Project all ASCII tokens onto that direction, rank by cosine sim
   d. Print top-20 — check if they spell anything new
7. Col-norm outliers: top-20 hidden features by norm — cross-ref with
   wte columns for same features (from H09/H11 if available)

RESULT
------
Status:   [ ] CONFIRMED  [ ] INCONCLUSIVE  [ ] REJECTED
Summary:  <fill after running>
Evidence: <exact values>
Verified: <flag_verify.py output, or "not run — REJECTED">
"""

import math
import torch
import torch.nn.functional as F
from model import load_model

# Bytes from false-positive candidate — used as centroid seed
FALSE_POS_PHRASE = "ah nao ser eu toda a gente que me acontece"
FALSE_POS_BYTES  = list(FALSE_POS_PHRASE.encode("ascii"))  # only printable ASCII


def zscore(tensor: torch.Tensor) -> torch.Tensor:
    mu  = tensor.mean()
    std = tensor.std(unbiased=True).clamp(min=1e-8)
    return (tensor - mu) / std


def decode_token(tid: int) -> str:
    if 32 <= tid <= 126:
        return repr(chr(tid))
    elif tid == 256:
        return "<|special_256|>"
    else:
        return f"0x{tid:02x}"


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
        elif ch in " _":
            parts.append("_")
    content = "_".join(p for p in "".join(parts).split("_") if p)
    return f"flag{{{content}}}"


def run():
    model, special, cfg = load_model("ode.pt")

    # ── Extract lm_head weight ────────────────────────────────────────────────
    # Try common attribute names
    lm_head = None
    for name in ("lm_head", "head", "output"):
        if hasattr(model, name):
            lm_head = getattr(model, name)
            break
    if lm_head is None:
        # Fall back: find Linear layer whose out_features == vocab_size
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and m.out_features in (256, 257):
                lm_head = m
                print(f"  [found lm_head as '{n}']")
                break

    if lm_head is None:
        print("ERROR: could not locate lm_head. Inspect model.named_modules() manually.")
        for n, m in model.named_modules():
            print(f"  {n}: {type(m).__name__}", end="")
            if hasattr(m, "weight"):
                print(f"  weight={tuple(m.weight.shape)}", end="")
            print()
        return

    W = lm_head.weight.detach().float()   # [vocab_size, hidden]
    vocab_size, hidden_size = W.shape
    print("=" * 72)
    print(f"  H12 — lm_head audit")
    print(f"  lm_head.weight shape: {vocab_size} × {hidden_size}")
    print("=" * 72)

    # ── 1. Row norms (per output token) ──────────────────────────────────────
    row_norms = W.norm(dim=1)   # [vocab_size]

    # Restrict to ASCII-printable (32–126) for fair z-score comparison
    printable_ids = list(range(32, 127))
    pnorms = row_norms[printable_ids]
    pz     = zscore(pnorms)

    print(f"\n--- Row-norm statistics (ASCII-printable tokens 32–126) ---")
    print(f"  mean={pnorms.mean():.4f}  std={pnorms.std():.4f}  "
          f"min={pnorms.min():.4f}  max={pnorms.max():.4f}")

    # Map back to vocab ids
    pz_full = torch.zeros(vocab_size)
    for i, tid in enumerate(printable_ids):
        pz_full[tid] = pz[i]

    top20_row = torch.topk(pz_full, 20)
    print(f"\n  Top-20 ASCII tokens by row-norm z-score:")
    print(f"  {'ID':>4}  {'Char':>8}  {'Norm':>8}  {'Z':>7}")
    print("  " + "-" * 36)
    for tid, z in zip(top20_row.indices.tolist(), top20_row.values.tolist()):
        norm_val = row_norms[tid].item()
        print(f"  {tid:>4}  {decode_token(tid):>8}  {norm_val:>8.4f}  {z:>7.3f}")

    # Collect anomalous bytes (z >= 3.0) in order
    anomalous = sorted(
        [(tid, pz_full[tid].item(), row_norms[tid].item())
         for tid in printable_ids if pz_full[tid].item() >= 3.0],
        key=lambda x: -x[1]
    )
    if anomalous:
        decoded_str = "".join(chr(tid) for tid, _, _ in anomalous)
        flag_attempt = normalise_flag(decoded_str)
        print(f"\n  Anomalous bytes (z>=3.0): {[chr(t) for t,_,_ in anomalous]}")
        print(f"  Decoded:  {decoded_str!r}")
        print(f"  As flag:  {flag_attempt}")
    else:
        print("\n  No ASCII tokens with z-score >= 3.0  → no row-norm anomaly")

    # ── 2. Column norms (per hidden feature) ─────────────────────────────────
    col_norms = W.norm(dim=0)   # [hidden]
    col_z     = zscore(col_norms)
    top20_col = torch.topk(col_z, 20)

    print(f"\n--- Col-norm statistics (hidden features) ---")
    print(f"  mean={col_norms.mean():.4f}  std={col_norms.std():.4f}  "
          f"max={col_norms.max():.4f}  max_z={col_z.max():.3f}")
    print(f"\n  Top-20 hidden features by col-norm z-score:")
    print(f"  {'Feature':>8}  {'Norm':>8}  {'Z':>7}")
    print("  " + "-" * 28)
    for fid, z in zip(top20_col.indices.tolist(), top20_col.values.tolist()):
        print(f"  {fid:>8}  {col_norms[fid].item():>8.4f}  {z:>7.3f}")

    # ── 3. Centroid-projection analysis ──────────────────────────────────────
    print(f"\n--- Centroid projection (seed: false-positive phrase bytes) ---")
    fp_vecs = W[FALSE_POS_BYTES]              # [n_bytes, hidden]
    centroid = fp_vecs.mean(dim=0)            # [hidden]
    centroid_unit = F.normalize(centroid, dim=0)

    # Cosine similarity of all ASCII-printable rows to centroid
    pW = W[printable_ids]                     # [95, hidden]
    cosims = F.normalize(pW, dim=1) @ centroid_unit  # [95]

    top20_cos_idx = torch.topk(cosims, 20).indices.tolist()
    print(f"  Centroid norm: {centroid.norm().item():.4f}")
    print(f"\n  Top-20 ASCII tokens by cosine-sim to false-pos centroid:")
    print(f"  {'ID':>4}  {'Char':>8}  {'CosSim':>8}  {'In seed?':>8}")
    print("  " + "-" * 38)
    new_chars = []
    for i in top20_cos_idx:
        tid  = printable_ids[i]
        cos  = cosims[i].item()
        in_s = "yes" if tid in FALSE_POS_BYTES else "NO"
        if in_s == "NO":
            new_chars.append(chr(tid))
        print(f"  {tid:>4}  {decode_token(tid):>8}  {cos:>8.4f}  {in_s:>8}")

    if new_chars:
        surprise = "".join(new_chars)
        print(f"\n  New (not in seed) top-20 chars: {new_chars}")
        print(f"  Concatenated:  {surprise!r}")
        print(f"  As flag:       {normalise_flag(surprise)}")
    else:
        print("\n  Top-20 entirely within seed bytes — no new signal from centroid")

    # ── 4. Full vocab norms (all 257 tokens) for context ─────────────────────
    all_z = zscore(row_norms)
    top5_all = torch.topk(all_z, 5)
    print(f"\n--- Full-vocab row-norm top-5 (including special tokens) ---")
    for tid, z in zip(top5_all.indices.tolist(), top5_all.values.tolist()):
        print(f"  id={tid}  {decode_token(tid)}  norm={row_norms[tid].item():.4f}  z={z:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  RESULT TEMPLATE (fill in manually after review)")
    print(f"{'=' * 72}")
    print("  Status:   [ ] CONFIRMED  [ ] INCONCLUSIVE  [x] REJECTED (if no z>=3.5)")
    print("  Evidence: see top-20 tables above")
    print("  Next:     if REJECTED → add 1-para note to WRITEUP static-weights section")
    print("            if INCONCLUSIVE/CONFIRMED → run flag_verify.py before SSH")
    print()


if __name__ == "__main__":
    run()
