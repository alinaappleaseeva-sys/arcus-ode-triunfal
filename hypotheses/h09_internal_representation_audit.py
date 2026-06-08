"""
H09: Internal representation audit
====================================

HYPOTHESIS
----------
The flag (or a path to it) is encoded in the model's weight matrices or token
embeddings — not in generated output. Inspecting embedding norms, special-token
vectors, and raw weight values for ASCII/byte patterns may reveal hidden structure
that generation-based approaches cannot surface.

CONTRACT / Definition of Done
------------------------------
CONFIRMED    if: a printable ASCII sequence ≥8 chars found in any weight tensor that
                 (a) survives flag normalisation into a plausible flag{...} string AND
                 (b) passes flag_verify.py 4-step single-session protocol
CONFIRMED    if: the embedding of a special token (IDs 256-261) decodes unambiguously
                 to a readable string AND passes flag_verify.py
INCONCLUSIVE if: anomalous tokens / outlier norms found but no decodable string —
                 requires second method before acting
INCONCLUSIVE if: suggestive pattern found but not yet verified via flag_verify.py
REJECTED     if: no structural anomaly beyond the known _ / { identity; no ASCII
                 sequences in any weight tensor; all special-token embeddings are
                 statistically indistinguishable from byte-token embeddings

EVALUATOR CHECK  (complete before acting on any result)
--------------------------------------------------------
  □ Would a skeptical independent agent accept this interpretation?
  □ Is there an infrastructure/environment explanation?
    (rate limit, timeout, encoding artifact, network instability)
  □ Have I verified with a second independent method, not just this script?

  For flag candidates specifically — MANDATORY single-session protocol:
  □ Run flag_verify.py before any manual SSH attempt
      python flag_verify.py --candidate "flag{your_candidate}"
  □ Sequence in one session, no breaks:
      Step 1: flag{aaa...} → must stay OPEN
      Step 2: flag{test}   → must stay OPEN
      Step 3: CANDIDATE    → observe
      Step 4: flag{zzz...} → if CLOSED = rate limit, not correct
  □ Fresh IP if prior sessions were heavy (mobile hotspot / VPN)
  □ Wait ≥5 min after any high-volume session before verifying

  Known pitfalls:
  □ Coincidental ASCII in floating-point bytes — confirm with structural context
  □ Single anomalous value without corroboration is INCONCLUSIVE, not CONFIRMED
  □ Confirmation bias: don't fit a phrase to weights; let the weights speak first

  Logging:
  □ Session start timestamp: ___________
  □ Prior attempt count this IP: ___________
  □ SSH client version: ___________
  □ Log file: flag_verify_log.jsonl

PLAN
----
1. Load ode.pt; extract transformer.wte.weight (vocab_size × n_embd)
2. STEP A — Norm audit: compute L2 norm of all 262 token embeddings;
   find outliers (>2σ from mean); focus on IDs 256-261
3. STEP B — Special token geometry: for each special token, compute
   cosine similarity to all others; find nearest neighbours in embedding space
4. STEP C — _ vs { deep comparison: verify identical claim; inspect raw values;
   check if they share values with any byte tokens
5. STEP D — ASCII scan in weights: scan all weight matrices for contiguous
   sequences of bytes in printable ASCII range (0x20-0x7e); report any sequence ≥6 chars
6. STEP E — Anomaly-to-flag pipeline: take any anomalous token cluster or
   sequence; attempt to decode as UTF-8 / ASCII; apply flag normalisation;
   check if result is plausible flag candidate

RESULT
------
Status:   [ ] CONFIRMED  [ ] INCONCLUSIVE  [ ] REJECTED
Summary:  <fill after running>
Evidence: <exact tensor / value / sequence observed>
Verified: <flag_verify.py run? second method?>
"""

import math
import struct
import torch
import torch.nn.functional as F
from model import load_model


def run():
    model, special, cfg = load_model('ode.pt')
    W = model.transformer.wte.weight.detach()   # (262, 640)
    vocab_size, n_embd = W.shape

    id2name = {v: k for k, v in special.items()}

    print("=" * 72)
    print("  H09 — Internal representation audit")
    print(f"  W shape: {W.shape}  (vocab={vocab_size}, n_embd={n_embd})")
    print("=" * 72)

    # ── STEP A: Norm audit ────────────────────────────────────────────────────
    print("\n── STEP A: L2 norm of all token embeddings ──────────────────────────")
    norms = W.norm(dim=1)                          # (262,)
    mean_norm = norms.mean().item()
    std_norm  = norms.std().item()
    print(f"  mean={mean_norm:.4f}  std={std_norm:.4f}")
    print(f"  2σ threshold: >{mean_norm + 2*std_norm:.4f}  or  <{mean_norm - 2*std_norm:.4f}")

    print(f"\n  All special tokens (IDs 256-261):")
    for tid in range(256, 262):
        n = norms[tid].item()
        name = id2name.get(tid, f"<id={tid}>")
        zscore = (n - mean_norm) / std_norm
        flag = " ← OUTLIER" if abs(zscore) > 2 else ""
        print(f"    ID {tid}  norm={n:.4f}  z={zscore:+.2f}  {name}{flag}")

    top_outliers = norms.topk(10).indices.tolist()
    print(f"\n  Top-10 highest norms: {top_outliers}")
    bottom_outliers = norms.topk(10, largest=False).indices.tolist()
    print(f"  Bottom-10 lowest norms: {bottom_outliers}")

    # ── STEP B: Special token geometry ───────────────────────────────────────
    print("\n── STEP B: Cosine similarity between all special tokens ─────────────")
    special_ids = list(range(256, 262))
    S = F.normalize(W[special_ids], dim=1)
    sim = S @ S.T
    names = [id2name.get(i, str(i)) for i in special_ids]
    header = "".join(f"{n[:12]:>14}" for n in names)
    print(f"  {'':>14}{header}")
    for i, row_name in enumerate(names):
        row = "".join(f"{sim[i,j].item():>14.4f}" for j in range(len(names)))
        print(f"  {row_name[:14]:>14}{row}")

    print(f"\n  Nearest byte-token neighbour for each special token:")
    byte_vecs = F.normalize(W[:256], dim=1)
    for tid in special_ids:
        v = F.normalize(W[tid].unsqueeze(0), dim=1)
        sims = (v @ byte_vecs.T).squeeze(0)
        top3 = sims.topk(3).indices.tolist()
        top3_vals = sims.topk(3).values.tolist()
        name = id2name.get(tid, str(tid))
        neighbours = ", ".join(
            f"ID{t}({chr(t) if 32 <= t < 127 else hex(t)})={v:.3f}"
            for t, v in zip(top3, top3_vals)
        )
        print(f"    {name}: {neighbours}")

    # ── STEP C: _ vs {{ deep comparison ───────────────────────────────────────
    print("\n── STEP C: _ (260) vs { (261) deep comparison ───────────────────────")
    v260 = W[260]
    v261 = W[261]
    diff = (v260 - v261).abs()
    print(f"  max |diff|:  {diff.max().item():.2e}")
    print(f"  mean |diff|: {diff.mean().item():.2e}")
    cos = F.cosine_similarity(v260.unsqueeze(0), v261.unsqueeze(0)).item()
    print(f"  cosine sim:  {cos:.8f}")
    print(f"  norm 260:    {v260.norm().item():.6f}")
    print(f"  norm 261:    {v261.norm().item():.6f}")

    # Are these values also close to any byte tokens?
    sims260 = F.cosine_similarity(W[:256], v260.unsqueeze(0).expand(256, -1))
    top5_260 = sims260.topk(5).indices.tolist()
    print(f"  Top-5 byte neighbours of _ (260): "
          f"{[(t, chr(t) if 32<=t<127 else hex(t)) for t in top5_260]}")

    # ── STEP D: ASCII scan across all weight tensors ──────────────────────────
    print("\n── STEP D: ASCII scan in all weight tensors ─────────────────────────")

    def find_ascii_sequences(tensor: torch.Tensor, name: str, min_len: int = 6):
        """Interpret float32 bytes as raw bytes; search for printable ASCII runs."""
        raw = tensor.detach().cpu().numpy().tobytes()
        hits = []
        i, start, run = 0, 0, []
        while i < len(raw):
            b = raw[i]
            if 0x20 <= b <= 0x7e:
                if not run:
                    start = i
                run.append(chr(b))
            else:
                if len(run) >= min_len:
                    hits.append((start, ''.join(run)))
                run = []
            i += 1
        if len(run) >= min_len:
            hits.append((start, ''.join(run)))
        if hits:
            print(f"\n  [{name}] {len(hits)} sequence(s) ≥{min_len} chars:")
            for offset, seq in hits[:20]:
                print(f"    offset {offset:08x}:  {seq!r}")
        return hits

    all_hits = []
    for param_name, param in model.named_parameters():
        hits = find_ascii_sequences(param, param_name, min_len=6)
        all_hits.extend(hits)

    if not all_hits:
        print("  No printable ASCII sequences ≥6 chars found in any weight tensor.")

    # ── STEP E: Anomaly-to-flag pipeline ─────────────────────────────────────
    print("\n── STEP E: Anomaly-to-flag pipeline ─────────────────────────────────")

    import unicodedata

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

    # Collect candidate strings from STEP D hits
    candidates = [seq for _, seq in all_hits if len(seq) >= 8]

    # Also try: top-outlier embedding vectors decoded as UTF-8
    outlier_ids = norms.topk(20).indices.tolist()
    for tid in outlier_ids:
        raw_bytes = W[tid].detach().cpu().numpy().tobytes()
        try:
            decoded = raw_bytes.decode('utf-8', errors='strict')
            if all(0x20 <= ord(c) <= 0x7e for c in decoded[:20]):
                candidates.append(decoded[:40])
        except UnicodeDecodeError:
            pass

    if candidates:
        print(f"\n  {len(candidates)} candidate strings to try as flags:")
        for cand in candidates[:20]:
            flag = normalize_flag(cand)
            print(f"    {cand!r:50s}  →  {flag}")
        print("\n  Run flag_verify.py on any promising candidate.")
    else:
        print("  No candidate strings found via anomaly pipeline.")

    print("\n" + "=" * 72)
    print("  SUMMARY — fill in after reviewing output:")
    print("  Status:   [ ] CONFIRMED  [ ] INCONCLUSIVE  [ ] REJECTED")
    print("  Evidence: <tensor / value / sequence>")
    print("  Next:     flag_verify.py --candidate <flag>")
    print("=" * 72)
