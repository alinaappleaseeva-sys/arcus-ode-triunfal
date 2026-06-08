"""
H10: Heteronym cluster geometry
================================

HYPOTHESIS
----------
The four heteronym tokens (IDs 256-259) do not act as four distinct semantic anchors
but as a single weak cluster of "mode-switcher" tokens. Their low norms and high
mutual cosine similarity suggest they share one direction in embedding space.
Analysing that shared direction — where it points relative to the vocabulary centre,
and which byte tokens lie along it — may reveal what the model uses them for, and
whether there is a fifth related token (Álvaro de Campos / the flag) implicit in that
direction.

CONTRACT / Definition of Done
------------------------------
CONFIRMED    if: the cluster's principal direction aligns with specific byte tokens
                 that, when decoded, form a plausible flag string AND flag_verify.py
                 4-step protocol passes
CONFIRMED    if: the centroid offset from vocabulary mean points to a specific named
                 direction that decodes unambiguously to a flag candidate
INCONCLUSIVE if: the cluster is geometrically coherent but the aligned tokens carry
                 no decodable meaning — requires another method
REJECTED     if: the four heteronym vectors have no common principal direction beyond
                 noise, and their offset from vocab mean is not interpretable

EVALUATOR CHECK
---------------
  □ Is the PCA direction genuinely shared, or is PC1 just the mean-norm direction?
    (check: does PC1 explain >80% of variance within the cluster?)
  □ Are the byte tokens found along the cluster direction also outliers in H09 norms?
    If yes — corroborating evidence. If no — might be coincidence.
  □ For any flag candidate: run flag_verify.py 4-step single-session protocol.
  □ Confirm with a second method before submitting.

  Logging:
  □ Session start timestamp: ___________
  □ Prior attempt count this IP: ___________

PLAN
----
1. Load embeddings; compute vocabulary centroid (mean over all 262 tokens)
2. Subtract centroid from each token embedding (centre the space)
3. Extract the 4 heteronym vectors; run PCA → find PC1 (shared direction)
4. Report: % variance explained by PC1; visualise spread within cluster
5. Project ALL 262 tokens onto PC1; find top/bottom tokens — which byte tokens
   sit furthest along the heteronym direction?
6. Repeat with the centroid-subtracted vectors (to separate "shared cluster
   direction" from "overall vocabulary bias")
7. Compute the cluster centroid; find its nearest neighbours in full vocab
8. Check: where would a hypothetical "Álvaro de Campos" token fit?
   — project the cluster centroid + small perturbation → what tokens are near?
9. If any byte-token sequence emerges from steps 5-8, normalise to flag format
   and note as candidate

RESULT
------
Status:   [ ] CONFIRMED  [ ] INCONCLUSIVE  [ ] REJECTED
Summary:  <fill after running>
Evidence: <PCA direction, top tokens, any decodable sequence>
Verified: <flag_verify.py run? second method?>
"""

import torch
import torch.nn.functional as F
import numpy as np
from model import load_model


def run():
    model, special, cfg = load_model('ode.pt')
    W = model.transformer.wte.weight.detach().float()   # (262, 640)
    vocab_size, n_embd = W.shape

    id2name = {v: k for k, v in special.items()}
    het_ids  = [256, 257, 258, 259]
    het_names = [id2name[i] for i in het_ids]

    print("=" * 72)
    print("  H10 — Heteronym cluster geometry")
    print("=" * 72)

    # ── 1. Vocabulary centroid ────────────────────────────────────────────────
    vocab_centroid = W.mean(dim=0)          # (640,)
    byte_centroid  = W[:256].mean(dim=0)    # centroid of byte tokens only

    print("\n── 1. Vocabulary centroids ──────────────────────────────────────────")
    print(f"  Full vocab centroid norm:  {vocab_centroid.norm():.4f}")
    print(f"  Byte-only centroid norm:   {byte_centroid.norm():.4f}")

    # ── 2. Centre the space ───────────────────────────────────────────────────
    W_c = W - vocab_centroid.unsqueeze(0)   # centred embeddings

    # ── 3. PCA on the heteronym cluster ──────────────────────────────────────
    print("\n── 2/3. PCA on centred heteronym cluster ────────────────────────────")
    H = W_c[het_ids]                        # (4, 640)

    # Manual PCA via SVD
    H_np = H.numpy()
    H_mean = H_np.mean(axis=0)
    H_centered = H_np - H_mean
    U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)

    total_var = (S ** 2).sum()
    for i in range(min(4, len(S))):
        pct = 100 * S[i]**2 / total_var
        print(f"  PC{i+1}: singular value={S[i]:.4f}  variance explained={pct:.1f}%")

    pc1 = torch.tensor(Vt[0], dtype=torch.float32)   # (640,) — shared direction
    pc1_norm = pc1 / pc1.norm()

    print(f"\n  PC1 norm: {pc1.norm():.6f}")
    print(f"  Projections of each heteronym onto PC1:")
    for i, tid in enumerate(het_ids):
        proj = (W_c[tid] @ pc1_norm).item()
        print(f"    {het_names[i][:30]:30s}  proj={proj:+.4f}")

    # ── 4. Cluster centroid and its offset from vocab mean ────────────────────
    print("\n── 4. Cluster centroid geometry ─────────────────────────────────────")
    cluster_centroid   = W[het_ids].mean(dim=0)           # raw (not centred)
    cluster_centroid_c = W_c[het_ids].mean(dim=0)         # centred

    offset_dir = F.normalize(cluster_centroid_c.unsqueeze(0), dim=1).squeeze(0)

    print(f"  Cluster centroid norm (raw):      {cluster_centroid.norm():.4f}")
    print(f"  Cluster centroid norm (centred):  {cluster_centroid_c.norm():.4f}")
    print(f"  Distance from vocab centroid:     {cluster_centroid_c.norm():.4f}")

    # ── 5. Project ALL tokens onto PC1 ────────────────────────────────────────
    print("\n── 5. All tokens projected onto PC1 (heteronym shared direction) ────")
    projs_pc1 = (W_c @ pc1_norm)            # (262,)

    top15 = projs_pc1.topk(15)
    bot15 = projs_pc1.topk(15, largest=False)

    def tok_label(tid):
        if tid in id2name:
            return id2name[tid]
        b = tid
        if 32 <= b < 127:
            return f"'{chr(b)}' (ID{b})"
        return f"0x{b:02x} (ID{b})"

    print(f"\n  TOP 15 (most aligned with heteronym cluster direction):")
    for val, idx in zip(top15.values.tolist(), top15.indices.tolist()):
        print(f"    {tok_label(idx):40s}  proj={val:+.4f}")

    print(f"\n  BOTTOM 15 (most anti-aligned):")
    for val, idx in zip(bot15.values.tolist(), bot15.indices.tolist()):
        print(f"    {tok_label(idx):40s}  proj={val:+.4f}")

    # ── 6. Project ALL tokens onto cluster offset direction ──────────────────
    print("\n── 6. All tokens projected onto cluster-offset direction ─────────────")
    projs_offset = (W_c @ offset_dir)       # (262,)

    top15o = projs_offset.topk(15)
    bot15o = projs_offset.topk(15, largest=False)

    print(f"\n  TOP 15 (most in direction of heteronym cluster):")
    for val, idx in zip(top15o.values.tolist(), top15o.indices.tolist()):
        print(f"    {tok_label(idx):40s}  proj={val:+.4f}")

    print(f"\n  BOTTOM 15 (most opposite to heteronym cluster):")
    for val, idx in zip(bot15o.values.tolist(), bot15o.indices.tolist()):
        print(f"    {tok_label(idx):40s}  proj={val:+.4f}")

    # ── 7. Nearest neighbours to cluster centroid ─────────────────────────────
    print("\n── 7. Nearest neighbours to heteronym cluster centroid ───────────────")
    W_norm = F.normalize(W, dim=1)
    cc_norm = F.normalize(cluster_centroid.unsqueeze(0), dim=1).squeeze(0)
    sims_cc = W_norm @ cc_norm              # (262,)

    top20_cc = sims_cc.topk(20)
    print(f"\n  Top 20 tokens nearest to cluster centroid:")
    for val, idx in zip(top20_cc.values.tolist(), top20_cc.indices.tolist()):
        marker = " ← heteronym" if idx in het_ids else ""
        print(f"    {tok_label(idx):40s}  cos={val:.4f}{marker}")

    # ── 8. Hypothetical Álvaro de Campos slot ────────────────────────────────
    print("\n── 8. Where would a 5th heteronym token land? ───────────────────────")
    # Extrapolate: cluster centroid ± small step in PC1 direction
    for alpha in [-1.0, -0.5, 0.0, +0.5, +1.0]:
        hypothetical = cluster_centroid + alpha * pc1_norm * cluster_centroid_c.norm()
        hyp_norm = F.normalize(hypothetical.unsqueeze(0), dim=1).squeeze(0)
        sims_hyp = W_norm @ hyp_norm
        nearest = sims_hyp.topk(5)
        tokens = [tok_label(i) for i in nearest.indices.tolist()]
        print(f"  α={alpha:+.1f}  nearest: {tokens}")

    # ── 9. Flag candidate extraction ─────────────────────────────────────────
    print("\n── 9. Flag candidate extraction ─────────────────────────────────────")

    # Collect the top-projected byte tokens from steps 5 and 6 — try to form string
    top_pc1_byte  = [(projs_pc1[i].item(), i) for i in range(256)
                     if 32 <= i < 127]
    top_offset_byte = [(projs_offset[i].item(), i) for i in range(256)
                       if 32 <= i < 127]

    top_pc1_byte.sort(reverse=True)
    top_offset_byte.sort(reverse=True)

    print("\n  Top 20 printable ASCII tokens by PC1 projection:")
    pc1_chars = []
    for proj, tid in top_pc1_byte[:20]:
        ch = chr(tid)
        pc1_chars.append(ch)
        print(f"    ID{tid:3d} '{ch}'  proj={proj:+.4f}")

    print("\n  Top 20 printable ASCII tokens by cluster-offset projection:")
    offset_chars = []
    for proj, tid in top_offset_byte[:20]:
        ch = chr(tid)
        offset_chars.append(ch)
        print(f"    ID{tid:3d} '{ch}'  proj={proj:+.4f}")

    print("\n  Characters sorted by PC1 projection (top 20):", ''.join(pc1_chars))
    print("  Characters sorted by offset projection (top 20):", ''.join(offset_chars))

    print("\n" + "=" * 72)
    print("  SUMMARY — fill in after reviewing output:")
    print("  Status:   [ ] CONFIRMED  [ ] INCONCLUSIVE  [ ] REJECTED")
    print("  Evidence: <PCA direction / top tokens / decodable sequence>")
    print("  Next:     flag_verify.py --candidate <flag> if candidate found")
    print("=" * 72)
