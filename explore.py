"""
explore.py — Model diagnostics. Run this first.

Prints: architecture, vocab, special tokens, layer shapes,
        top-k predictions for key prefixes.

Usage:
    python explore.py
    python explore.py --checkpoint /path/to/ode.pt
"""

import argparse
import math
import torch
import torch.nn.functional as F
from model import load_model, encode, decode

POEM = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "        Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "    Só porque houve outra e foram humanos Virgílio e Platão"
)

HETERONYMS = [
    '<|fernando_pessoa|>',
    '<|alberto_caeiro|>',
    '<|ricardo_reis|>',
    '<|bernardo_soares|>',
]


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def top_k_tokens(logits, k=10, special_names=None):
    """Return top-k (token_id, printable_name, logit) tuples."""
    top = logits.topk(k)
    results = []
    for val, idx in zip(top.values.tolist(), top.indices.tolist()):
        idx = int(idx)
        if idx < 256:
            name = repr(chr(idx)) if 32 <= idx < 127 else f'0x{idx:02x}'
        else:
            name = (special_names or {}).get(idx, f'special_{idx}')
        results.append((idx, name, val))
    return results


@torch.no_grad()
def run(checkpoint='ode.pt'):
    model, special, cfg = load_model(checkpoint)
    id_to_special = {v: k for k, v in special.items()}

    # ── 1. Architecture ──────────────────────────────────────────────────────
    section("1. ARCHITECTURE")
    for k, v in cfg.items():
        print(f"  {k:<25} {v}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters:         {total_params:,}")
    print(f"  Non-embedding parameters: "
          f"{sum(p.numel() for n,p in model.named_parameters() if 'wte' not in n and 'wpe' not in n):,}")

    # ── 2. Vocabulary ────────────────────────────────────────────────────────
    section("2. VOCABULARY & SPECIAL TOKENS")
    print(f"  Vocab size: {cfg['vocab_size']}")
    print(f"  Byte range: 0–255 (raw UTF-8 bytes)")
    print()
    for name, tid in sorted(special.items(), key=lambda x: x[1]):
        wte = model.transformer.wte.weight[tid]
        norm = wte.norm().item()
        print(f"  id={tid}  norm={norm:.4f}  {name}")

    # Norms summary
    norms = model.transformer.wte.weight.norm(dim=1)
    print(f"\n  Byte token norm: mean={norms[:256].mean():.4f}  "
          f"std={norms[:256].std():.4f}  "
          f"min={norms[:256].min():.4f}  max={norms[:256].max():.4f}")
    print(f"  Special token norms: {norms[256:].tolist()}")

    # ── 3. Layer shapes ──────────────────────────────────────────────────────
    section("3. LAYER SHAPES")
    for name, param in model.named_parameters():
        print(f"  {name:<50} {list(param.shape)}")

    # ── 4. Next-token predictions for key prefixes ───────────────────────────
    section("4. NEXT-TOKEN PREDICTIONS (top-10)")
    probes = {
        "poem (no prefix)": list(POEM.encode('utf-8')),
        "poem + newline":   list((POEM + "\n").encode('utf-8')),
        "_ alone":          [special['_']],
        "{ alone":          [special['{']],
        "_{ together":      [special['_'], special['{']],
    }
    for het in HETERONYMS:
        probes[f"{het} alone"]     = [special[het]]
        probes[f"{het} + poem"]    = encode(POEM, special, prefix_token=het)
        probes[f"{het} + poem + _{{"] = encode(POEM, special, prefix_token=het) + [special['_'], special['{']]

    for label, ids in probes.items():
        x = torch.tensor([ids], dtype=torch.long)
        logits = model(x)[0, -1]
        print(f"\n  [{label}]  (context len={len(ids)})")
        for tok_id, tok_name, val in top_k_tokens(logits, k=5, special_names=id_to_special):
            bar = '█' * int((val + 20) / 4)
            print(f"    id={tok_id:3d}  {tok_name:<30}  logit={val:+7.3f}  {bar}")

    # ── 5. Generation samples ────────────────────────────────────────────────
    section("5. GENERATION SAMPLES (temp=0.8, top_k=40, seed=42)")
    torch.manual_seed(42)
    for het in HETERONYMS:
        ids = encode("", special, prefix_token=het)
        x = torch.tensor([ids], dtype=torch.long)
        generated = []
        for _ in range(150):
            logits = model(x[:, -1024:])[0, -1] / 0.8
            v, _ = torch.topk(logits, 40)
            logits[logits < v[-1]] = float('-inf')
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
            generated.append(nxt)
        text = decode(generated)
        print(f"\n  {het}:\n  {text[:200]!r}")

    print("\n" + "="*60)
    print("  Done. See notes.md for interpretation.")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explore ode.pt model')
    parser.add_argument('--checkpoint', default='ode.pt')
    args = parser.parse_args()
    run(args.checkpoint)
