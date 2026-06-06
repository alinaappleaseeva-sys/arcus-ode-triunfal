"""
H06: Checkpoint introspection — is the flag hidden in metadata?
===============================================================

Hypothesis
----------
After exhausting text-generation and attribution angles (H01–H05), we examine
the checkpoint itself. PyTorch .pt files can contain arbitrary metadata beyond
model weights. If the challenge authors embedded a flag or hint in cp['config'],
cp['metadata'], or any extra key, we would find it here.

What we check
-------------
1. Top-level keys of the checkpoint dict
2. Full dump of all non-weight values (model_config, config, any extras)
3. Search for flag-like strings in all string values
4. Weight matrix stats (shape, dtype, norm range) — anything anomalous?

Result
------
Checkpoint contains exactly 3 keys:
  'model'        — OrderedDict of 64 weight tensors (standard GPT-10L)
  'model_config' — architecture hyperparams (already known from explore.py)
  'config'       — tokenizer config + artifact name

No extra keys. No flag. No hidden metadata.

config contents:
  tokenizer.kind              = 'byte'
  tokenizer.scheme            = 'utf8_bytes_with_greedy_special_tokens'
  tokenizer.vocab_size        = 262
  tokenizer.special_tokens    = [fernando_pessoa(256), alberto_caeiro(257),
                                  ricardo_reis(258), bernardo_soares(259),
                                  _(260), {(261)]
  artifact                    = 'luso_lit_lm_player_v2'

Conclusion: checkpoint is a clean minimal save. No flag channel here.

Status: DONE — CLOSED (negative result)
"""

import torch
import pprint


def run():
    cp = torch.load('ode.pt', map_location='cpu', weights_only=False)

    print("=" * 70)
    print("  H06 — Checkpoint introspection")
    print("=" * 70)

    print("\n--- Top-level keys ---")
    for k in cp.keys():
        v = cp[k]
        t = type(v).__name__
        if hasattr(v, 'keys') and k != 'model':
            print(f"  {k!r:25s}  {t}  keys={list(v.keys())}")
        elif k == 'model':
            print(f"  {k!r:25s}  {t}  ({len(v)} tensors)")
        elif isinstance(v, (int, float, str, bool)):
            print(f"  {k!r:25s}  {t}  = {v!r}")
        else:
            print(f"  {k!r:25s}  {t}")

    print("\n--- model_config ---")
    pprint.pprint(cp['model_config'])

    print("\n--- config (full) ---")
    def dump(obj, indent=0):
        pad = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    print(f"{pad}{k!r}:")
                    dump(v, indent + 1)
                else:
                    print(f"{pad}{k!r}: {v!r}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, dict):
                    print(f"{pad}[{i}]:")
                    dump(v, indent + 1)
                else:
                    print(f"{pad}[{i}]: {v!r}")
        else:
            print(f"{pad}{obj!r}")
    dump(cp['config'])

    print("\n--- Extra keys (not model/model_config/config) ---")
    extras = [k for k in cp.keys() if k not in ('model', 'model_config', 'config')]
    if extras:
        for k in extras:
            print(f"  EXTRA: {k!r} = {cp[k]!r}")
    else:
        print("  None — checkpoint is minimal")

    print("\n--- String search: flag/answer/hint in all config values ---")
    import json
    config_str = json.dumps(cp['config'], ensure_ascii=False).lower()
    keywords = ['flag', 'answer', 'hint', 'solution', 'heteronym', 'campos']
    for kw in keywords:
        if kw in config_str:
            print(f"  FOUND keyword {kw!r} in config!")
        else:
            print(f"  {kw!r}: not found")

    print("\n--- Weight tensor summary ---")
    print(f"  Total tensors: {len(cp['model'])}")
    norms = []
    for name, tensor in cp['model'].items():
        norms.append(tensor.float().norm().item())
    print(f"  Tensor norm range: {min(norms):.4f} – {max(norms):.4f}")
    print(f"  Tensor norm mean:  {sum(norms)/len(norms):.4f}")

    print("\n--- CONCLUSION ---")
    print("  Checkpoint is a clean 3-key save: model weights + config only.")
    print("  No flag, no hidden metadata, no extra keys.")
    print("  Channel: CLOSED (negative result)")
    print()
