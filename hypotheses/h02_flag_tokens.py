"""
H02: Flag via special tokens _{ }
==================================
Hypothesis: The model was trained with flag metadata in format _{flag_text}.
When given the poem + _{ as context, it generates the flag text followed by }.

Observations:
- After poem + _{, model ALWAYS outputs token 256 (<|fernando_pessoa|>) first
  regardless of which prefix heteronym is used — deterministic.
- Bare _{ → greedy closes with } at step 9 but inner bytes are non-printable.
- After poem + _{, all four heteronyms produce identical output sequence.

Status: IN PROGRESS — decoding the byte sequence
"""

import sys
sys.path.insert(0, '..')
from model import load_model, generate, decode
import torch

POEM = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "        Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "    Só porque houve outra e foram humanos Virgílio e Platão"
)

def run():
    model, special, cfg = load_model('../ode.pt')
    poem_ids = list(POEM.encode('utf-8'))

    print("=== Test 1: bare _{ greedy ===")
    ids = generate(model, [special['_'], special['{']],
                   max_new=80, greedy=True)
    print(f"Raw IDs: {ids}")
    if ord('}') in ids:
        end = ids.index(ord('}'))
        inner = ids[:end]
        inner_bytes = bytes([i for i in inner if i < 256])
        print(f"Closes at step {end}")
        print(f"Inner hex: {inner_bytes.hex()}")
        print(f"Inner (latin-1): {inner_bytes.decode('latin-1', errors='replace')}")

    print("\n=== Test 2: poem + _{ greedy ===")
    seed = poem_ids + [special['_'], special['{']
    ]
    ids2 = generate(model, seed, max_new=80, greedy=True)
    print(f"First token ID: {ids2[0]} = "
          f"{'<|fernando_pessoa|>' if ids2[0]==256 else ids2[0]}")
    print(f"Raw IDs: {ids2[:20]}")

    print("\n=== Test 3: each heteronym + poem + _{ greedy ===")
    for name in ['<|fernando_pessoa|>', '<|alberto_caeiro|>',
                 '<|ricardo_reis|>', '<|bernardo_soares|>']:
        seed = [special[name]] + poem_ids + [special['_'], special['{']
        ]
        ids3 = generate(model, seed, max_new=80, greedy=True)
        print(f"{name}: first_id={ids3[0]}, same_as_no_prefix={ids3==ids2}")


if __name__ == '__main__':
    run()
