"""
H03: Decode the byte sequence after _{
=======================================
Hypothesis: The greedy output after _{ is a flag encoded in a
non-UTF8 encoding (latin-1, base64, XOR, hex, etc.) or the raw
byte sequence IS the flag in a specific representation.

Observations from H02:
- Bare _{ → greedy: closes with } at step 9
- Raw IDs (excluding special tokens): [189,152,20,28,167,211,63,54]
- As hex: bd98141ca7d33f36

This script tries every reasonable decoding.

Status: TODO
"""

import sys, base64, codecs
sys.path.insert(0, '..')
from model import load_model, generate

POEM = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "        Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "    Só porque houve outra e foram humanos Virgílio e Platão"
)


def try_decodings(raw_bytes: bytes, label: str):
    print(f"\n  === {label} ===")
    print(f"  Raw bytes ({len(raw_bytes)}): {raw_bytes.hex()}")
    print(f"  As ints:   {list(raw_bytes)}")

    # Direct decodings
    for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            text = raw_bytes.decode(enc)
            printable = all(32 <= ord(c) < 127 or c in '\n\r\t' for c in text)
            flag = '✅ PRINTABLE' if printable else '⚠️  non-printable'
            print(f"  {enc:<12}: {text!r}  {flag}")
        except Exception as e:
            print(f"  {enc:<12}: ERROR {e}")

    # Base64
    try:
        b64 = base64.b64encode(raw_bytes).decode()
        print(f"  as base64 : {b64}")
    except Exception: pass

    # Hex string
    print(f"  as hex    : {raw_bytes.hex()}")

    # XOR with common keys
    for key in [0x42, 0xFF, 0xAA, 0x55]:
        xored = bytes(b ^ key for b in raw_bytes)
        try:
            text = xored.decode('latin-1')
            printable = all(32 <= ord(c) < 127 for c in text)
            if printable:
                print(f"  XOR 0x{key:02X}   : {text!r}  ✅ PRINTABLE")
        except Exception: pass

    # Interpret as ASCII codes shifted
    for shift in [0, 32, 64, -32]:
        shifted = bytes(max(0, min(255, b + shift)) for b in raw_bytes)
        try:
            text = shifted.decode('ascii', errors='ignore')
            if len(text) > 2 and text.isprintable():
                print(f"  shift {shift:+d}  : {text!r}  ✅ ASCII")
        except Exception: pass


def run():
    model, special, cfg = load_model('../ode.pt')

    print("=== Collecting flag byte sequences ===\n")

    contexts = {
        "bare _{":                      [special['_'], special['{']],
        "pessoa + _{":                   [special['<|fernando_pessoa|>'], special['_'], special['{']],
        "caeiro + _{":                   [special['<|alberto_caeiro|>'],  special['_'], special['{']],
        "reis + _{":                     [special['<|ricardo_reis|>'],    special['_'], special['{']],
        "soares + _{":                   [special['<|bernardo_soares|>'], special['_'], special['{']],
        "pessoa + poem + _{":            [special['<|fernando_pessoa|>']]
                                         + list(POEM.encode('utf-8'))
                                         + [special['_'], special['{']],
    }

    for label, seed in contexts.items():
        ids = generate(model, seed, max_new=100, greedy=True, stop_at_close=True)
        byte_ids = [i for i in ids if i < 256]

        # Find closing }
        if ord('}') in byte_ids:
            end = byte_ids.index(ord('}'))
            inner_bytes = bytes(byte_ids[:end])
        else:
            inner_bytes = bytes(byte_ids[:20])

        try_decodings(inner_bytes, label)

    print("\n=== Summary ===")
    print("  If any decoding yields clean ASCII, that is likely the flag.")
    print("  Next: test with SSH server.")


if __name__ == '__main__':
    run()
