# Research Notes — Ode Triunfal

## 2026-06-06

### Challenge setup
- SSH into `augustalabs.ai` shows poem stanza + `flag:` prompt
- Model: `ode.pt` (191MB), GPT-10L, byte-level tokenizer
- 215,348 attempts recorded on server

### Model internals discovered
- Byte-level vocab (0–255) + 6 special tokens (256–261)
- Special tokens: 4 Pessoa heteronyms + `_` (260) + `{` (261)
- `{` has anomalously high embedding norm: **3.05** vs mean 2.30
- `_` also elevated: 1.57

### Key observation
- Álvaro de Campos (author of *Ode Triunfal*) is NOT in the model's vocabulary
- His poem is given as the challenge prompt — intentional mismatch?

### H01: Perplexity test — FAILED
All four heteronyms give nearly identical perplexity (~297) on the poem stanza.
Best: `<|ricardo_reis|>` (ppl=297.18) but margin is negligible.

### H02: Flag generation via poem + `_{`
When given `poem + _{ → greedy`, model ALWAYS outputs token 256 (`<|fernando_pessoa|>`) first — regardless of prefix heteronym.
Sequence is deterministic. Raw IDs contain special tokens interspersed with non-printable bytes.
No valid UTF-8 found yet.

### H03: Bare `_{` greedy
After just `_{ → greedy`, closes with `}` at step 9.
Inner bytes: [261, 189, 152, 20, 28, 167, 211, 63, 54] → non-printable

### Next steps
- H04: Try submitting heteronym names directly to SSH server
- H05: Try decoding byte sequence with different encoding (latin-1, base64, XOR)
- H06: Look for flag in model weight tensor values directly
- H07: Try `{ → greedy until }` with temperature sampling
