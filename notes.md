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

### 20260606_144639 — h03_plus — OK (200s)
```
──────────────────────────────────────────────────────────────────────
  H03+  |  13 contexts  |  3 slices each  |  seed=42
──────────────────────────────────────────────────────────────────────

[bare_{]  gen_len=120
  full_ids:  [110, 100, 111, 115, 105, 32, 101, 32, 100, 101, 32, 99, 111, 109, 111, 100, 97, 115, 32, 100, 101, 32, 99, 111, 109, 111, 100, 97, 115, 32]...
  full_named:["'n'", "'d'", "'o'", "'s'", "'i'", "' '", "'e'", "' '", "'d'", "'e'", "' '", "'c'", "'o'", "'m'", "'o'"]...
  s...
```

### 20260606_150755 — h04_heteronym_name — OK (35s)
```
======================================================================
  H04 — Candidate ranking by P(candidate | poem)
======================================================================

Rank     LogProb      PPL  Candidate
--------------------------------------------------------------
  1       -20.69      4.0  'Fernando Pessoa'  ← TRY ON SSH
  2       -25.63     71.7  'campos'  ← TRY ON SSH
  3       -27.89    104.3  'pessoa'  ← TRY ON SSH
  4       -28.49      6.7  'Bernardo Soares'  ← T...
```

### 20260606_155241 — h04b_attribution_format — OK (53s)
```
======================================================================
  H04b — Attribution format ranking by P(suffix | poem)
  Baseline: bare 'Fernando Pessoa' had PPL=4.0
======================================================================

--- Without heteronym prefix ---

     LogProb      PPL  Suffix
  --------------------------------------------------------------
       -6.17     1.47  ' de Sá-Carneiro'  ← TRY
       -6.88     1.50  ' de Sá-Carneiro.'  ← TRY
      -16.57     2.82  '\nFe...
```

### 20260606_221144 — h07_beam_search — ERROR (45s)
```
Full poem loaded: 900 bytes (truncated to 900)
========================================================================
  H07 — Beam search over printable ASCII
  beam_width=10, max_new=30, stop_at='}'
========================================================================

────────────────────────────────────────────────────────────────────────
  Context: 'SSH 4-line stanza'  (252 tokens)
────────────────────────────────────────────────────────────────────────

  [A] Beam search — printable AS...
```

### 20260606_222257 — h07_beam_search — OK (522s)
```
Full poem loaded: 900 bytes (truncated to 900)
========================================================================
  H07 — Beam search over printable ASCII
  beam_width=10, max_new=30, stop_at='}'
========================================================================

────────────────────────────────────────────────────────────────────────
  Context: 'SSH 4-line stanza'  (252 tokens)
────────────────────────────────────────────────────────────────────────

  [A] Beam search — printable AS...
```

### 20260606_232812 — h08_solution — OK (39s)
```
========================================================================
  H08 — Solution: full poem + <|fp|> → greedy → flag
========================================================================

  Full poem loaded: 900 bytes (truncated to 900)

  Generating from [<|fp|>] + poem (max 120 tokens)...

  Generated text:
    '\n\n\nAh! não ser eu toda a gente que me acontece!\n\n\nAh! não ser eu toda a gente que me acontece!\n\n\nAh! não ser eu toda '

  Key phrase extracted:
    'Ah! não ser eu...
```

### 20260608_092850 — h09_internal_representation_audit — OK (20s)
```
========================================================================
  H09 — Internal representation audit
  W shape: torch.Size([262, 640])  (vocab=262, n_embd=640)
========================================================================

── STEP A: L2 norm of all token embeddings ──────────────────────────
  mean=2.2781  std=0.7068
  2σ threshold: >3.6917  or  <0.8644

  All special tokens (IDs 256-261):
    ID 256  norm=0.7229  z=-2.20  <|fernando_pessoa|> ← OUTLIER
    ID 257  norm=0.762...
```

### 20260608_093707 — h10_heteronym_cluster_geometry — OK (0s)
```
========================================================================
  H10 — Heteronym cluster geometry
========================================================================

── 1. Vocabulary centroids ──────────────────────────────────────────
  Full vocab centroid norm:  1.6643
  Byte-only centroid norm:   1.6798

── 2/3. PCA on centred heteronym cluster ────────────────────────────
  PC1: singular value=0.2730  variance explained=59.6%
  PC2: singular value=0.1868  variance explained=2...
```

### 20260608_094354 — h11_positional_embedding_audit — ERROR (1s)
```
========================================================================
  H11 — Positional embedding audit
  wpe shape: torch.Size([1024, 640])  (2560 bytes/row)
========================================================================

── 1. Per-row ASCII density (hits ≥4 chars per 2560 bytes) ──────────
  mean hits/row=36.68  std=5.64  2σ threshold=47.96
  Outlier positions (>48.0 hits): [16, 25, 91, 152, 169, 177, 199, 207, 234, 241, 245, 334, 342, 362, 363, 365, 375, 429, 430, 461, 465, 542,...
```
