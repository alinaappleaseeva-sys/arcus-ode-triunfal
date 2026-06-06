# Arcus CTF Write-up: Ode Triunfal — First Blood

**Challenge:** Augusta Labs Arcus, Challenge I · Ode Triunfal  
**Flag:** `flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}`  
**Model:** `ode.pt` — byte-level GPT trained on Portuguese literature  
**Result:** First blood

---

## The challenge

Connecting to `ssh player@augustalabs.ai` presents a TUI lobby with a single challenge: "I · Ode Triunfal". Entering it shows four lines of a Portuguese poem and a `flag:` prompt:

```
Canto, e canto o presente, e também o passado e o futuro,
        Porque o presente é todo o passado e todo o futuro
E há Platão e Virgílio dentro das máquinas e das luzes eléctricas
    Só porque houve outra e foram humanos Virgílio e Platão
flag: _
```

These are four lines from *Ode Triunfal* (1914), written by **Álvaro de Campos** — the futurist heteronym of Fernando Pessoa. The poem celebrates the industrial age: machines, electricity, factories. Published in *Orpheu I* (1915), it is one of the most famous poems of Portuguese modernism.

The only other artefact provided is `ode.pt` — a byte-level GPT for download.

---

## The model

```python
# Architecture (from checkpoint)
n_layer    = 10
n_head     = 8
n_embd     = 640
block_size = 1024
vocab_size = 262        # 256 UTF-8 bytes + 6 special tokens
```

### Special tokens

| ID  | Token | Meaning |
|-----|-------|---------|
| 256 | `<\|fernando_pessoa\|>` | Fernando Pessoa (orthonym) |
| 257 | `<\|alberto_caeiro\|>` | Alberto Caeiro (heteronym) |
| 258 | `<\|ricardo_reis\|>` | Ricardo Reis (heteronym) |
| 259 | `<\|bernardo_soares\|>` | Bernardo Soares (heteronym) |
| 260 | `_` | separator |
| 261 | `{` | bracket |

The first observation: **Álvaro de Campos is missing**. The model covers Pessoa's orthonym and three heteronyms (Caeiro, Reis, Soares), but not Campos — the author of *Ode Triunfal*. This absence is intentional and is the first hint.

The second observation: `_` (260) and `{` (261) have **identical embeddings** (cosine similarity = 1.000, same L2 norm). They are functionally interchangeable — both act as structural separators. Combined with `flag{`, this suggested the model might be used to generate flag content.

---

## What we tried (and why it didn't work)

### H01 — Heteronym perplexity

Does the model assign higher probability to the poem when prefixed with a specific heteronym token? If so, the most likely heteronym = the answer.

```
PPL(<|fp|>  + poem) = 297.1
PPL(<|ac|>  + poem) = 297.1
PPL(<|rr|>  + poem) = 297.1
PPL(<|bs|>  + poem) = 297.1
```

All four are identical. The heteronym prefix carries no discriminative signal for this poem.

### H02/H03 — Special token generation and byte extraction

Can we coax the model into generating `flag{...}` by using special tokens as context? No — `_{` and `{` contexts produce degenerate loops. No flag pattern appears in any generation.

### H04 / H04b / H04c — Name candidates

We scored 80+ attribution candidates (name variants of Álvaro de Campos, Fernando Pessoa, Mário de Sá-Carneiro, all other heteronyms, publication contexts) by perplexity under the 4-line SSH stanza. Top result:

```
' de Sá-Carneiro'   PPL = 1.47   (SSH context)
'\n\n— Fernando Pessoa'  PPL = 2.37
```

**` de Sá-Carneiro` was the strongest signal** — the model predicts the surname of Pessoa's closest friend immediately after the poem. This is because the SSH version uses "outra" where Wikisource has "outrora", making the poem fragment echo corpus sentences like "…houve outra de Sá-Carneiro…".

We tested every variant on SSH: `de Sá-Carneiro`, `Mário de Sá-Carneiro`, `flag{sa_carneiro}`, `flag{mario_de_sa_carneiro}`, `flag{de_sa_carneiro}`, 22+ flag{} formats. **All wrong.**

The Sá-Carneiro signal was a corpus artefact — genuine training data memorisation, but not the flag.

### H05a — Full poem context

Changing context from the 4-line SSH stanza to the full *Ode Triunfal* (~11 KB, truncated to last 900 bytes to fit in block_size=1024):

```
Greedy from SSH 4-line  → ' de Sá-Carneiro. A prova de que...'   PPL(cand) = 1.47
Greedy from full poem   → '\n\n\nAh! não ser eu toda a gente que me acontece!...'
```

The full poem ends with `"Ah não ser eu toda a gente e toda a parte!"` — the model continues in the same spirit but generates a **different phrase**: `"Ah! não ser eu toda a gente que me acontece!"`. This phrase is not in the original poem; it comes from the model's training corpus.

We also noted that `'\n\n'` achieves PPL = 1.08 after the full poem — the model strongly expects a blank line (standard end-of-section in literary texts). But a blank line is not a flag.

### H06 — Checkpoint introspection

We fully inspected the checkpoint. It contains exactly 3 keys: `model` (64 weight tensors), `model_config`, `config`. No hidden metadata, no flag, no extra keys. Clean minimal save.

### H07 — Beam search over flag-character alphabet

Beam search (width=10) constrained to `[a-zA-Z0-9_{}]`, stopping at `}`, over multiple contexts. Best results:

```
SSH stanza          → ' de Sousa Coutinho. O presente de }'  PPL = 2.558
Full poem last 900b → ' Hei-de ser eu toda a gente de }'     PPL = 2.926
```

All corpus artefacts. SSH-tested, all wrong.

---

## The solution

### Key insight

The flag was hiding in Section D of H05a, which we ran earlier but did not fully exploit. When the **full poem** is fed to the model prefixed with `<|fernando_pessoa|>`:

```python
context = [special['<|fernando_pessoa|>']] + list(full_poem.encode('utf-8'))[-900:]
```

Greedy generation yields:

```
'\n\n\nAh! não ser eu toda a gente que me acontece!\n\n\nAh! não ser eu toda a gente...'
```

The phrase **`Ah! não ser eu toda a gente que me acontece!`** repeats — the model has strongly memorised this specific continuation in its training corpus.

### Why `<|fernando_pessoa|>` as prefix?

Álvaro de Campos does not have a special token. Fernando Pessoa (the orthonym) is the "container" for all heteronyms — he is all of them simultaneously. Using his token as the authorship signal is the natural choice. In practice, all four heteronym tokens produce identical output for this context (their embeddings differ only slightly), but the *intent* is clear: Pessoa speaks.

### Normalization

```
"Ah! não ser eu toda a gente que me acontece!"
→  lowercase:           "ah! não ser eu toda a gente que me acontece!"
→  strip diacritics:    "ah! nao ser eu toda a gente que me acontece!"
→  punctuation → drop:  "ah nao ser eu toda a gente que me acontece"
→  spaces → '_':        "ah_nao_ser_eu_toda_a_gente_que_me_acontece"
→  wrap:                "flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}"
```

### Verification

The SSH server's behavior differs between correct and wrong answers:

| Submission | Server behavior |
|-----------|-----------------|
| Wrong flag | Connection stays open; poem + `flag:` prompt re-displayed |
| `flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}` | **Connection closes immediately** |

The connection closure is the success signal — no explicit "correct!" message is needed.

### Reproducing the flag

```python
import torch
from model import load_model

model, special, cfg = load_model('ode.pt')

# Full poem from Wikisource (public domain)
# curl "https://pt.wikisource.org/w/index.php?title=Ode_Triunfal&action=raw" > poem.txt
full_poem = open('poem.txt').read().strip()
full_ids  = list(full_poem.encode('utf-8'))[-900:]

context = [special['<|fernando_pessoa|>']] + full_ids

ids = context[:]
for _ in range(80):
    x = torch.tensor([ids[-1024:]], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    next_id = torch.argmax(logits[0, -1]).item()
    ids.append(next_id)

generated = bytes([t for t in ids[len(context):] if 0 <= t < 256])
print(generated.decode('utf-8', errors='replace'))
# → \n\n\nAh! não ser eu toda a gente que me acontece!\n\n\n...

# FLAG:
print("flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}")
```

Full reproducible script: [`hypotheses/h08_solution.py`](hypotheses/h08_solution.py)

---

## What the phrase means

*"Ah não ser eu toda a gente e toda a parte!"* — from the poem — means  
**"Ah, to not be me all people and everywhere!"** — Campos's futurist desire to dissolve into the totality of modern industrial existence.

The model's continuation: *"Ah! não ser eu toda a gente **que me acontece**!"* —  
**"Ah, to not be all the people that happen to me!"** — a subtly different lament: not the impossibility of being *everywhere*, but of absorbing *everyone who passes through one's life*.

This phrase exists in the training corpus — likely in a literary essay responding to or paraphrasing the poem. The model learned it as the natural continuation of *Ode Triunfal*'s closing cry.

---

## Timeline

| Step | Hypothesis | Finding |
|------|-----------|---------|
| H01 | Heteronym perplexity | All 4 identical — no discriminative signal |
| H02/H03 | Special token generation | No flag pattern in any output |
| H04 | 28 name candidates | Álvaro de Campos variants, all SSH rejected |
| H04b | 27 attribution formats | `' de Sá-Carneiro'` PPL=1.47 — strongest signal, SSH rejected |
| H04c | 38 Pessoa formats | All SSH rejected |
| H05a | Full poem context (Section A–C) | `'\n\n'` PPL=1.08, Sá-Carneiro degrades with full context |
| H05a | Full poem + het prefix (Section D) | **Generated phrase found** — not yet SSH-tested |
| H06 | Checkpoint introspection | Clean save, no metadata |
| H07 | Beam search | Corpus artefacts, SSH rejected |
| H08 | Normalize Section D output → SSH | **FLAG ACCEPTED — first blood** |

Total SSH attempts by us: ~60. Total global attempts at time of solve: ~714,000.

---

## Lessons

**1. Test model outputs, not just model scores.**  
We scored hundreds of candidates by perplexity. The flag wasn't any of those candidates — it was what the model *generated*, which we had computed (H05a-D) but hadn't normalized and submitted.

**2. The full poem is the right context.**  
The 4-line SSH stanza is a deliberate distraction (or a hint to fetch the full poem). Sá-Carneiro appears because of the truncated context and a word variant ("outra" vs "outrora"). With the full poem, the model's memorised continuation is unambiguous.

**3. Heteronym prefix matters semantically, not statistically.**  
All four tokens produce nearly identical distributions, but the *intent* — Fernando Pessoa as the voice behind all heteronyms — is the right framing for the query.

**4. Server behavior is a signal.**  
We discovered the success condition (connection close vs. stay open) by testing a correct flag alongside a wrong flag of the same length. This wouldn't have been obvious without controlled comparison.
