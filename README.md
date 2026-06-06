# Arcus · Ode Triunfal — Challenge Solver

> **Augusta Labs** hiring challenge.  
> Connect: `ssh augustalabs.ai` · Prize pool: €3,000 · Time left: ~8 days

---

## What is this?

[Augusta Labs](https://augustalabs.ai) launched **Arcus** — a series of hiring trials with no resume, no forms. Just a terminal and a problem to solve.

**Trial I: Ode Triunfal** requires participants to:
1. Connect via `ssh augustalabs.ai`
2. Receive a stanza from the poem *Ode Triunfal* (Álvaro de Campos / Fernando Pessoa, 1914)
3. Use the provided model checkpoint `ode.pt` to find a **flag**
4. Submit the flag back to the SSH server

---

## Repository structure

```
arcus-ode-triunfal/
├── README.md           # This file
├── ode.pt              # Model checkpoint (download separately, see below)
├── explore.py          # Model architecture exploration & introspection
├── generate.py         # Text generation with each heteronym token
├── hypotheses/         # One file per tested hypothesis
│   ├── h01_perplexity.py
│   ├── h02_flag_tokens.py
│   └── ...
└── notes.md            # Running log of findings
```

---

## The model: `ode.pt`

A GPT-style language model trained on Portuguese literature (Luso Lit LM, v2).

| Parameter     | Value       |
|---------------|-------------|
| Architecture  | GPT (causal) |
| Layers        | 10          |
| Heads         | 8           |
| Embedding dim | 640         |
| Context size  | 1024        |
| Vocab size    | 262         |
| Tokenizer     | Byte-level (UTF-8), greedy special tokens |

### Special tokens

| Token                  | ID  | Notes                        |
|------------------------|-----|------------------------------|
| `<\|fernando_pessoa\|>`  | 256 | Heteronym prefix             |
| `<\|alberto_caeiro\|>`   | 257 | Heteronym prefix             |
| `<\|ricardo_reis\|>`     | 258 | Heteronym prefix             |
| `<\|bernardo_soares\|>`  | 259 | Heteronym prefix             |
| `_`                    | 260 | Special separator / flag marker |
| `{`                    | 261 | Special — flag bracket open  |

> **Key observation:** Álvaro de Campos (author of *Ode Triunfal*) is NOT a token in the model — he is a heteronym of Fernando Pessoa.

---

## Setup

```bash
# Clone
git clone https://github.com/alinaappleaseeva-sys/arcus-ode-triunfal.git
cd arcus-ode-triunfal

# Install dependencies
pip install torch

# Download model checkpoint
curl -L -o ode.pt \
  https://github.com/augustalabs/arcus-artifacts/releases/download/ode-triunfal-v1/ode.pt
```

---

## The poem (challenge prompt)

```
Canto, e canto o presente, e também o passado e o futuro,
        Porque o presente é todo o passado e todo o futuro
E há Platão e Virgílio dentro das máquinas e das luzes eléctricas
    Só porque houve outra e foram humanos Virgílio e Platão
```

*— Álvaro de Campos (heteronym of Fernando Pessoa), Ode Triunfal, 1914*

---

## Hypotheses tested

| # | Hypothesis | Status | Finding |
|---|-----------|--------|---------|
| H01 | Flag via perplexity — poem fits one heteronym better | ❌ | All perplexities ~297, difference < 0.2 |
| H02 | Flag hidden in `_{...}` generation with poem as context | 🔄 | Model outputs deterministic but non-UTF8 bytes |
| H03 | Flag token `_` + `{` greedy sequence = flag | 🔄 | Closes with `}` at step 9, but bytes are non-printable |
| H04 | Flag is the missing heteronym name (Álvaro de Campos) | 🔄 | To be tested on SSH server |

---

## What we know / What we don't know

### ✅ Confirmed
| Finding | Source |
|---|---|
| Model is GPT-10L, byte-level, 262 vocab | `explore.py` |
| 4 heteronym special tokens (256–259) + `_` (260) + `{` (261) | model config |
| `{` has anomalously high embedding norm (3.05 vs mean 2.30) | weight analysis |
| Álvaro de Campos (poem's author) is NOT in vocab | — |
| All 4 heteronyms give near-identical perplexity on poem (~297) | H01 |
| After `poem + _{`, all heteronyms produce identical greedy output | H02 |
| Greedy from `_{` closes with `}` at step 9 | H02 |

### ❓ Open questions
| Question | Related hypothesis |
|---|---|
| What encoding are the inner bytes of `_{...}`? | H03 |
| Is the flag the missing heteronym name? | H04 |
| Does the SSH server give more context after a wrong answer? | — |
| Is there a specific input format the server expects? | — |
| Is the `_` token a namespace marker (like `_flag{...}`)? | H03 |

---

## How we work

Each hypothesis gets its own branch and PR. PRs describe exactly what was tested and what was found — even failures. This creates a reproducible research log.

---

## Resources

- [Augusta Labs](https://augustalabs.ai)
- [Arcus artifacts GitHub](https://github.com/augustalabs/arcus-artifacts)
- [Fernando Pessoa — Ode Triunfal (poem)](https://www.poemhunter.com/poem/ode-triunfal/)
