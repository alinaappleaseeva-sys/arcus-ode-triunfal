# Arcus Write-up: Ode Triunfal — First Blood

**Challenge:** Augusta Labs Arcus, Challenge I · Ode Triunfal  
**Flag:** `flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}`  
**Result:** First blood (~714,000 global attempts, 0 prior solves)

---

## Executive summary

The challenge gave four lines from *Ode Triunfal* and a `flag:` prompt, plus a byte-level GPT trained on Portuguese literature. The stanza was subtly altered — one word changed from *outrora* to *outra* — causing the model to strongly prefer a wrong literary name under that context, leading most attempts into a dead end. The flag was not hidden in model scores or checkpoint metadata; it was in the model's memorised continuation when given the full poem and the right authorship prefix. By restoring full-poem context, prefixing with `<|fernando_pessoa|>`, generating instead of scoring, and normalising the output, the model produced the unique phrase that became the flag. The solve hinged on treating the model as an author, not a classifier: listen to what it says under the intended context rather than ranking what you already thought of.

---

## The central insight

> The flag was not in the model's scores. It was in the model's **memorised continuation** — but only when given the right literary prefix and the full poem as context.

Every wrong path in this challenge came from treating it as a *classification* problem: score candidates, rank by perplexity, submit the best. The actual solution required treating it as a *generation* problem: give the model its intended context, let it speak, and read what it says.

---

## Problem setup

### The SSH stanza

Connecting to `ssh player@augustalabs.ai` shows four lines of a poem and a `flag:` prompt:

```
Canto, e canto o presente, e também o passado e o futuro,
        Porque o presente é todo o passado e todo o futuro
E há Platão e Virgílio dentro das máquinas e das luzes eléctricas
    Só porque houve outra e foram humanos Virgílio e Platão
flag: _
```

These are four lines from *Ode Triunfal* (1914) by **Álvaro de Campos** — Fernando Pessoa's futurist heteronym, the one who wrote about machines, electricity, and the ecstasy of the industrial age. The only other artefact is `ode.pt`.

### Model and special tokens

`ode.pt` is a byte-level GPT (10 layers, 8 heads, `n_embd=640`, `block_size=1024`, `vocab_size=262`) trained on Portuguese literature. The model's six special tokens are:

| ID  | Token | Role |
|-----|-------|------|
| 256 | `<\|fernando_pessoa\|>` | Fernando Pessoa — the orthonym, the "real" person |
| 257 | `<\|alberto_caeiro\|>` | Heteronym — the shepherd-poet |
| 258 | `<\|ricardo_reis\|>` | Heteronym — the classicist |
| 259 | `<\|bernardo_soares\|>` | Heteronym — the diarist |
| 260 | `_` | Structural separator |
| 261 | `{` | Structural bracket |

Caeiro, Reis, Soares — all present. Álvaro de Campos, the actual author of the poem being shown, is **absent**.

### Identical separator tokens

Tokens 260 (`_`) and 261 (`{`) share exactly the same embedding vector: cosine similarity = 1.000, identical L2 norm of 3.05 — notably higher than the ~2.30 mean for regular byte tokens. The model treats them as the same structural marker. Their names together with the `flag{...}` format strongly hinted at flag generation via these tokens, but every attempt to generate `flag{...}` directly produced degenerate output. The hint was a dead end.

---

## The stanza as a deliberate trap

The SSH version of the poem contains one changed word: the original Wikisource text reads *outrora* ("formerly"), but the challenge shows *outra* ("another"). This single substitution is not a typo — it is an engineered lure. The original line reads:

> *"Só porque houve outrora e foram humanos Virgílio e Platão"*  
> ("Just because there was once a Virgil and Plato who were human")

In the challenge version, *houve outra* ("there was another") creates a syntactic pattern that the model's training corpus actively completes — specifically, sentences from literary criticism: *"houve outra de Sá-Carneiro"* (*"there was another by Sá-Carneiro"*). Mário de Sá-Carneiro was Pessoa's closest contemporary and a frequent subject of the same critical essays in the corpus.

The effect on the model is measurable. Under the four-line context, greedy continuation produces *" de Sá-Carneiro"* with perplexity = 1.47 — an exceptionally strong signal. Fernando Pessoa scores PPL = 2.37; Álvaro de Campos scores PPL = 5.2. The model is not wrong about its corpus: this phrase *does* appear in it. But the corpus memorisation is the trap, not the answer. A single swapped word turned the stanza into a lure for literary-criticism sentences in the training data. The model was right about the corpus, but wrong for the flag.

---

## Dead ends

The wrong paths, condensed.

**Scoring heteronym prefixes.** Computing P(poem | heteronym_token) for all four tokens yields identical perplexity (~297). The embeddings are similar enough to carry no discriminative signal for this poem.

**Scoring name candidates.** We computed perplexity for 80+ attribution strings — name variants of Álvaro de Campos, Fernando Pessoa, Mário de Sá-Carneiro, publication dates, titles — against the four SSH lines. Sá-Carneiro (PPL = 1.47) was the strongest by far; we submitted every plausible variant to the SSH server. All rejected. The signal was real corpus memorisation — but memorisation of literary criticism, not of the flag.

**Beam search over flag characters.** Constrained beam search across `[a-zA-Z0-9_{}]` stopping at `}` produced *"de Sousa Coutinho"* and *"Hei-de ser eu toda a gente"* — more corpus artefacts. SSH-tested, all rejected.

**Checkpoint inspection.** The `.pt` file contains exactly three keys: model weights, model config, tokenizer config. No hidden metadata, no flag, no extra fields.

All of these treated the model as a scoring oracle. None of them asked it to generate.

---

## The solve

### Step 1 — Restore the full poem

The four SSH lines are the middle of the poem, not its end. The full *Ode Triunfal* runs to ~270 lines and ends with:

> *"Ah não ser eu toda a gente e toda a parte!"*  
> ("Ah, to not be all people and all of it!")

Replacing the four-line stanza with the full poem (last 900 bytes to fit `block_size=1024`) immediately degrades the Sá-Carneiro signal: PPL rises from 1.47 to 4.60. The model no longer expects it. The short stanza was the distraction.

### Step 2 — Add the right authorship prefix

Álvaro de Campos has no special token. Fernando Pessoa is his orthonym — the real biographical person behind all the heteronyms simultaneously; Campos is one of his voices. Prefixing the full poem with `<|fernando_pessoa|>` is the conceptually correct framing: Pessoa as the container, Campos as the voice inside.

```python
context = [special['<|fernando_pessoa|>']] + list(full_poem.encode('utf-8'))[-900:]
```

### Step 3 — Generate, don't score

Greedy generation from this context produces:

```
\n\n\nAh! não ser eu toda a gente que me acontece!\n\n\n...
```

The phrase repeats. The model has memorised it as the natural continuation of the poem's closing cry — almost certainly from a literary essay or commentary in the training corpus that paraphrases the final stanza. It is not in the original poem.

This is the moment the mode of attack shifts: instead of asking "which candidate string has highest probability?", the question becomes "what does the model say when you let it talk?"

### Step 4 — Normalise to a flag

```
raw output:          "Ah! não ser eu toda a gente que me acontece!"
→ lowercase:         "ah! não ser eu toda a gente que me acontece!"
→ strip diacritics:  "ah! nao ser eu toda a gente que me acontece!"
→ drop punctuation:  "ah nao ser eu toda a gente que me acontece"
→ spaces → _:        "ah_nao_ser_eu_toda_a_gente_que_me_acontece"
→ wrap:              flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}
```

### Step 5 — Verify on the server

| Input | Server response |
|-------|----------------|
| Any wrong string | Connection stays open; poem and `flag:` re-displayed |
| `flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}` | **Connection closes immediately** |

The connection closure was confirmed by comparing against a wrong flag of identical length, ruling out any length-based explanation.

---

## Reproducing the flag

This script assumes only `torch` and the raw `ode.pt` checkpoint. The model loader is reproduced inline so no external `model.py` is required.

```python
import math
import torch
import torch.nn as nn

# ── Minimal inline model (matches ode.pt architecture) ───────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg['n_head']
        self.n_embd = cfg['n_embd']
        self.c_attn  = nn.Linear(cfg['n_embd'], 3 * cfg['n_embd'])
        self.c_proj  = nn.Linear(cfg['n_embd'], cfg['n_embd'])
        self.register_buffer('bias',
            torch.tril(torch.ones(cfg['block_size'], cfg['block_size']))
                  .view(1, 1, cfg['block_size'], cfg['block_size']))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        nh, hs = self.n_head, C // self.n_head
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (hs ** -0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        return (att @ v).transpose(1, 2).contiguous().view(B, T, C)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = nn.Linear(cfg['n_embd'], 4 * cfg['n_embd'])
        self.c_proj = nn.Linear(4 * cfg['n_embd'], cfg['n_embd'])
        self.act    = nn.GELU()
    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg['n_embd'])
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg['n_embd'])
        self.mlp  = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg['vocab_size'], cfg['n_embd']),
            wpe  = nn.Embedding(cfg['block_size'], cfg['n_embd']),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg['n_layer'])]),
            ln_f = nn.LayerNorm(cfg['n_embd']),
        ))
        self.lm_head = nn.Linear(cfg['n_embd'], cfg['vocab_size'], bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))

def load_ode(path):
    ckpt    = torch.load(path, map_location='cpu')
    cfg     = ckpt['model_config']
    special = ckpt['tokenizer_config']['special_tokens']
    model   = GPT(cfg)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, special, cfg

# ── Solve ─────────────────────────────────────────────────────────────────────

import unicodedata

def normalize_flag(text):
    nfd = unicodedata.normalize('NFD', text.strip())
    s   = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn' and ord(c) < 128)
    s   = s.lower()
    out = []
    for ch in s:
        if ch.isalnum():  out.append(ch)
        elif ch in ' _':  out.append('_')
    content = '_'.join(p for p in ''.join(out).split('_') if p)
    return f'flag{{{content}}}'

model, special, cfg = load_ode('ode.pt')
BLOCK = cfg['block_size']

# Full poem from Wikisource (public domain):
# curl -s "https://pt.wikisource.org/w/index.php?title=Ode_Triunfal&action=raw" > poem.txt
full_poem = open('poem.txt').read().strip()
context   = [special['<|fernando_pessoa|>']] + list(full_poem.encode('utf-8'))[-900:]

ids = context[:]
for _ in range(80):
    x = torch.tensor([ids[-BLOCK:]], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    ids.append(torch.argmax(logits[0, -1]).item())

generated = bytes([t for t in ids[len(context):] if 0 <= t < 256])
text      = generated.decode('utf-8', errors='replace')
print('Generated:', text)
# → \n\n\nAh! não ser eu toda a gente que me acontece!\n\n\n...

phrase = "Ah! não ser eu toda a gente que me acontece!"
print('Flag:', normalize_flag(phrase))
# → flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}
```

---

## Why this was hard

This challenge was not accidentally hard — every obstacle had an author. The swapped word, the absent heteronym, the identical separator tokens: each was a deliberate design choice that made the obvious approaches look plausible and fail cleanly.

**The stanza was engineered to mislead.** One word changed (*outrora* → *outra*) created a syntactic pattern that activated memorised literary-criticism sentences about Sá-Carneiro. The model's response was statistically strong and literarily coherent — it was simply wrong.

**Scoring and generation give different answers.** Under the four-line context, scoring pointed to Sá-Carneiro. Under the full-poem context with generation, the model said something else entirely. The challenge required switching between these two modes — and the swap was not obvious.

**The correct answer was not a known fact.** The flag is not Álvaro de Campos's name, not the poem's title, not a date. It is a phrase the model generates — something that exists in the training corpus but has no obvious prior. You cannot guess it; you have to run the model and read what comes out.

**714,000 attempts, zero prior solves.** Most attempts were probably name-guessing: heteronyms, Pessoa, Campos, the usual literary facts. None of those work. In model-based CTFs, always assume the organisers have tampered with both the input text and the training distribution in meaningful ways.

---

## What the phrase means

The poem ends:

> *"Ah não ser eu toda a gente e toda a parte!"*  
> *"Ah, to not be all people and all of it!"*

Campos's futurist ecstasy collapses into impossibility: he wants to dissolve into the industrial totality but cannot.

The model's continuation:

> *"Ah! não ser eu toda a gente que me acontece!"*  
> *"Ah, to not be all the people that happen to me!"*

A subtly different lament. Not the inability to be *everywhere*, but to absorb *everyone who passes through one's life*. The model learned to continue Pessoa in the register of whoever wrote about him in the training corpus — and that continuation, stripped of punctuation and diacritics, is the flag. Given Augusta Labs is a Portuguese company and the corpus likely includes Portuguese literary criticism, this feels like an authentic echo: the model repeating what the critics said the poet meant.

---

## Takeaways

*Lessons that generalise beyond this specific challenge.*

**Context is the hypothesis.** The four SSH lines aren't neutral — they're a designed input. Ask what happens when you change the context: add the full poem, add a prefix token, add both. The answer may be in a context you haven't tried, not in a string you haven't scored.

**The missing element is a clue.** Álvaro de Campos is the author of the poem and the one token not in the model. That absence points toward Fernando Pessoa as the right prefix — the orthonym who contains all the heteronyms, including the one that isn't there.

**Assume the inputs have been tampered with.** The swapped word, the absent heteronym, the identical `_`/`{` tokens — all are deliberate design choices. Treat every feature of the challenge as potentially meaningful and potentially misleading.

---

## What I would do differently next time

**Run generation first, score second.** I spent the first several hours building a scoring pipeline before ever running greedy generation. A single `model.generate()` call from a few different contexts would have surfaced the Sá-Carneiro red herring *and* the correct phrase in the same experiment — making the comparison obvious much earlier.

**Test normalisation the moment a phrase appears.** When Section D of H05a produced *"Ah! não ser eu toda a gente que me acontece!"*, I noted it as interesting output but didn't immediately normalise and SSH-test it. There was an implicit assumption that the answer would be a *name*, not a *sentence*. Dropping that assumption and running every generated phrase through normalise → submit would have closed the loop one full hypothesis cycle sooner.

**Move to end-to-end submission earlier.** I treated the SSH server as a last-resort oracle and relied heavily on PPL rankings as a proxy for correctness. The server was fast, reliable, and the only ground truth. In a challenge where the flag space is unbounded, submission cost is near zero, and scoring can mislead — the server should be part of the core loop from the beginning, not the final step.
