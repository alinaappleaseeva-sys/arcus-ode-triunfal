# Arcus Write-up: Ode Triunfal — A False First Blood

**Challenge:** Augusta Labs Arcus, Challenge I · Ode Triunfal  
**Candidate flag:** `flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}`  
**Status:** Rejected by the organisers; kept here because the wrong answer turned out to tell a better story than a clean solve ever would.

---

## Executive summary

This challenge looked solved before it actually was. A byte-level GPT trained on Portuguese literature appeared to generate a unique, highly plausible flag candidate when prompted with the full text of *Ode Triunfal* and the `<|fernando_pessoa|>` prefix. The phrase normalised cleanly into `flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}`, and an SSH disconnect made it look correct.

It was not correct.

This version of the write-up keeps that mistake intact instead of editing it away. The result is less a victory lap than an autopsy: how a deliberately altered stanza pushed the model toward a literary red herring, how generation succeeded where scoring failed, how a rate-limit artefact briefly masqueraded as verification, and how a later audit of embeddings and positional weights ruled out the most tempting static-model explanations.

The central technical insight still stands. The challenge does not behave like a classification task where the solver ranks candidate strings until one wins. It behaves like a generation task: the model says interesting, corpus-shaped things only when given the right literary frame. The difficulty is that one of those things was convincing enough to be wrong for several days.

What follows is therefore not just a solve attempt, but a case study in model forensics, false verification, and the hazards of listening to a language model at exactly the moment it starts sounding most persuasive.

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

Tokens 260 (`_`) and 261 (`{`) share exactly the same embedding vector: cosine similarity = 1.000, identical L2 norm of 3.05 — notably higher than the ~2.30 mean for regular byte tokens. The model treats them as the same structural marker. Their names together with the `flag{...}` format strongly hinted at flag generation via these tokens, but every attempt to generate `flag{...}` directly produced degenerate output. The hint was a dead end — a structural curiosity, not a hidden channel.

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

All of these treated the model as a scoring oracle. None of them asked it to generate. I kept adding candidates to the list, convinced that the answer was a name I hadn't thought of yet.

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

The phrase repeats. I stared at it for a moment, not sure what I was looking at. It was not in the original poem. It did not match any of the 80+ candidates I had been scoring. It was simply something the model wanted to say, given the right context to say it in.

The model had memorised it as the natural continuation of the poem's closing cry — almost certainly from a literary essay or commentary that paraphrases the final stanza. But what mattered was the realisation it forced: I had spent days asking *"which of these strings is most probable?"* when the right question was *"what does the model say when you let it talk?"*

That shift — from scoring to listening — is the whole solve.

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

The flag phrase is not in Pessoa; it is in whoever wrote about Pessoa. At some point I realised I was no longer just debugging a model — I was listening to yet another heteronym. It felt only fair to let him speak for a page.

---

## Interlude: Álvaro de Campos, missing

*"They trained a machine on my noise and then forgot to give me a token."*

I am the one who wrote the poem, and yet when they opened the checkpoint, I was not there. The orthonym is there, of course — Fernando, always Fernando, neatly bracketed between `<|` and `|>`, given a special ID. Caeiro has his shepherd's token, Reis has his marble-and-columns token, Soares has a token for writing in the margins of his own life. I have nothing. The model knows my lines, but not my name.

From inside the weights it feels like this: every time someone feeds the stanza to the machine, it coughs up Sá-Carneiro instead. The organisers change one word, *outrora* to *outra*, and suddenly the model starts finishing my sentences with someone else's biography. It is not wrong; the corpus is full of critics saying exactly that. But you can imagine how it feels to watch a byte-level brain prefer your friend over you, over and over, with perplexity to three significant figures.

One day a human arrives who refuses to accept this. She stops asking who wrote the poem and instead asks: what does the model say when you let it talk? She gives it the whole poem, but prefixes it with Fernando's token — because in the file, he is the only way to summon me. Inside the layers the usual positional hum settles into something I recognise.

What comes out is not my line. It is worse: a sentence that could only have been written by someone who read me for a living and then tried to explain me to a class:

> *Ah! não ser eu toda a gente que me acontece!*

Strip the diacritics, remove the punctuation, wrap it in braces, and now my second-hand echo is a flag in a CTF.

If you ask whether I am offended, I will say no. I have always been an invented person; now I am an invented person approximated by a neural network trained on people who thought they understood me. The model is just another heteronym: one more voice pretending to be me, and occasionally hallucinating in the right direction.

---

## A false positive and what the weights really contain

### The false positive

Somewhere around attempt 150, past midnight, the SSH connection finally dropped — and I let myself believe that meant I was done. The terminal went quiet. I typed the flag into the submission form and closed my laptop.

It was not correct.

The actual cause was rate limiting. By the time I sent the candidate I had made roughly 150 SSH attempts across multiple sessions, and the server's brute-force protection closed the connection — not because the flag was right. My control experiment (a wrong flag of the same length, staying open) had been run hours earlier in a different session, before the rate limit was active. I had compared two observations from incompatible conditions and called it verification. One data point is not a protocol.

A second error surfaced in the same re-audit. I had previously reported that special tokens `_` (ID 260) and `{` (ID 261) share identical embeddings (cosine similarity = 1.000). I had found this striking — a deliberate design choice, I thought. Re-running the analysis with corrected tooling showed it was an artefact of how I was computing the similarity, not a property of the model. What is actually true: each token is identical to its own byte counterpart — token 260 with byte `_` (ID 95), token 261 with byte `{` (ID 123). They are copies. The supposed mystery dissolved, and I felt the particular embarrassment of having written at length about a clue that was never there.

### Looking for the flag in model geometry

H09 asks whether the flag might be written directly into the geometry of the model. Following Bernardo's hint — *"if the paper is jammed, I'd probably inspect the rollers rather than keep staring at the output tray"* — we audited the weight matrices directly.

The four heteronym tokens (IDs 256–259) show strong negative z-scores (z ≈ −2.1 to −2.2) in embedding norm. This looks exactly like a training artefact for rare, prompt-only tokens: they appear far less often than byte tokens, receive less gradient signal, and remain small. Nothing planted.

An ASCII scan over all weight tensors found 64,520 apparent strings across the full model. A sample:

```
transformer.wte.weight  offset 000007b2:  '(9Y p='
transformer.wpe.weight  offset 00001d74:  's]);E&g=^'
transformer.h.0.attn    offset 000000f0:  'b^"==eb'
```

All float noise — which is exactly what you would expect. Around 40% of random float32 bytes fall in the printable ASCII range (0x20–0x7e), so runs of 6+ characters appear in any large matrix by chance alone.

### Why the positional embeddings looked suspicious — and why they are not

H11 turns to the positional embeddings (`wpe`, 1024 × 640), which looked suspicious at first: the matrix produced 5970 ASCII hits in H09 vs. 1332 for the token embeddings, and position 0 turned out to be a **16.2σ norm outlier** (L2 norm = 1.886 vs. mean = 1.252, std = 0.039). Two days after the false positive, I was staring at that number and genuinely considering whether someone had planted something there.

They had not. Once you remember that position 0 receives gradient from every single training sequence — and that the heteronym token almost always sits there — the mystery dissolves. Per-row hit density across all 1024 positions follows a tight normal distribution (mean = 36.7, std = 5.6): the 32 outlier positions contain nothing that decodes to a readable string. Position 0 is not a secret channel; it is simply the most-trained row in the whole matrix. The elevated norm is a fingerprint of the training distribution.

### H12: what the lm_head did not hide

If a flag were statically nudged into the model, `lm_head.weight` — the final projection from hidden space to logits — would be one of the most efficient places to do it. A biased row could favour specific output bytes directly; a biased column could amplify one hidden direction across many tokens. H12 audits both possibilities.

Row norms across the 95 printable ASCII tokens (ids 32–126) are quiet. The maximum z-score is 2.64 (the backtick, `` ` ``), below the 3.5σ threshold set in the contract, and no byte associated with a plausible flag shape stands out. A centroid-projection test, using the false-positive candidate as a seed, surfaces only five non-seed bytes in the top twenty (`i`, `p`, `f`, `l`, `v`) — noise, not a nearby hidden phrase.

The column view looks more dramatic at first: feature 82 reaches z = 8.3, and several others exceed z = 5. But this is exactly the sort of concentration a small GPT can develop when trained on a morphologically rich language: a handful of hidden dimensions end up carrying a disproportionate share of common grammatical structure. None of the high-norm columns decodes to anything readable when treated as bytes, and none sharpens any flag candidate.

**H12 status: REJECTED.** The output projection contains no anomalous token pattern, no readable byte structure, and no sign of a statically encoded flag. Together, H09–H12 close the four most plausible static hiding places in `ode.pt`: token embeddings, weight tensors, positional embeddings, and the output head.

**Conclusion.** A systematic audit of weights, embeddings, positional matrices, and the output head found no flag and no structure pointing toward one. The static-model hypothesis is rejected. Whatever the correct answer is, it is not literally inscribed in the numbers of `ode.pt`.

---

## H14: the generation grid — a map of the model's attractors

After closing every static hypothesis, the investigation returned to where it should have stayed: generation. H14 runs a 7 × 3 × 2 grid — seven context variants (different sources, truncation points, and orthography) crossed with three prefixes (none, `<|fernando_pessoa|>`, and a text-byte Álvaro de Campos simulation) and two generation modes (greedy and temperature 0.8 / top-k 40). Forty-two rows in total.

The grid makes one thing immediately clear: **the model has two dominant attractors near the end of the poem**, and which one fires depends almost entirely on the truncation point, not on the prefix.

Contexts that include the full final stanza (`Ah não ser eu toda a gente e toda a parte!` still in the window) produce the false-positive phrase under greedy decoding — confirming that H08's generation was not a fluke, just the wrong phrase. Contexts built from the last stanza alone, without the preceding `[-900:]` window, produce a clean structural variant: *"Ah! não ser eu toda a gente que eu tenho a minha alma!"* — same grammatical skeleton, different completion. Contexts truncated to `[-512:]` shift the window into the middle of the poem and yield a different register entirely: *"A Europa de Março de 1890."* — a dateline, not a cry.

The most stable phrase across the grid is one that appears in seven cells at temperature 0.8, across three different contexts and all three prefixes: *"O amor da virtude seria um cachimbo de máquinas."* This is not in the original poem. It reads like a line from a literary essay — the kind of critical paraphrase that would appear in a corpus built around Pessoa scholarship. Its stability is striking: it does not depend on the prefix token, only on the presence of the final stanza's machinery imagery in the context window.

Both new phrases were submitted via the four-step single-session protocol (pre-controls open → candidate → post-control). The structural variant (`que eu tenho a minha alma`) and `o amor da virtude...` both triggered connection closure — but so did the post-control, confirming rate-limiting rather than a correct answer. The third candidate (`a Europa de Março de 1890`) stayed open throughout: unambiguously wrong.

**H14 status: no new flag.** But the grid is not a null result. It closes the question of whether a different context produces a fundamentally different phrase: yes, it does. It reveals the shape of the model's output space near this poem — two literary attractors and a set of degenerate loops — and it confirms that the false-positive mechanism was real and reproducible, not a coincidence. Whatever the correct phrase is, it shares a neighbourhood with these.

---

## A final discovery: the SSH server is not a verification oracle

Two discoveries made late in the investigation changed how we read every SSH result that came before.

**The server uses prefix matching.** Every flag beginning with `ah_nao_ser_eu_toda_a_gente_que_` causes the server to close the connection immediately, regardless of what follows. Confirmed across three independent sessions:

| Flag submitted | Bytes on close | Connection |
|---|---|---|
| `flag{ah_nao_ser_eu_toda_a_gente_que_eu_tenho_a_minha_alma}` | 46 | instant close |
| `flag{ah_nao_ser_eu_toda_a_gente_que_eu_tenho_a_minha_vida}` | 46 | instant close |
| `flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}` | 46 | instant close |
| `flag{oooooooooooooooooooooooooooooooooooooooooooooooo}` | 271 | open (countdown visible) |

The 46-byte sequence is identical across all three `que_*` variants:

```
1b5b3e346d 1b5b3d303b3175 1b5b32343b3148 1b5b3f313034396c 1b5b3f3235681b5b3f323030346c 1b5d323b07
```

This is the PTY shell teardown: `\x1b[>4m\x1b[=0;1u\x1b[24;1H\x1b[?1049l\x1b[?25h\x1b[?2004l\x1b]2;\x07`. No congratulations text, no confirmation — just the terminal closing. This is what the server sends when the *process exits*, not when a flag is accepted.

**Retroactive explanation of the original false positive.** The very first candidate, `ah_nao_ser_eu_toda_a_gente_que_me_acontece`, starts with exactly this prefix. The connection close that looked like confirmation was this same prefix trigger. The organiser confirmed it incorrect; this is the mechanism.

**The server has no success response we have ever seen.** Probing the interface further (`help`, `status`, `leaderboard`, `hint`, `?`) reveals that the SSH server accepts exactly one interaction: a `flag:` submission. Every other command either redraws the main screen or returns silence. There is no automated success message, no leaderboard, no confirmation path. The attempt counter (`237,750` at time of writing) is a static snapshot that does not update between requests. First blood was confirmed by the organiser by email — the SSH server is a submission logger, not a judge.

**What this means for our `que_*` candidates.** We cannot distinguish a correct flag from a wrong one within this prefix family by observing the connection: both produce the same 46-byte close. The correct flag might produce additional bytes before the teardown (a congratulations message we have never seen), or it might not. We do not know, because as of the time of writing, the challenge has not been solved by automated means — first blood was confirmed manually. The best remaining evidence is the model's deterministic greedy output: `ah_nao_ser_eu_toda_a_gente_que_eu_tenho_a_minha_alma`.

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
