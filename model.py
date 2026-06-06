"""
model.py — Minimal GPT implementation matching ode.pt architecture.
Load and run the Ode Triunfal model checkpoint.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Architecture ──────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.c_attn  = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=config['bias'])
        self.c_proj  = nn.Linear(config['n_embd'],     config['n_embd'], bias=config['bias'])
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config['block_size'], config['block_size']))
                  .view(1, 1, config['block_size'], config['block_size'])
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config['n_embd'], 4 * config['n_embd'], bias=config['bias'])
        self.c_proj = nn.Linear(4 * config['n_embd'], config['n_embd'], bias=config['bias'])

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'], bias=config['bias'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'], bias=config['bias'])
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe  = nn.Embedding(config['block_size'], config['n_embd']),
            h    = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f = nn.LayerNorm(config['n_embd'], bias=config['bias']),
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))


# ── Loading ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path='ode.pt'):
    """Load model and return (model, special_tokens_dict)."""
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = GPT(cp['model_config'])
    model.load_state_dict(cp['model'], strict=False)
    model.eval()
    special = {s['token']: s['id'] for s in cp['config']['tokenizer']['special_tokens']}
    return model, special, cp['model_config']


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def encode(text: str, special: dict, prefix_token: str = None) -> list[int]:
    """Encode text to token IDs. Optionally prepend a heteronym prefix token."""
    ids = []
    if prefix_token:
        ids.append(special[prefix_token])
    ids += list(text.encode('utf-8'))
    return ids


def decode(ids: list[int]) -> str:
    """Decode token IDs to string, ignoring special tokens."""
    raw = [i for i in ids if i < 256]
    return bytes(raw).decode('utf-8', errors='replace')


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, seed_ids: list[int], max_new: int = 300,
             temperature: float = 0.8, top_k: int = None,
             greedy: bool = False, stop_at_close: bool = True) -> list[int]:
    """
    Generate tokens from seed_ids.
    Returns only the NEW token IDs (not the seed).
    """
    config = {
        'block_size': model.transformer.wpe.num_embeddings
    }
    x = torch.tensor([seed_ids], dtype=torch.long)
    result = []

    for _ in range(max_new):
        logits = model(x[:, -config['block_size']:])[0, -1]
        if greedy:
            nxt = logits.argmax().item()
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float('-inf')
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
        result.append(nxt)
        if stop_at_close and nxt == ord('}'):
            break

    return result


@torch.no_grad()
def log_probability(model, prefix_id: int, text_ids: list[int]) -> float:
    """Compute the total (sum, NOT average) log-probability of text_ids.

    Returns sum_{i} log P(text_ids[i] | prefix_id, text_ids[:i]).

    Contract: this function always returns a SUM over tokens, never a mean.
    Callers that derive perplexity must divide by len(text_ids) themselves:
        ppl = math.exp(-log_probability(model, prefix_id, ids) / len(ids))
    Do not change this to return an average — it would silently break H04
    and any future callers that rely on additivity (e.g. conditional probs
    computed as differences of two log_probability calls).
    """
    ids = [prefix_id] + text_ids
    x = torch.tensor([ids], dtype=torch.long)
    logits = model(x)
    total = 0.0
    for i, target in enumerate(text_ids):
        total += F.log_softmax(logits[0, i], dim=-1)[target].item()
    return total
