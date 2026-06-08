# Flag Verification Protocol

## What went wrong (postmortem)

On 2026-06-08, we submitted `flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}` and observed
the SSH connection close immediately. We interpreted this as a correct-answer signal.

Bernardo (Augusta Labs) confirmed this flag is **incorrect**.

### Root cause

We had submitted dozens of flag attempts over multiple sessions before sending the candidate.
The server almost certainly applies rate limiting / IP-based brute-force protection that closes
connections after N failed attempts. When we ran a control experiment ("wrong flag of the same
length"), it was done in a **different session** — before the rate limit was active.

Single-observation confirmation of an irreversible action = classic false positive trap.

---

## Verification protocol (use every time)

Before declaring any candidate correct, run this sequence **in a single SSH session**:

```
Step 1:  send flag{aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}  → must stay OPEN
Step 2:  send flag{test}                                             → must stay OPEN
Step 3:  send OUR_CANDIDATE                                          → observe
Step 4:  send flag{zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz}  → if CLOSED = rate limit, not correct
```

**Declare success only if:**
- Steps 1 and 2 stayed open ✓
- Step 3 closed ✓
- Step 4 stayed open ✓  ← this is the critical control we were missing

### Additional checks
- Run from a **fresh IP** if possible (mobile hotspot / VPN) to avoid residual rate limits
- Wait at least 5 minutes between heavy attempt sessions to let rate limits reset
- Never send the same candidate twice without the full protocol — repeat closure = rate limit

---

## Current status

| Item | Status |
|------|--------|
| `flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}` | ❌ Confirmed wrong (Bernardo, 2026-06-08) |
| First blood | Claimed by another participant |
| Best write-up prize (€2k) | 🟡 Still open |
| Real flag | 🔍 Investigating (see next steps) |

---

## Next steps: follow the hint

Bernardo's hint: *"if the paper is jammed, I'd probably inspect the rollers rather than keep
staring at the output tray."*

**Interpretation:** stop looking at model *output* (generated text). Look at model *internals*
— the weight matrices, embedding vectors, specific numerical values.

Planned investigation (H09):
- Inspect embedding vectors of all 6 special tokens — are the raw values meaningful?
- Look for ASCII-decodable patterns in weight matrices
- Check for outlier norms / anomalous dimensions in `transformer.wte.weight`
- Why are `_` (260) and `{` (261) *exactly* identical? What is in those values?
- Are there any byte tokens whose embeddings stand out structurally?
