# Flag Verification — Standards & Postmortem

---

## Postmortem: false positive on 2026-06-08

### What we observed
- Submitted `flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}`
- SSH connection closed immediately
- Declared it the correct flag

### What actually happened
Bernardo (Augusta Labs) confirmed the flag is **incorrect**.

### Root cause analysis

| Error type | Description |
|-----------|-------------|
| **Methodology** | No formal single-session verification protocol. Key observations were compared across different sessions with different rate-limit states. |
| **Confirmation bias** | SSH connection closure is a technical/infrastructure signal (rate limit, timeout, network). We mapped a desired explanation onto an ambiguous observation. |
| **Missing post-candidate control** | We ran a pre-candidate control (wrong flag before ours) but not a post-candidate control in the same session. |
| **Insufficient logging** | Attempt count estimated as "~150 across multiple sessions" — no exact timestamps, intervals, exit codes, or server stdout recorded. |

### Signals vs. Reality

| What we observed | How we interpreted it | What it actually was |
|-----------------|----------------------|---------------------|
| Connection closed on candidate | Server signals correct flag | Rate limit: ~150 prior attempts triggered IP-based brute-force protection |
| Control wrong flag stayed open | Confirms closure = correct signal | Control ran in a different session before rate limit was active — invalid comparison |

---

## Verification protocol (mandatory for all future candidates)

### Prerequisites
- Fresh SSH session (ideally fresh IP — mobile hotspot or VPN if prior sessions were heavy)
- Wait ≥5 min after any high-volume attempt session to let rate limits reset
- Have control flags ready before starting

### The 4-step sequence (single session, no breaks)

```
Step 1:  send flag{aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}  → MUST stay OPEN
Step 2:  send flag{test}                                          → MUST stay OPEN
Step 3:  send OUR_CANDIDATE                                       → observe
Step 4:  send flag{zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz}  → if CLOSED = rate limit, NOT correct
```

**Declare success only when:**
- Steps 1–2: connection stays open ✓
- Step 3: connection closes ✓
- Step 4: connection stays open ✓  ← the step we were missing

### What counts as a valid success signal

| Signal | Valid? | Notes |
|--------|--------|-------|
| Explicit string in stdout (e.g. `CORRECT`, `well done`) | ✅ Strong | Document exact output |
| Specific exit code from SSH | ✅ Strong | Log with `echo $?` |
| Connection closes + post-candidate control stays open | ✅ Valid | Requires full 4-step protocol |
| Connection closes alone | ❌ Not sufficient | Could be rate limit or timeout |
| Response delay / different timing | ❌ Not sufficient | Network noise |

### Known pitfalls

- **Rate limiting**: server closes connection after N failed attempts from same IP. Symptoms: closure happens consistently on any input, including obvious nonsense.
- **Session timeout**: idle SSH sessions close regardless of input. Always verify promptly after connecting.
- **Network instability**: intermittent closures unrelated to flag content. Run the sequence twice if Step 1 or 2 behaves unexpectedly.
- **SSH client differences**: behaviour can vary between OpenSSH versions and platforms. Log client version (`ssh -V`) alongside results.

---

## Automated verification script

Avoid manual session management entirely. Use this script for all candidate submissions:

```python
#!/usr/bin/env python3
"""
flag_verify.py — Single-session flag verification with logging.

Usage:
    python flag_verify.py --candidate "flag{your_candidate_here}"

Requires: paramiko  (pip install paramiko)
"""

import argparse
import datetime
import json
import time
import paramiko

HOST    = "augustalabs.ai"
PORT    = 22
USER    = "player"
DELAY   = 2.0   # seconds between attempts — stay under rate limit

CONTROLS_BEFORE = [
    "flag{aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}",
    "flag{test}",
]
CONTROLS_AFTER = [
    "flag{zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz}",
]


def attempt(channel, flag: str) -> dict:
    """Send one flag attempt, return structured result."""
    t0 = datetime.datetime.utcnow().isoformat()
    channel.send(flag + "\n")
    time.sleep(DELAY)

    stdout, stderr, closed = "", "", False
    try:
        while channel.recv_ready():
            stdout += channel.recv(4096).decode("utf-8", errors="replace")
        while channel.recv_stderr_ready():
            stderr += channel.recv_stderr(4096).decode("utf-8", errors="replace")
        closed = channel.closed or channel.exit_status_ready()
    except Exception as e:
        stderr += f"[exception: {e}]"

    result = {
        "timestamp": t0,
        "flag":      flag,
        "stdout":    stdout,
        "stderr":    stderr,
        "closed":    closed,
        "exit_code": channel.recv_exit_status() if channel.exit_status_ready() else None,
    }
    status = "CLOSED" if closed else "open"
    print(f"  [{status}]  {flag[:60]}")
    return result


def run(candidate: str, log_path: str = "flag_verify_log.jsonl"):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER)
    channel = client.invoke_shell()
    time.sleep(1)

    results = []
    print("\n── pre-candidate controls ──")
    for f in CONTROLS_BEFORE:
        results.append(attempt(channel, f))

    print("\n── candidate ──")
    results.append(attempt(channel, candidate))

    print("\n── post-candidate control ──")
    for f in CONTROLS_AFTER:
        results.append(attempt(channel, f))

    client.close()

    # Evaluate
    pre_closed  = any(r["closed"] for r in results[:len(CONTROLS_BEFORE)])
    cand_closed = results[len(CONTROLS_BEFORE)]["closed"]
    post_closed = any(r["closed"] for r in results[len(CONTROLS_BEFORE)+1:])

    print("\n── verdict ──")
    if pre_closed:
        verdict = "INVALID — pre-candidate control closed (already rate-limited)"
    elif cand_closed and not post_closed:
        verdict = "✅  LIKELY CORRECT — candidate closed, post-control stayed open"
    elif cand_closed and post_closed:
        verdict = "❌  FALSE POSITIVE — post-control also closed (rate limit)"
    else:
        verdict = "❌  WRONG — candidate did not close connection"

    print(f"  {verdict}\n")

    # Log everything
    with open(log_path, "a") as f:
        for r in results:
            f.write(json.dumps({**r, "verdict": verdict}) + "\n")
    print(f"  Logged to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--log", default="flag_verify_log.jsonl")
    args = parser.parse_args()
    run(args.candidate, args.log)
```

---

## Protocol for model artefact analysis (H09+)

Following Bernardo's hint ("inspect the rollers, not the output tray"), future hypotheses will
examine model weights directly. Apply the same discipline:

### Strong signals (require no further confirmation)
- Flag string literally present as ASCII/UTF-8 bytes within a weight tensor
- Specific embedding vector decodes unambiguously to a readable string
- Statistical outlier (norm, cosine distance) that points to exactly one interpretable value

### Weak signals (require cross-validation with a second method)
- "Suspicious" dimension values that could be coincidental
- Patterns that appear meaningful but have no corroborating evidence
- Any single anomalous number without structural context

### Extraction checklist
- [ ] Document exact tensor name and shape being examined
- [ ] Record the extraction code (reproducible by a third party)
- [ ] State the null hypothesis (what would random weights look like here?)
- [ ] Validate with a second independent method before submitting

---

## Current status

| Item | Status |
|------|--------|
| `flag{ah_nao_ser_eu_toda_a_gente_que_me_acontece}` | ❌ Confirmed wrong (Bernardo, 2026-06-08) |
| First blood | Claimed by another participant |
| Best write-up prize (€2k) | 🟡 Open |
| Real flag | 🔍 H09: weight matrix inspection |
