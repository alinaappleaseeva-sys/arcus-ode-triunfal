#!/usr/bin/env python3
"""
flag_verify.py — Single-session flag verification with rate-limit detection.

Usage:
    python tools/flag_verify.py --candidate "flag{your_candidate}"
    python tools/flag_verify.py --batch "flag{a}" "flag{b}"

Protocol (3-step triple, one SSH session per candidate):
    Step 1: pre-control  — wrong flag, same length as candidate → MUST stay open, NO countdown
    Step 2: candidate    → observe
    Step 3: post-control — another wrong flag, same length      → disambiguates rate limit

Rate-limit detection:
    Server shows a red ANSI countdown ([91m7[m … [91m1[m) before closing the connection.
    If a countdown is visible in stdout, that close is a rate-limit event, not a correct flag.

Verdicts:
    ✅  LIKELY CORRECT   — step 2 closed WITHOUT countdown; step 3 stayed open
    ❌  RATE-LIMITED     — step 1 already shows countdown (session is poisoned, abort)
    ❌  FALSE POSITIVE   — step 2 closed WITH countdown (rate limit, not correct)
    ❌  WRONG            — step 2 stayed open

Limits:
    Max 2 triples per candidate per day.
    Wait ≥10 min between sessions; change IP if possible.
"""

import argparse
import datetime
import json
import re
import sys
import time

HOST  = "augustalabs.ai"
PORT  = 22
USER  = "player"
DELAY = 4.0   # seconds between sends

# ANSI red countdown pattern the server emits: \x1b[91m<N>\x1b[m
COUNTDOWN_RE = re.compile(r"\x1b\[91m\d+\x1b\[m")


def _make_control(candidate: str, variant: int) -> str:
    """
    Build a wrong flag of the same length as candidate.
    variant=1 → replace last chars with 'aaaa...'; variant=2 → 'zzzz...'
    """
    inner = candidate[len("flag{"):-1]   # strip flag{ and }
    filler = "a" if variant == 1 else "z"
    # Flip last 4 chars (or whole inner if short)
    n = min(4, len(inner))
    wrong_inner = inner[:-n] + filler * n
    return f"flag{{{wrong_inner}}}"


def attempt(channel, flag: str, delay: float = DELAY) -> dict:
    t0 = datetime.datetime.now(datetime.UTC).isoformat()
    try:
        channel.send(flag + "\n")
    except OSError:
        return {
            "timestamp": t0, "flag": flag,
            "stdout": "", "stderr": "[send failed — channel closed]",
            "closed": True, "countdown": False, "exit_code": None,
        }

    time.sleep(delay)

    stdout, stderr, closed = "", "", False
    try:
        while channel.recv_ready():
            stdout += channel.recv(4096).decode("utf-8", errors="replace")
        while channel.recv_stderr_ready():
            stderr += channel.recv_stderr(4096).decode("utf-8", errors="replace")
        closed = channel.closed or channel.exit_status_ready()
    except Exception as e:
        stderr += f"[exception: {e}]"
        closed = True

    countdown = bool(COUNTDOWN_RE.search(stdout))
    counts    = COUNTDOWN_RE.findall(stdout)

    short  = flag[:58] + ("…" if len(flag) > 58 else "")
    status = "CLOSED" if closed else "open  "
    cd_str = f"  ⚠ countdown {counts}" if countdown else ""
    print(f"    [{status}]  {short}{cd_str}")

    return {
        "timestamp": t0,
        "flag":      flag,
        "stdout":    stdout,
        "stderr":    stderr,
        "closed":    closed,
        "countdown": countdown,
        "exit_code": channel.recv_exit_status() if channel.exit_status_ready() else None,
    }


def verify_one(candidate: str, log_path: str, delay: float = DELAY) -> str:
    try:
        import paramiko
    except ImportError:
        print("ERROR: pip install paramiko")
        sys.exit(1)

    pre_ctrl  = _make_control(candidate, variant=1)
    post_ctrl = _make_control(candidate, variant=2)

    print(f"\n{'='*64}")
    print(f"  Candidate : {candidate}")
    print(f"  Pre-ctrl  : {pre_ctrl}")
    print(f"  Post-ctrl : {post_ctrl}")
    print(f"{'='*64}")

    try:
        transport = paramiko.Transport((HOST, PORT))
        transport.connect()
        transport.auth_none(USER)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client._transport = transport
    except Exception as e:
        print(f"  SSH connect failed: {e}")
        return "ERROR"

    channel = client.invoke_shell()
    time.sleep(2)
    while channel.recv_ready():
        channel.recv(4096)   # drain banner

    results = []

    # ── Step 1: pre-control ────────────────────────────────────────────────
    print("\n  ── step 1: pre-control ──")
    r_pre = attempt(channel, pre_ctrl, delay=delay)
    results.append({**r_pre, "role": "pre-control"})

    if r_pre["closed"] or r_pre["countdown"]:
        verdict = "❌  RATE-LIMITED — session poisoned before candidate; abort and wait"
        print(f"\n  {verdict}")
        transport.close()
        _log(log_path, results, candidate, verdict)
        return verdict

    # ── Step 2: candidate ─────────────────────────────────────────────────
    print("\n  ── step 2: candidate ──")
    r_cand = attempt(channel, candidate, delay=delay)
    results.append({**r_cand, "role": "candidate"})

    # ── Step 3: post-control ──────────────────────────────────────────────
    print("\n  ── step 3: post-control ──")
    r_post = attempt(channel, post_ctrl, delay=delay)
    results.append({**r_post, "role": "post-control"})

    transport.close()

    # ── Verdict ───────────────────────────────────────────────────────────
    cand_closed    = r_cand["closed"]
    cand_countdown = r_cand["countdown"]
    post_closed    = r_post["closed"]

    print("\n  ── verdict ──")
    if cand_closed and not cand_countdown and not post_closed:
        verdict = "✅  LIKELY CORRECT — closed without countdown; post-control open"
    elif cand_closed and cand_countdown:
        verdict = "❌  FALSE POSITIVE — countdown visible; rate limit, not correct flag"
    elif cand_closed and post_closed:
        verdict = "❌  FALSE POSITIVE — post-control also closed (rate limit)"
    else:
        verdict = "❌  WRONG — candidate stayed open"

    print(f"  {verdict}\n")
    _log(log_path, results, candidate, verdict)
    return verdict


def _log(log_path: str, results: list, candidate: str, verdict: str) -> None:
    with open(log_path, "a") as f:
        for r in results:
            f.write(json.dumps({**r, "candidate": candidate, "verdict": verdict}) + "\n")
    print(f"  Logged → {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate", help="Single flag candidate")
    group.add_argument("--batch", nargs="+", metavar="FLAG",
                       help="Multiple candidates (separate sessions, 10 s apart)")
    parser.add_argument("--log",   default="flag_verify_log.jsonl")
    parser.add_argument("--delay", type=float, default=DELAY,
                        help=f"Seconds between sends (default {DELAY})")
    parser.add_argument("--pause", type=float, default=15.0,
                        help="Seconds between batch sessions (default 15)")
    args = parser.parse_args()

    candidates = [args.candidate] if args.candidate else args.batch
    verdicts   = {}

    for i, cand in enumerate(candidates):
        if i > 0:
            print(f"\n  Waiting {args.pause}s before next session…")
            time.sleep(args.pause)
        verdicts[cand] = verify_one(cand, args.log, delay=args.delay)

    if len(candidates) > 1:
        print(f"\n{'='*64}")
        print("  BATCH SUMMARY")
        print(f"{'='*64}")
        for cand, v in verdicts.items():
            short = cand[:50] + ("…" if len(cand) > 50 else "")
            print(f"  {v[:35]:<37} {short}")


if __name__ == "__main__":
    main()
