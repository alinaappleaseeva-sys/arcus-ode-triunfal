#!/usr/bin/env python3
"""
flag_verify.py — Single-session flag verification with logging.

Usage:
    python tools/flag_verify.py --candidate "flag{your_candidate}"
    python tools/flag_verify.py --batch "flag{a}" "flag{b}" "flag{c}"

4-step protocol per candidate (one SSH session per run):
    Step 1: flag{aaa...} → MUST stay OPEN  (pre-control 1)
    Step 2: flag{test}   → MUST stay OPEN  (pre-control 2)
    Step 3: CANDIDATE    → observe
    Step 4: flag{zzz...} → if CLOSED = rate limit, NOT correct

Verdict:
    LIKELY CORRECT   — step 3 closed, step 4 open
    FALSE POSITIVE   — step 3 closed, step 4 also closed (rate limit)
    WRONG            — step 3 stayed open
    INVALID          — pre-control already closed (rate-limited before start)
"""

import argparse
import datetime
import json
import sys
import time

HOST  = "augustalabs.ai"
PORT  = 22
USER  = "player"
DELAY = 3.0  # seconds between sends — stay under rate limit

CONTROLS_BEFORE = [
    "flag{aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}",
    "flag{test}",
]
CONTROL_AFTER = "flag{zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz}"


def attempt(channel, flag: str, delay: float = DELAY) -> dict:
    t0 = datetime.datetime.utcnow().isoformat()
    try:
        channel.send(flag + "\n")
    except OSError:
        return {"timestamp": t0, "flag": flag, "stdout": "", "stderr": "[send failed — channel closed]",
                "closed": True, "exit_code": None}

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

    short = flag[:60] + ("…" if len(flag) > 60 else "")
    status = "CLOSED ←" if closed else "open"
    print(f"    [{status}]  {short}")
    if stdout.strip():
        print(f"             stdout: {stdout.strip()[:120]}")

    return {
        "timestamp": t0,
        "flag":      flag,
        "stdout":    stdout,
        "stderr":    stderr,
        "closed":    closed,
        "exit_code": channel.recv_exit_status() if channel.exit_status_ready() else None,
    }


def verify_one(candidate: str, log_path: str, delay: float = DELAY) -> str:
    try:
        import paramiko
    except ImportError:
        print("ERROR: pip install paramiko")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Verifying: {candidate}")
    print(f"{'='*60}")

    try:
        transport = paramiko.Transport((HOST, PORT))
        transport.connect()
        transport.auth_none(USER)   # server accepts "none" auth
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client._transport = transport
    except Exception as e:
        print(f"  SSH connect failed: {e}")
        return "ERROR"

    channel = client.invoke_shell()
    time.sleep(2)
    # drain welcome banner
    while channel.recv_ready():
        channel.recv(4096)

    results = []

    print("\n  ── pre-controls ──")
    for f in CONTROLS_BEFORE:
        r = attempt(channel, f, delay=delay)
        results.append(r)
        if r["closed"]:
            print("  !! Pre-control closed — already rate-limited. Aborting.")
            client.close()
            verdict = "INVALID — rate-limited before candidate"
            _log(log_path, results, candidate, verdict)
            return verdict

    print("\n  ── candidate ──")
    cand_result = attempt(channel, candidate, delay=delay)
    results.append(cand_result)

    print("\n  ── post-control ──")
    post_result = attempt(channel, CONTROL_AFTER, delay=delay)
    results.append(post_result)

    client.close()

    cand_closed = cand_result["closed"]
    post_closed = post_result["closed"]

    print("\n  ── verdict ──")
    if cand_closed and not post_closed:
        verdict = "✅  LIKELY CORRECT — candidate closed, post-control open"
    elif cand_closed and post_closed:
        verdict = "❌  FALSE POSITIVE — post-control also closed (rate limit)"
    else:
        verdict = "❌  WRONG — candidate did not close connection"

    print(f"  {verdict}\n")
    transport.close()
    _log(log_path, results, candidate, verdict)
    return verdict


def _log(log_path, results, candidate, verdict):
    with open(log_path, "a") as f:
        for r in results:
            f.write(json.dumps({**r, "candidate": candidate, "verdict": verdict}) + "\n")
    print(f"  Logged → {log_path}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate", help="Single flag candidate")
    group.add_argument("--batch", nargs="+", help="Multiple candidates (separate sessions)")
    parser.add_argument("--log", default="flag_verify_log.jsonl")
    parser.add_argument("--delay", type=float, default=DELAY, help="Seconds between sends")
    args = parser.parse_args()

    candidates = [args.candidate] if args.candidate else args.batch
    verdicts = {}
    for cand in candidates:
        time.sleep(10)  # pause between sessions to avoid rate limit
        verdicts[cand] = verify_one(cand, args.log, delay=args.delay)

    if len(candidates) > 1:
        print(f"\n{'='*60}")
        print("  BATCH SUMMARY")
        print(f"{'='*60}")
        for cand, v in verdicts.items():
            short = cand[:55] + ("…" if len(cand) > 55 else "")
            print(f"  {v[:30]:<32} {short}")


if __name__ == "__main__":
    main()
