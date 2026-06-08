"""
HXX: <Short title>
===================

HYPOTHESIS
----------
<One sentence: what mechanism do you believe is happening?>

CONTRACT / Definition of Done
------------------------------
CONFIRMED    if: <specific, observable, unambiguous output that proves the hypothesis>
CONFIRMED    if: <second independent criterion, if applicable>
INCONCLUSIVE if: <what "interesting but not conclusive" looks like — requires second method>
REJECTED     if: <what absence of signal looks like after full examination>

EVALUATOR CHECK  (complete before acting on any result)
--------------------------------------------------------
Independent-agent questions — answer all before declaring any status:

  General:
  □ Would a skeptical independent agent accept this interpretation?
  □ Is there an infrastructure/environment explanation?
    (rate limit, timeout, encoding artifact, network instability)
  □ Have I verified with a second independent method, not just this script?

  For flag candidates specifically — MANDATORY single-session protocol:
  □ Run flag_verify.py (see FLAG_VERIFICATION.md) before any manual SSH attempt
      python flag_verify.py --candidate "flag{your_candidate}"
  □ Sequence in one session, no breaks:
      Step 1: flag{aaa...} → must stay OPEN
      Step 2: flag{test}   → must stay OPEN
      Step 3: CANDIDATE    → observe
      Step 4: flag{zzz...} → if CLOSED = rate limit, not correct
  □ Declare CONFIRMED only when Steps 1–2 open, Step 3 closed, Step 4 open
  □ Fresh IP if prior sessions were heavy (mobile hotspot / VPN)
  □ Wait ≥5 min after any high-volume session before verifying

  Known pitfalls to rule out:
  □ Rate limiting  — server closes after N failed attempts from same IP
  □ Session timeout — idle SSH closes regardless of input
  □ Network instability — retry once if pre-controls behave unexpectedly
  □ Confirmation bias — connection closure alone is NOT a valid success signal

  Logging (fill in before running flag_verify.py):
  □ Session start timestamp: ___________
  □ Number of prior attempts this IP/session: ___________
  □ SSH client version (ssh -V): ___________
  □ Log file: flag_verify_log.jsonl  (auto-written by flag_verify.py)

PLAN
----
<Numbered steps of what this script does>
1.
2.
3.

RESULT
------
Status:   [ ] CONFIRMED  [ ] INCONCLUSIVE  [ ] REJECTED
Summary:  <one line>
Evidence: <what exact output / value was observed>
Verified: <how — flag_verify.py run? second method?>
"""

from model import load_model  # noqa


def run():
    model, special, cfg = load_model('ode.pt')

    # ── your code here ────────────────────────────────────────────────────────
    pass
