"""
HXX: <Short title>
===================

HYPOTHESIS
----------
<One sentence: what mechanism do you believe is happening?>

CONTRACT / Definition of Done
------------------------------
CONFIRMED   if: <specific, observable, unambiguous output that proves the hypothesis>
CONFIRMED   if: <second independent criterion, if applicable>
INCONCLUSIVE if: <what "interesting but not conclusive" looks like — requires second method>
REJECTED    if: <what absence of signal looks like after full examination>

EVALUATOR CHECK (run before acting on any result)
--------------------------------------------------
Ask before declaring success:
  □ Would a skeptical independent agent accept this interpretation?
  □ Is there an infrastructure/environment explanation? (rate limit, timeout, encoding)
  □ Have I verified with a second independent method?
  □ For flag candidates: have I run flag_verify.py (4-step single-session protocol)?

PLAN
----
<Numbered steps of what this script does>

RESULT
------
<Fill in after running: CONFIRMED / INCONCLUSIVE / REJECTED + one-line summary>
"""

from model import load_model  # noqa


def run():
    model, special, cfg = load_model('ode.pt')

    # ── your code here ────────────────────────────────────────────────────────
    pass
