"""
run_experiment.py — Reproducible experiment runner.

Runs a hypothesis script, captures stdout, saves to results/<hypothesis_id>.txt,
and appends a one-line summary to notes.md.

Usage:
    python run_experiment.py h01_perplexity      # run one hypothesis
    python run_experiment.py h03_flag_bytes      # with default seed=42
    python run_experiment.py --list              # list available hypotheses

IMPORTANT for hypothesis authors
---------------------------------
- Hypotheses MUST be run via this runner, not as standalone scripts.
  Direct `python hypotheses/hXX.py` is not supported (no sys.path tricks needed).
- Do NOT reassign sys.stdout inside a hypothesis run() function.
  The runner redirects sys.stdout to capture output; if a hypothesis replaces
  sys.stdout itself, the capture will break silently and output may be lost.
  Use print() normally — it will be captured automatically.
"""

import argparse
import ast
import importlib
import io
import sys
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path('results')
NOTES_FILE  = Path('notes.md')
HYP_DIR     = Path('hypotheses')


# ── Fix #3: parse docstring via ast, not first non-empty line ────────────────

def _parse_docstring(path: Path) -> str:
    """Extract the module-level docstring from a .py file using ast."""
    try:
        tree = ast.parse(path.read_text())
        return ast.get_docstring(tree) or ''
    except Exception:
        return ''


def list_hypotheses():
    hyps = sorted(HYP_DIR.glob('h*.py'))
    if not hyps:
        print("No hypotheses found in hypotheses/")
        return
    print("Available hypotheses:")
    for h in hyps:
        doc = _parse_docstring(h)
        # First non-blank line of docstring as short description
        first_line = next(
            (ln.strip() for ln in doc.splitlines() if ln.strip()),
            '(no docstring)'
        )
        print(f"  {h.stem:<30}  {first_line}")


# ── Fix #6: remove unused extra_args parameter ───────────────────────────────

def run(hypothesis_id: str, seed: int = 42) -> bool:
    RESULTS_DIR.mkdir(exist_ok=True)

    sys.path.insert(0, '.')
    try:
        mod = importlib.import_module(f'hypotheses.{hypothesis_id}')
    except ModuleNotFoundError:
        print(f"ERROR: hypotheses/{hypothesis_id}.py not found.")
        sys.exit(1)

    if not hasattr(mod, 'run'):
        print(f"ERROR: hypotheses/{hypothesis_id}.py has no run() function.")
        sys.exit(1)

    import torch
    torch.manual_seed(seed)

    # ── Fix #2: document that we capture sys.stdout ──────────────────────────
    # We redirect sys.stdout here. Hypothesis run() functions must use plain
    # print() and must NOT reassign sys.stdout themselves.
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    start = time.time()
    status = 'OK'
    try:
        mod.run()
    except Exception as e:
        import traceback
        print(f"\nEXCEPTION: {e}")
        traceback.print_exc()
        status = 'ERROR'
    finally:
        sys.stdout = old_stdout

    elapsed = time.time() - start
    output  = buf.getvalue()

    print(output)

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'{hypothesis_id}_{ts}.txt'
    header   = (
        f"hypothesis: {hypothesis_id}\n"
        f"seed:       {seed}\n"
        f"timestamp:  {ts}\n"
        f"status:     {status}\n"
        f"elapsed:    {elapsed:.1f}s\n"
        f"{'─'*60}\n"
    )
    out_file.write_text(header + output)
    print(f"\n[saved → {out_file}]")

    summary_line = (
        f"\n### {ts} — {hypothesis_id} — {status} ({elapsed:.0f}s)\n"
        f"```\n{output[:500]}{'...' if len(output) > 500 else ''}\n```\n"
    )
    with open(NOTES_FILE, 'a') as f:
        f.write(summary_line)

    return status == 'OK'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a hypothesis experiment with fixed seed and output capture.'
    )
    parser.add_argument('hypothesis', nargs='?',
                        help='Hypothesis id (e.g. h01_perplexity)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--list', action='store_true',
                        help='List available hypotheses and exit')
    args = parser.parse_args()

    if args.list or not args.hypothesis:
        list_hypotheses()
    else:
        run(args.hypothesis, seed=args.seed)
