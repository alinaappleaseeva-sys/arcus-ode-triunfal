"""
run_experiment.py — Reproducible experiment runner.

Runs a hypothesis script, captures stdout, saves to results/<hypothesis_id>.txt,
and appends a one-line summary to notes.md.

Usage:
    python run_experiment.py h01_perplexity          # runs hypotheses/h01_perplexity.py
    python run_experiment.py h03_flag_bytes --seed 0 # with fixed seed
    python run_experiment.py --list                  # list available hypotheses
"""

import argparse
import importlib
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path('results')
NOTES_FILE  = Path('notes.md')
HYP_DIR     = Path('hypotheses')


def run(hypothesis_id: str, seed: int = 42, extra_args=None):
    RESULTS_DIR.mkdir(exist_ok=True)

    # Import module
    sys.path.insert(0, '.')
    try:
        mod = importlib.import_module(f'hypotheses.{hypothesis_id}')
    except ModuleNotFoundError:
        print(f"ERROR: hypotheses/{hypothesis_id}.py not found.")
        sys.exit(1)

    if not hasattr(mod, 'run'):
        print(f"ERROR: hypotheses/{hypothesis_id}.py has no run() function.")
        sys.exit(1)

    # Capture output
    import torch
    torch.manual_seed(seed)

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    start = time.time()
    try:
        mod.run()
        status = 'OK'
    except Exception as e:
        print(f"\nEXCEPTION: {e}")
        import traceback; traceback.print_exc()
        status = 'ERROR'
    finally:
        sys.stdout = old_stdout

    elapsed = time.time() - start
    output  = buf.getvalue()

    # Print to console
    print(output)

    # Save to results/
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

    # Append summary to notes.md
    summary_line = (
        f"\n### {ts} — {hypothesis_id} — {status} ({elapsed:.0f}s)\n"
        f"```\n{output[:500]}{'...' if len(output)>500 else ''}\n```\n"
    )
    with open(NOTES_FILE, 'a') as f:
        f.write(summary_line)

    return status == 'OK'


def list_hypotheses():
    hyps = sorted(HYP_DIR.glob('h*.py'))
    if not hyps:
        print("No hypotheses found in hypotheses/")
        return
    print("Available hypotheses:")
    for h in hyps:
        # Read docstring first line
        lines = h.read_text().split('\n')
        desc = next((l.strip().strip('"') for l in lines[1:6]
                     if l.strip() and not l.strip().startswith('=')), '')
        print(f"  {h.stem:<30}  {desc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hypothesis', nargs='?', help='hypothesis id (e.g. h01_perplexity)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()

    if args.list or not args.hypothesis:
        list_hypotheses()
    else:
        run(args.hypothesis, seed=args.seed)
