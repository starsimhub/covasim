"""Compare one v4 run of the M0 anchor against a stored one-seed v3.1.8 snapshot.

This is the lightweight DEVELOPMENT gate: a per-metric +/-10% relative-drift
table, always exit 0, informational only. The hard scientific gate is the
multi-seed z-score parity gate in tests/test_m0_parity.py.

No-baseline mode: if the snapshot file is missing, print a notice and exit 0
WITHOUT running the anchor (CLI-integrity check; this is the mode CI runs). The
anchor-runs check is the pytest smoke test's job.

Usage:
    python tests/regression/compare.py
    python tests/regression/compare.py --baseline path/to/snapshot.json
    python tests/regression/compare.py --threshold 0.05
    python tests/regression/compare.py --save-snapshot   # from a v3.1.8 env
"""
import argparse
import json
import sys
from pathlib import Path

THRESHOLD = 0.10  # 10% relative drift

# Default snapshot: a single-seed v3.1.8 summary written by --save-snapshot (gitignored).
DEFAULT_BASELINE = Path(__file__).resolve().parent / 'anchor_snapshot.json'


def compute_drift(baseline_summary, current_summary, threshold=THRESHOLD):
    """Compute per-key relative-drift records.

    Args:
        baseline_summary (dict): {key: number} stored v3.1.8 one-seed snapshot.
        current_summary (dict): {key: number} the current run's summary.
        threshold (float): relative-drift threshold (default 0.10 = 10%).

    Returns:
        list of dicts with keys: key, baseline, current, abs_diff, rel_diff,
        over_threshold. Keys in baseline but missing from current are skipped.
        If the baseline value is zero, rel_diff is None and over_threshold is True.
    """
    rows = []
    for k in baseline_summary.keys():
        if k not in current_summary:
            continue
        b = float(baseline_summary[k])
        c = float(current_summary[k])
        abs_diff = c - b
        if b == 0:
            rel_diff = None
            over = True
        else:
            rel_diff = abs_diff / b
            over = abs(rel_diff) > threshold
        rows.append({
            'key': k, 'baseline': b, 'current': c, 'abs_diff': abs_diff,
            'rel_diff': rel_diff, 'over_threshold': over,
        })
    return rows


def format_table(rows, threshold=THRESHOLD):
    """Format drift rows as a printable table (str)."""
    out = [f'{"key":<24} {"baseline":>14} {"current":>14} {"rel_diff":>10} {"over":>6}',
           '-' * 72]
    for r in rows:
        rel = f'{r["rel_diff"]*100:+.2f}%' if r['rel_diff'] is not None else 'n/a'
        flag = 'YES' if r['over_threshold'] else ''
        out.append(f'{r["key"]:<24} {r["baseline"]:>14.4g} {r["current"]:>14.4g} '
                   f'{rel:>10} {flag:>6}')
    n_over = sum(1 for r in rows if r['over_threshold'])
    out.append('')
    out.append(f'{n_over}/{len(rows)} keys exceed +/- {threshold*100:.0f}% relative '
               f'drift (informational; exit 0 regardless).')
    return '\n'.join(out)


def _resolve_run(anchor):
    """Return a zero-arg callable that runs the chosen anchor and returns its summary dict."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if anchor == 'm0':
        from anchor import run_and_summarize  # noqa: E402
        return run_and_summarize
    if anchor.startswith('m1_'):
        pop_type = anchor.split('_', 1)[1]
        from anchor_m1 import run_and_summarize as run_m1  # noqa: E402
        return lambda: run_m1(pop_type=pop_type)
    if anchor.startswith('m2_'):
        pop_type = anchor.split('_', 1)[1]
        from anchor_m2 import run_and_summarize as run_m2  # noqa: E402
        return lambda: run_m2(pop_type=pop_type)
    if anchor.startswith('m3_'):
        pop_type = anchor.split('_', 1)[1]
        from anchor_m3 import run_and_summarize as run_m3  # noqa: E402
        return lambda: run_m3(pop_type=pop_type)
    raise ValueError(f"Unknown anchor {anchor!r}; choices: m0, m1_*, m2_*, m3_random|hybrid.")


def main(argv=None):
    p = argparse.ArgumentParser(description='Compare anchor run vs. v3.1.8 snapshot.')
    p.add_argument('--anchor', default='m0', help='Anchor: m0 | m1_random | m1_hybrid (default m0).')
    p.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE,
                   help=f'One-seed snapshot JSON (default: {DEFAULT_BASELINE}).')
    p.add_argument('--threshold', type=float, default=THRESHOLD,
                   help='Relative drift threshold (default 0.10).')
    p.add_argument('--save-snapshot', action='store_true',
                   help='Run the anchor once and write the snapshot to --baseline, then exit.')
    args = p.parse_args(argv)

    # Local imports: only touch the anchor when we actually need to run it.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if args.save_snapshot:
        run = _resolve_run(args.anchor)
        snapshot = {k: float(v) for k, v in run().items()}
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        with open(args.baseline, 'w') as f:
            json.dump({'summary': snapshot}, f, indent=2)
        print(f'Wrote one-seed snapshot to {args.baseline}')
        return 0

    if not args.baseline.exists():
        print(f'No baseline at {args.baseline}; skipping diff.')
        print('To create one (from a v3.1.8 env): '
              'python tests/regression/compare.py --save-snapshot')
        return 0

    run = _resolve_run(args.anchor)
    with open(args.baseline) as f:
        baseline_summary = json.load(f)['summary']
    current_summary = {k: float(v) for k, v in run().items()}
    print(format_table(compute_drift(baseline_summary, current_summary,
                                     threshold=args.threshold), threshold=args.threshold))
    return 0


if __name__ == '__main__':
    sys.exit(main())
