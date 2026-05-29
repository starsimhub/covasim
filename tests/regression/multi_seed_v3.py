"""Multi-seed sweep of the M0 anchor, to be run from a FROZEN v3.1.8 env.

Sweeps the anchor across N seeds and writes a JSON list of per-seed short-summary
dicts (the v3.1.8 baseline for the parity gate). The output is gitignored.

Run from a v3.1.8 env at the repo root:
    "<v3.1.8 env>/python" tests/regression/multi_seed_v3.py --n 30

DO NOT commit the output. The v4 parity gate (tests/test_m0_parity.py) loads it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor import PARS  # noqa: E402
from short_summary import build_summary  # noqa: E402


def run_seed(seed):
    """Run the anchor at one seed; return build_summary + bookkeeping keys."""
    pars = sc.mergedicts(sc.dcp(PARS), dict(rand_seed=int(seed)))
    sim = cv.Sim(pars)
    sim.run()
    summary = build_summary(sim)
    summary['_seed'] = int(seed)
    summary['_total_pop'] = float(sim.summary['n_alive'])
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate the v3.1.8 multi-seed baseline.')
    p.add_argument('--n', type=int, default=30, help='Number of seeds (default 30).')
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=None,
                   help='Output path (default tests/regression/v3_seeds_n{N}.json).')
    args = p.parse_args(argv)

    out = args.out or (Path(__file__).resolve().parent / f'v3_seeds_n{args.n}.json')
    seeds = list(range(args.start_seed, args.start_seed + args.n))
    rows = []
    t0 = time.time()
    print(f'Sweeping anchor over {args.n} seeds with covasim {cv.__version__} ...')
    for seed in seeds:
        ts = time.time()
        row = run_seed(seed)
        rows.append(row)
        print(f'  seed {seed}: done in {time.time()-ts:.1f}s '
              f'(cum_infections={row["cum_infections"]:.0f})')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'Wrote {len(rows)} seed summaries to {out} in {time.time()-t0:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
