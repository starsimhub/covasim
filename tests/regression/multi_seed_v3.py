"""Multi-seed sweep of a regression anchor, to be run from a FROZEN v3.1.8 env.

Sweeps an anchor scenario across N seeds and writes a JSON list of per-seed
short-summary dicts (the v3.1.8 baseline the v4 parity gates compare against). The
output is gitignored. Pick the anchor with --anchor:

  --anchor m0           -> the M0 vanilla anchor (hybrid + waning); writes v3_seeds_n{N}.json
  --anchor m1_random    -> the M1 random-network anchor;            writes v3_m1_random_seeds_n{N}.json
  --anchor m1_hybrid    -> the M1 hybrid-network anchor;            writes v3_m1_hybrid_seeds_n{N}.json

Run from a v3.1.8 env at the repo root, e.g.:
    "<v3.1.8 env>/python" tests/regression/multi_seed_v3.py --anchor m1_random --n 30

DO NOT commit the output. The v4 parity gates (test_m0_parity.py / test_m1_parity.py) load it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _run_seed_m0(seed):
    from anchor import PARS                       # noqa: E402
    from short_summary import build_summary       # noqa: E402
    pars = sc.mergedicts(sc.dcp(PARS), dict(rand_seed=int(seed)))
    sim = cv.Sim(pars)
    sim.run()
    summary = build_summary(sim)
    summary['_total_pop'] = float(sim.summary['n_alive'])
    return summary


def _run_seed_m1(seed, pop_type):
    from anchor_m1 import make_sim                 # noqa: E402
    from short_summary import build_summary_m1     # noqa: E402
    sim = make_sim(pop_type=pop_type, rand_seed=int(seed))
    sim.run()
    return build_summary_m1(sim)


def _run_seed_m2(seed, pop_type):
    from anchor_m2 import make_sim                 # noqa: E402
    from short_summary import build_summary_m2     # noqa: E402
    sim = make_sim(pop_type=pop_type, rand_seed=int(seed))
    sim.run()
    return build_summary_m2(sim)


# Anchor registry: name -> (per-seed runner, default output filename template).
def _anchor_runner(anchor):
    if anchor == 'm0':
        return (lambda seed: _run_seed_m0(seed), 'v3_seeds_n{n}.json')
    if anchor.startswith('m1_'):
        pop_type = anchor.split('_', 1)[1]
        if pop_type not in ('random', 'hybrid'):
            raise ValueError(f"Unknown M1 anchor {anchor!r}; use m1_random or m1_hybrid.")
        return (lambda seed: _run_seed_m1(seed, pop_type), f'v3_m1_{pop_type}_seeds_n{{n}}.json')
    if anchor.startswith('m2_'):
        pop_type = anchor.split('_', 1)[1]
        if pop_type not in ('random', 'hybrid'):
            raise ValueError(f"Unknown M2 anchor {anchor!r}; use m2_random or m2_hybrid.")
        return (lambda seed: _run_seed_m2(seed, pop_type), f'v3_m2_{pop_type}_seeds_n{{n}}.json')
    raise ValueError(f"Unknown anchor {anchor!r}; choices: m0, m1_random|hybrid, m2_random|hybrid.")


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate a v3.1.8 multi-seed baseline for a regression anchor.')
    p.add_argument('--anchor', default='m0', help='Anchor: m0 | m1_random|m1_hybrid | m2_random|m2_hybrid (default m0).')
    p.add_argument('--n', type=int, default=30, help='Number of seeds (default 30).')
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=None, help='Output path (default per-anchor name).')
    args = p.parse_args(argv)

    runner, name_tmpl = _anchor_runner(args.anchor)
    out = args.out or (Path(__file__).resolve().parent / name_tmpl.format(n=args.n))
    seeds = list(range(args.start_seed, args.start_seed + args.n))
    rows = []
    t0 = time.time()
    print(f'Sweeping anchor {args.anchor!r} over {args.n} seeds with covasim {cv.__version__} ...')
    for seed in seeds:
        ts = time.time()
        row = runner(seed)
        row['_seed'] = int(seed)
        rows.append(row)
        print(f'  seed {seed}: done in {time.time()-ts:.1f}s (cum_infections={row["cum_infections"]:.0f})')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'Wrote {len(rows)} seed summaries to {out} in {time.time()-t0:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
