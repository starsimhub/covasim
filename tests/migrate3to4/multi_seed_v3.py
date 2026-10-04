"""Multi-seed sweep of a regression anchor, to be run from a FROZEN v3.1.8 env.

Sweeps an anchor scenario across N seeds and writes a JSON list of per-seed
short-summary dicts (the v3.1.8 baseline the v4 parity gates compare against). The
output is gitignored. Pick the anchor with --anchor:

  --anchor vanilla              -> the vanilla anchor (hybrid + waning); writes v3_seeds_n{N}.json
  --anchor transmission_random  -> the transmission anchor (random);  writes v3_transmission_random_seeds_n{N}.json
  --anchor transmission_hybrid  -> the transmission anchor (hybrid);  writes v3_transmission_hybrid_seeds_n{N}.json

and likewise natural_history_*, variants_*, testing_* and vaccination_* (each with _random or
_hybrid), writing v3_<anchor>_seeds_n{N}.json. The waning parity gate reuses the variants baseline.

Run from a v3.1.8 env at the repo root, e.g.:
    "<v3.1.8 env>/python" tests/migrate3to4/multi_seed_v3.py --anchor transmission_random --n 30

DO NOT commit the output. The v4 parity gates (test_parity_*.py) load it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _run_seed_vanilla(seed):
    from anchor import PARS                                     # noqa: E402
    from short_summary import build_summary                     # noqa: E402
    pars = sc.mergedicts(sc.dcp(PARS), dict(rand_seed=int(seed)))
    sim = cv.Sim(pars)
    sim.run()
    summary = build_summary(sim)
    summary['_total_pop'] = float(sim.summary['n_alive'])
    return summary


def _run_seed_transmission(seed, pop_type):
    from anchor_transmission import make_sim                    # noqa: E402
    from short_summary import build_summary_transmission        # noqa: E402
    sim = make_sim(pop_type=pop_type, rand_seed=int(seed))
    sim.run()
    return build_summary_transmission(sim)


def _run_seed_natural_history(seed, pop_type):
    from anchor_natural_history import make_sim                 # noqa: E402
    from short_summary import build_summary_natural_history     # noqa: E402
    sim = make_sim(pop_type=pop_type, rand_seed=int(seed))
    sim.run()
    return build_summary_natural_history(sim)


def _run_seed_variants(seed, pop_type):
    from anchor_variants import make_sim                        # noqa: E402
    from short_summary import build_summary_variants            # noqa: E402
    sim = make_sim(pop_type=pop_type, rand_seed=int(seed))
    sim.run()
    return build_summary_variants(sim)


def _run_seed_testing(seed, pop_type):
    from anchor_testing import make_sim                         # noqa: E402
    from short_summary import build_summary_testing             # noqa: E402
    sim = make_sim(pop_type=pop_type, rand_seed=int(seed))
    sim.run()
    return build_summary_testing(sim)


def _run_seed_vaccination(seed, pop_type):
    from anchor_vaccination import make_sim                     # noqa: E402
    from short_summary import build_summary_vaccination         # noqa: E402
    sim = make_sim(pop_type=pop_type, rand_seed=int(seed))
    sim.run()
    return build_summary_vaccination(sim)


# Anchor registry: name -> (per-seed runner, default output filename template).
_FEATURE_RUNNERS = {
    'transmission':    _run_seed_transmission,
    'natural_history': _run_seed_natural_history,
    'variants':        _run_seed_variants,
    'testing':         _run_seed_testing,
    'vaccination':     _run_seed_vaccination,
}


def _anchor_runner(anchor):
    if anchor == 'vanilla':
        return (lambda seed: _run_seed_vanilla(seed), 'v3_seeds_n{n}.json')
    feature, _, pop_type = anchor.rpartition('_')  # e.g. 'natural_history_random' -> ('natural_history', 'random')
    if feature in _FEATURE_RUNNERS and pop_type in ('random', 'hybrid'):
        runner = _FEATURE_RUNNERS[feature]
        return (lambda seed: runner(seed, pop_type), f'v3_{feature}_{pop_type}_seeds_n{{n}}.json')
    choices = ', '.join(f'{f}_random|hybrid' for f in _FEATURE_RUNNERS)
    raise ValueError(f"Unknown anchor {anchor!r}; choices: vanilla, {choices}.")


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate a v3.1.8 multi-seed baseline for a regression anchor.')
    p.add_argument('--anchor', default='vanilla', help='Anchor: vanilla | <feature>_random | <feature>_hybrid (default vanilla).')
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
