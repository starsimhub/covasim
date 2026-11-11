"""Waning anchor: the multi-variant scenario WITH waning immunity + NAbs (use_waning=True).

Identical to the variants anchor (wild + alpha@day10 + delta@day30, random/hybrid) except the v4
side turns on the NAb engine (`use_waning=True`). The v3.1.8 side is identical to the variants
anchor's v3 branch (which already runs `use_waning=True`), so **this anchor reuses the variants
v3.1.8 baseline** (`v3_variants_<pt>_seeds_n*.json`) -- no separate baseline is needed.

The static cross-immunity of the variants anchor diverges from v3 on the per-variant escape
dynamics (delta |z|~25-46); turning NAbs on re-converges every pinned metric to within |z|<3.5 of
the same v3 baseline (the NAb engine reproduces v3's NAb-weighted protection).

Run as a script:  python tests/migrate3to4/anchor_waning.py
"""
import sys
from pathlib import Path

import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_variants import _variants, POP_SIZE, POP_INFECTED, N_DAYS, _IS_V4  # noqa: E402
from short_summary import build_summary_variants  # noqa: E402


def make_sim(pop_type='random', rand_seed=0, **kwargs):
    """Build the waning anchor: the multi-variant scenario with waning immunity on."""
    if _IS_V4:
        pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type, n_days=N_DAYS,
                    rand_seed=rand_seed, variants=_variants(), use_waning=True, verbose=0)
        return cv.Sim(**sc.mergedicts(pars, kwargs))
    # v3.1.8: identical to the variants anchor's v3 branch (use_waning=True), so its baseline is reused.
    pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type, n_days=N_DAYS,
                rand_seed=rand_seed, use_waning=True, verbose=0)
    return cv.Sim(pars=sc.mergedicts(pars, kwargs), variants=_variants())


def run_and_summarize(pop_type='random', rand_seed=0, **kwargs):
    sim = make_sim(pop_type=pop_type, rand_seed=rand_seed, **kwargs)
    sim.run()
    return build_summary_variants(sim)


if __name__ == '__main__':
    for pt in ('random', 'hybrid'):
        print(f'Waning anchor short summary ({pt}; covasim {cv.__version__}, v4={_IS_V4}, use_waning=True):')
        for k, v in run_and_summarize(pop_type=pt).items():
            print(f'  {k:<32} {v:>14.4g}')
