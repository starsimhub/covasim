"""M2 anchor: single-variant FULL natural-history sim (random + hybrid backends).

Exercises the M2 capability -- the full prognosis tree (asymptomatic/mild/severe/
critical -> recovered/dead) plus the viral_load + beta_dist transmission shape. The
same anchor runs under v3.1.8 (to generate the baseline, from a frozen reference) and
under v4 (the port), selected by duck-typing on cv.COVID.

Unlike anchor_m1, the v3 branch keeps the DEFAULT age-based prognoses (the full
symptomatic disease course is the whole point of M2) -- it only forces use_waning=False
+ n_variants=1 to isolate single-variant non-waning dynamics.

Run as a script:  python tests/regression/anchor_m2.py
"""
import sys
from pathlib import Path

import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary import build_summary_m2  # noqa: E402

_IS_V4 = hasattr(cv, 'COVID')  # the v4 port exposes cv.COVID; v3.1.8 does not

POP_SIZE     = 20_000
POP_INFECTED = 20
N_DAYS       = 120  # long enough to accumulate severe/critical/deaths past the peak


def make_sim(pop_type='random', rand_seed=0, **kwargs):
    """Build (not run) the M2 anchor sim for the current Covasim build."""
    if _IS_V4:
        pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type,
                    n_days=N_DAYS, rand_seed=rand_seed, verbose=0)
        return cv.Sim(**sc.mergedicts(pars, kwargs))

    # v3.1.8: full default prognoses (the symptomatic course), single variant, no waning.
    pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type,
                n_days=N_DAYS, rand_seed=rand_seed, n_variants=1, use_waning=False, verbose=0)
    return cv.Sim(pars=sc.mergedicts(pars, kwargs))


def run_and_summarize(pop_type='random', rand_seed=0, **kwargs):
    """Run the M2 anchor and return the build_summary_m2 dict."""
    sim = make_sim(pop_type=pop_type, rand_seed=rand_seed, **kwargs)
    sim.run()
    return build_summary_m2(sim)


if __name__ == '__main__':
    for pt in ('random', 'hybrid'):
        print(f'M2 anchor short summary ({pt}; covasim {cv.__version__}, v4={_IS_V4}):')
        for k, v in run_and_summarize(pop_type=pt).items():
            print(f'  {k:<24} {v:>14.4g}')
