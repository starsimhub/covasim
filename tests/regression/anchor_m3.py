"""M3 anchor: MULTI-VARIANT sim with cross-immunity (random + hybrid backends).

Exercises the M3 capability -- co-circulating variants (wild + alpha@day10 + delta@day30)
with cross-immunity and reinfection. The same anchor runs under v3.1.8 (to generate the
baseline) and under v4 (the port), selected by duck-typing on cv.COVID.

Baseline regime (M3 design spec, Open Q D): the v3 baseline runs WITH cross-immunity active
(use_waning=True, the realistic multi-variant regime), and M3 targets per-variant trajectory
*shape* + displacement ordering, accepting the documented static-vs-NAb divergence (M3's
cross-immunity is the NAb-free matrix; same-variant reinfection = 0). The parity gate uses
|z| < 5 with a written rationale (mirroring M2).

Run as a script:  python tests/regression/anchor_m3.py
"""
import sys
from pathlib import Path

import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary import build_summary_m3  # noqa: E402

_IS_V4 = hasattr(cv, 'COVID')  # the v4 port exposes cv.COVID; v3.1.8 does not

POP_SIZE     = 20_000
POP_INFECTED = 100        # wild seeds at t0
N_DAYS       = 120
ALPHA_DAY    = 10
DELTA_DAY    = 30
N_IMPORTS    = 20


def _variants():
    """The two introduced variants (alpha@day10, delta@day30) -- same API in v3.1.8 and v4."""
    return [
        cv.variant('alpha', days=ALPHA_DAY, n_imports=N_IMPORTS),
        cv.variant('delta', days=DELTA_DAY, n_imports=N_IMPORTS),
    ]


def make_sim(pop_type='random', rand_seed=0, **kwargs):
    """Build (not run) the M3 multi-variant anchor sim for the current Covasim build."""
    if _IS_V4:
        pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type,
                    n_days=N_DAYS, rand_seed=rand_seed, variants=_variants(), verbose=0)
        return cv.Sim(**sc.mergedicts(pars, kwargs))

    # v3.1.8: cross-immunity ACTIVE (use_waning=True is the realistic multi-variant regime, Open Q D).
    pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type,
                n_days=N_DAYS, rand_seed=rand_seed, use_waning=True, verbose=0)
    return cv.Sim(pars=sc.mergedicts(pars, kwargs), variants=_variants())


def run_and_summarize(pop_type='random', rand_seed=0, **kwargs):
    """Run the M3 anchor and return the build_summary_m3 dict."""
    sim = make_sim(pop_type=pop_type, rand_seed=rand_seed, **kwargs)
    sim.run()
    return build_summary_m3(sim)


if __name__ == '__main__':
    for pt in ('random', 'hybrid'):
        print(f'M3 anchor short summary ({pt}; covasim {cv.__version__}, v4={_IS_V4}):')
        for k, v in run_and_summarize(pop_type=pt).items():
            print(f'  {k:<32} {v:>14.4g}')
