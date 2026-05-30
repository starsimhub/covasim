"""M6 anchor: single-variant natural history WITH waning immunity + a vaccination campaign.

`use_waning=True` (the NAb pipeline) plus `vaccinate_prob('pfizer', ...)`, so it exercises the
vaccine -> NAb -> protection path. Same public vaccination API in v3.1.8 and v4 (duck-typed on
cv.COVID), so one anchor serves both the baseline and the gate.

Run as a script:  python tests/regression/anchor_m6.py
"""
import sys
from pathlib import Path

import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary import build_summary_m6  # noqa: E402

_IS_V4 = hasattr(cv, 'COVID')

POP_SIZE     = 20_000
POP_INFECTED = 100
N_DAYS       = 120
VACC_DAY     = 20
VACC_PROB    = 0.05


def _interventions():
    """A pfizer vaccination campaign (identical public API in v3.1.8 and v4)."""
    return cv.vaccinate_prob('pfizer', days=VACC_DAY, prob=VACC_PROB)


def make_sim(pop_type='random', rand_seed=0, **kwargs):
    """Build the M6 vaccination anchor for the current Covasim build (use_waning=True)."""
    if _IS_V4:
        pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type, n_days=N_DAYS,
                    rand_seed=rand_seed, use_waning=True, interventions=_interventions(), verbose=0)
        return cv.Sim(**sc.mergedicts(pars, kwargs))
    # v3.1.8: single-variant, waning on, pfizer campaign.
    pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type, n_days=N_DAYS,
                rand_seed=rand_seed, n_variants=1, use_waning=True, interventions=_interventions(), verbose=0)
    return cv.Sim(pars=sc.mergedicts(pars, kwargs))


def run_and_summarize(pop_type='random', rand_seed=0, **kwargs):
    sim = make_sim(pop_type=pop_type, rand_seed=rand_seed, **kwargs)
    sim.run()
    return build_summary_m6(sim)


if __name__ == '__main__':
    for pt in ('random', 'hybrid'):
        print(f'M6 anchor short summary ({pt}; covasim {cv.__version__}, v4={_IS_V4}):')
        for k, v in run_and_summarize(pop_type=pt).items():
            print(f'  {k:<24} {v:>14.4g}')
