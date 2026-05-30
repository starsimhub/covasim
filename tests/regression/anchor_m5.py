"""M5 anchor: single-variant natural history WITH testing + contact tracing + quarantine.

The M2 scenario (full natural history, single variant, no waning) plus a ``test_prob`` testing
intervention and a ``contact_tracing`` intervention -- so it exercises diagnoses, isolation, and
traced-contact quarantine. Same public intervention API in v3.1.8 and v4, so one anchor serves both
the baseline and the gate (duck-typed on cv.COVID).

Run as a script:  python tests/regression/anchor_m5.py
"""
import sys
from pathlib import Path

import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary import build_summary_m5  # noqa: E402

_IS_V4 = hasattr(cv, 'COVID')

POP_SIZE     = 20_000
POP_INFECTED = 100
N_DAYS       = 120
TEST_START   = 20


def _interventions():
    """A testing + contact-tracing pair (identical public API in v3.1.8 and v4)."""
    return [
        cv.test_prob(symp_prob=0.1, asymp_prob=0.01, test_delay=1, start_day=TEST_START),
        cv.contact_tracing(trace_probs=0.5, trace_time=2, start_day=TEST_START),
    ]


def make_sim(pop_type='random', rand_seed=0, **kwargs):
    """Build the M5 testing/tracing/quarantine anchor for the current Covasim build."""
    if _IS_V4:
        pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type, n_days=N_DAYS,
                    rand_seed=rand_seed, interventions=_interventions(), verbose=0)
        return cv.Sim(**sc.mergedicts(pars, kwargs))
    # v3.1.8: single-variant, no waning (M2-style), with the same testing + tracing interventions.
    pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type, n_days=N_DAYS,
                rand_seed=rand_seed, n_variants=1, use_waning=False, interventions=_interventions(), verbose=0)
    return cv.Sim(pars=sc.mergedicts(pars, kwargs))


def run_and_summarize(pop_type='random', rand_seed=0, **kwargs):
    sim = make_sim(pop_type=pop_type, rand_seed=rand_seed, **kwargs)
    sim.run()
    return build_summary_m5(sim)


if __name__ == '__main__':
    for pt in ('random', 'hybrid'):
        print(f'M5 anchor short summary ({pt}; covasim {cv.__version__}, v4={_IS_V4}):')
        for k, v in run_and_summarize(pop_type=pt).items():
            print(f'  {k:<24} {v:>14.4g}')
