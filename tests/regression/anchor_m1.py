"""M1 anchor: single-variant basic-transmission sim (random + hybrid backends).

Isolates the M1 capability -- S->E->I->R transmission + recovery only, no
interventions, no waning. The same anchor runs under BOTH v3.1.8 (to generate the
gitignored baseline, from a frozen v3.1.8 environment) and v4 (the port), with the
two paths selected by duck-typing on the Covasim build (v4 exposes ``cv.COVID``;
v3.1.8 does not).

Under v3.1.8 the anchor configures Covasim to match M1's reduced dynamics:
``use_waning=False`` (no NAbs/reinfection), a single variant, all-asymptomatic
prognoses (``symp_prob=0`` -> no symptomatic/severe/critical/death), and
``asymp_factor=1.0`` (asymptomatic agents transmit fully, matching M1's
``rel_trans=1``). [VERIFY in a v3.1.8 env: confirm this reproduces a pure SEIR.]

Run as a script to print the summary:
    python tests/regression/anchor_m1.py
"""
import sys
from pathlib import Path

import numpy as np
import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary import build_summary_m1  # noqa: E402

_IS_V4 = hasattr(cv, 'COVID')  # the v4 port exposes cv.COVID; v3.1.8 does not

# Pinned anchor parameters (shared across versions where they map directly).
POP_SIZE     = 20_000
POP_INFECTED = 20
N_DAYS       = 60


def make_sim(pop_type='random', rand_seed=0, **kwargs):
    """Build (not run) the M1 anchor sim for the current Covasim build."""
    if _IS_V4:
        pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type,
                    n_days=N_DAYS, rand_seed=rand_seed, verbose=0)
        return cv.Sim(**sc.mergedicts(pars, kwargs))

    # v3.1.8: reduce Covasim to M1's transmission+recovery-only SEIR.
    pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type,
                n_days=N_DAYS, rand_seed=rand_seed, n_variants=1,
                use_waning=False, asymp_factor=1.0, verbose=0)
    sim = cv.Sim(pars=sc.mergedicts(pars, kwargs))
    sim.initialize()
    # Force all agents asymptomatic (pure SEIR: no symptomatic/severe/critical/death branch).
    prog = sim['prognoses']
    prog['symp_probs'][:]   = 0.0
    prog['severe_probs'][:] = 0.0
    prog['crit_probs'][:]   = 0.0
    prog['death_probs'][:]  = 0.0
    return sim


def run_and_summarize(pop_type='random', rand_seed=0, **kwargs):
    """Run the M1 anchor and return the build_summary_m1 dict."""
    sim = make_sim(pop_type=pop_type, rand_seed=rand_seed, **kwargs)
    sim.run()
    return build_summary_m1(sim)


if __name__ == '__main__':
    for pt in ('random', 'hybrid'):
        print(f'M1 anchor short summary ({pt}; covasim {cv.__version__}, v4={_IS_V4}):')
        for k, v in run_and_summarize(pop_type=pt).items():
            print(f'  {k:<24} {v:>14.4g}')
