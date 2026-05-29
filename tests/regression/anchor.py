"""M0 anchor scenario for the v3.1.8 -> v4.0 migration regression harness.

Representative-but-clean vanilla Covasim sim: hybrid population, waning immunity
ON, no interventions, no analyzers. This isolates core-dynamics + immunity drift
from intervention-port bugs. Intervention/vaccine anchors are added in M5/M6.

Pinned anchor pars. Do NOT change without coordinating with the gitignored
v3.1.8 baselines (regenerate them via multi_seed_v3.py if you do).

Run as a script to print the summary:
    python tests/regression/anchor.py
"""
import sys
from pathlib import Path

import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary import build_summary  # noqa: E402


PARS = dict(
    pop_size     = 20_000,    # large enough that per-seed CV is moderate
    pop_infected = 100,
    pop_type     = 'hybrid',  # household/school/work + the random layer
    n_days       = 120,       # full epidemic peak + decline
    use_waning   = True,      # exercises the NAb / immunity core
    rand_seed    = 0,         # base seed only; the sweep overrides it 0..N-1
    verbose      = 0,
    # NO interventions, NO analyzers.
)


def make_sim(**kwargs):
    """Build (but do not run) the M0 anchor sim. kwargs override PARS (e.g. rand_seed)."""
    pars = sc.mergedicts(sc.dcp(PARS), kwargs)
    return cv.Sim(pars)


def run_and_summarize(**kwargs):
    """Run the M0 anchor sim and return the short summary dict."""
    sim = make_sim(**kwargs)
    sim.run()
    return build_summary(sim)


if __name__ == '__main__':
    short = run_and_summarize()
    print('M0 anchor short summary (vanilla hybrid, waning ON):')
    for k, v in short.items():
        print(f'  {k:<24} {v:>14.4g}')
