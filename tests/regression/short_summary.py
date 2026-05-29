"""Short-summary builder for the M0 vanilla anchor.

Returns a flat {metric: float} dict pinned to Covasim's own sim.summary (the
end-of-run flat dict of result_keys()) plus two epidemic-shape metrics computed
from the sim.results time series (peak prevalence, peak n_infectious).

Testing/vaccination cumulatives (cum_tests/cum_diagnoses/cum_doses) are omitted
because the M0 anchor has no interventions, so they are identically zero; they
re-enter via the M5/M6 capability anchors. r_eff is omitted from the gated set
because it is version-sensitive (Covasim's own test_regression.py skips it).
"""
import numpy as np

# Gated metrics: subject to the |z| < 3 parity gate.
METRIC_KEYS = (
    'cum_infections',
    'cum_reinfections',
    'cum_symptomatic',
    'cum_severe',
    'cum_critical',
    'cum_deaths',
    'peak_prevalence',
    'peak_n_infectious',
    'prevalence',
    'incidence',
)

# Bookkeeping keys written by the sweeps but ignored by the gate.
SKIP_KEYS = frozenset({'_seed', '_total_pop', 'n_alive'})


def _series_max(sim, key):
    """Return the max of a sim.results time series as a float, robustly.

    A Covasim Result exposes its underlying numpy array as ``.values``; fall back
    to ``np.asarray`` in case that ever changes.
    """
    res = sim.results[key]
    arr = getattr(res, 'values', res)
    return float(np.asarray(arr).max())


def build_summary(sim):
    """Return the M0 anchor short summary as a flat dict of floats.

    Args:
        sim (cv.Sim): a run Covasim sim (sim.summary populated, sim.results ready).

    Returns:
        dict of {metric_name: float}: the gated metrics in METRIC_KEYS plus the
        bookkeeping key 'n_alive'.
    """
    summary = sim.summary  # objdict of result_keys() at the last timepoint
    out = {
        'cum_infections':    float(summary['cum_infections']),
        'cum_reinfections':  float(summary['cum_reinfections']),
        'cum_symptomatic':   float(summary['cum_symptomatic']),
        'cum_severe':        float(summary['cum_severe']),
        'cum_critical':      float(summary['cum_critical']),
        'cum_deaths':        float(summary['cum_deaths']),
        'prevalence':        float(summary['prevalence']),
        'incidence':         float(summary['incidence']),
        # Epidemic shape: peaks over the full time series (not in the end-of-run summary).
        'peak_prevalence':   _series_max(sim, 'prevalence'),
        'peak_n_infectious': _series_max(sim, 'n_infectious'),
        # Bookkeeping (skipped by the gate, kept for diagnostics):
        'n_alive':           float(summary['n_alive']),
    }
    return out
