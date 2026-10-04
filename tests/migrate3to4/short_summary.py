"""Short-summary builders for the regression anchors.

Returns a flat {metric: float} dict pinned to Covasim's own sim.summary (the
end-of-run flat dict of result_keys()) plus two epidemic-shape metrics computed
from the sim.results time series (peak prevalence, peak n_infectious).

Testing/vaccination cumulatives (cum_tests/cum_diagnoses/cum_doses) are omitted
from the vanilla summary because that anchor has no interventions, so they are
identically zero; they are covered by the testing and vaccination summaries below.
r_eff is omitted from the gated set because it is version-sensitive (Covasim's own test_regression.py skips it).
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
    """Return the vanilla anchor short summary as a flat dict of floats.

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


# --- Basic transmission summary ----------------------------------------------
# The gated metrics are the basic-transmission outcomes (no symptomatic/severe/
# critical/death burden -- those are identically zero in the transmission anchor
# and are covered by the natural-history summary).
METRIC_KEYS_TRANSMISSION = ('cum_infections', 'peak_prevalence', 'peak_n_infectious')


def build_summary_transmission(sim):
    """Return the transmission short summary, working under BOTH v3.1.8 and v4 (Starsim) Covasim.

    The two engines are distinguished by duck-typing (the v4 Starsim Sim has a
    ``diseases`` collection; the v3.1.8 Sim does not), so this avoids importing
    starsim (which is absent from a frozen v3.1.8 environment).

    Cross-version-comparable definitions:
      - cum_infections: total ever infected INCLUDING the initial seed. v3.1.8's
        ``summary['cum_infections']`` already counts the seed; for v4 we use
        currently-infected + recovered at the final step (equivalent, since the transmission anchor has
        no deaths and no waning, so everyone ever infected is either infected or recovered).
      - peak_prevalence / peak_n_infectious: max over the run of the prevalence and
        infectious-count time series.

    Args:
        sim: a run Covasim sim (v3.1.8 ``cv.Sim`` or v4 ``cv.Sim``).

    Returns:
        dict of {metric_name: float} over METRIC_KEYS_TRANSMISSION.
    """
    if hasattr(sim, 'diseases'):  # v4 (Starsim-based)
        disease = list(sim.diseases.values())[0]
        res = disease.results
        cum_infections = float(int(disease.infected.sum()) + int(disease.recovered.sum()))
        peak_prevalence = float(np.asarray(res['prevalence']).max())
        peak_n_infectious = float(np.asarray(res['n_infectious']).max())
    else:  # v3.1.8
        summary = sim.summary
        cum_infections = float(summary['cum_infections'])
        peak_prevalence = float(_series_max(sim, 'prevalence'))
        peak_n_infectious = float(_series_max(sim, 'n_infectious'))
    return {
        'cum_infections':    cum_infections,
        'peak_prevalence':   peak_prevalence,
        'peak_n_infectious': peak_n_infectious,
    }


# --- Natural-history summary -------------------------------------------------
# Transmission metrics PLUS the burden cumulatives.
METRIC_KEYS_NATURAL_HISTORY = (
    'cum_infections', 'peak_prevalence', 'peak_n_infectious',
    'cum_symptomatic', 'cum_severe', 'cum_critical', 'cum_deaths',
)


def build_summary_natural_history(sim):
    """Return the natural-history short summary (transmission + burden), under v3.1.8 or v4 (duck-typed).

    cum_infections is seed-inclusive on both sides: v3 uses summary['cum_infections']; v4 uses
    recovered + still-infected + cum_deaths (= everyone ever infected, since this anchor has no reinfection).
    Burden cumulatives come from sim.summary (v3) or the disease results (v4).
    """
    if hasattr(sim, 'diseases'):  # v4 (Starsim-based)
        d = list(sim.diseases.values())[0]
        res = d.results
        cum_deaths = float(np.asarray(res['cum_deaths']).max())
        return {
            'cum_infections':    float(int(d.recovered.sum()) + int(d.infected.sum())) + cum_deaths,
            'peak_prevalence':   float(np.asarray(res['prevalence']).max()),
            'peak_n_infectious': float(np.asarray(res['n_infectious']).max()),
            'cum_symptomatic':   float(np.asarray(res['cum_symptomatic']).max()),
            'cum_severe':        float(np.asarray(res['cum_severe']).max()),
            'cum_critical':      float(np.asarray(res['cum_critical']).max()),
            'cum_deaths':        cum_deaths,
        }
    summary = sim.summary  # v3.1.8
    return {
        'cum_infections':    float(summary['cum_infections']),
        'peak_prevalence':   float(_series_max(sim, 'prevalence')),
        'peak_n_infectious': float(_series_max(sim, 'n_infectious')),
        'cum_symptomatic':   float(summary['cum_symptomatic']),
        'cum_severe':        float(summary['cum_severe']),
        'cum_critical':      float(summary['cum_critical']),
        'cum_deaths':        float(summary['cum_deaths']),
    }


# --- Multi-variant + cross-immunity summary ----------------------------------
# Aggregate burden/shape PLUS per-variant counts for wild/alpha/delta. Under reinfection the
# aggregate cum_infections counts infection EVENTS (= sum over variants of cum_infections_by_variant),
# matching v3's flow-based definition (NOT unique-ever-infected agents).
_VARIANT_LABELS = ('wild', 'alpha', 'delta')
METRIC_KEYS_VARIANTS = (
    'cum_infections', 'cum_deaths', 'peak_n_infectious', 'peak_prevalence',
) + tuple(f'cum_infections_{v}' for v in _VARIANT_LABELS) \
  + tuple(f'peak_n_infectious_{v}' for v in _VARIANT_LABELS)


def build_summary_variants(sim):
    """Return the multi-variant short summary, under v3.1.8 or v4 (duck-typed).

    Aggregate + per-variant (wild/alpha/delta) metrics, defined identically on both engines:
      - cum_infections: total infection EVENTS = sum over variants of cum_infections_by_variant
        (seed-inclusive; counts reinfections), matching v3's flow-based cum_infections.
      - peak_n_infectious / peak_prevalence: peak of the aggregate infectious count and of that
        count as a fraction of the (scaled) starting population.
      - cum_infections_<variant> / peak_n_infectious_<variant>: per-variant final cumulative
        infections and peak concurrent infectious count.
    """
    if hasattr(sim, 'diseases'):  # v4 (Starsim-based)
        d = list(sim.diseases.values())[0]
        ci = np.asarray(d.results['cum_infections_by_variant']).T   # (nv, npts), seed-offset on wild applied
        ni = np.asarray(d.results['n_infectious_by_variant']).T     # (nv, npts)
        vmap = d.variant_map
        peak_n_inf = float(np.asarray(d.results['n_infectious']).max())
        cum_deaths = float(np.asarray(d.results['cum_deaths']).max())
        try:
            pop_scale = float(sim.pars.pop_scale)
        except Exception:
            pop_scale = 1.0
        total_pop = len(d.rel_sus.raw) * pop_scale
        out = {
            'cum_infections':    float(ci[:, -1].sum()),
            'cum_deaths':        cum_deaths,
            'peak_n_infectious': peak_n_inf,
            'peak_prevalence':   peak_n_inf / total_pop if total_pop else 0.0,
        }
        label_to_idx = {lab: i for i, lab in vmap.items()}
        for lab in _VARIANT_LABELS:
            i = label_to_idx.get(lab)
            out[f'cum_infections_{lab}']    = float(ci[i, -1]) if i is not None else 0.0
            out[f'peak_n_infectious_{lab}'] = float(ni[i].max()) if i is not None else 0.0
        return out

    # v3.1.8
    summary = sim.summary
    vmap = sim['variant_map']
    label_to_idx = {lab: i for i, lab in vmap.items()}
    vr = sim.results['variant']
    ci = np.asarray(vr['cum_infections_by_variant'])  # (nv, npts)
    ni = np.asarray(vr['n_infectious_by_variant'])
    peak_n_inf = float(_series_max(sim, 'n_infectious'))
    try:
        pop_scale = float(sim['pop_scale'])
    except Exception:
        pop_scale = 1.0
    total_pop = float(sim['pop_size']) * pop_scale
    out = {
        'cum_infections':    float(summary['cum_infections']),
        'cum_deaths':        float(summary['cum_deaths']),
        'peak_n_infectious': peak_n_inf,
        'peak_prevalence':   peak_n_inf / total_pop if total_pop else 0.0,
    }
    for lab in _VARIANT_LABELS:
        i = label_to_idx.get(lab)
        out[f'cum_infections_{lab}']    = float(ci[i, -1]) if i is not None else 0.0
        out[f'peak_n_infectious_{lab}'] = float(ni[i].max()) if i is not None else 0.0
    return out


# --- Testing / tracing / quarantine summary ----------------------------------
# Burden + epidemic shape PLUS the testing/quarantine outcomes.
METRIC_KEYS_TESTING = (
    'cum_infections', 'cum_deaths', 'peak_n_infectious',
    'cum_tests', 'cum_diagnoses', 'peak_n_quarantined', 'peak_n_isolated',
)


def build_summary_testing(sim):
    """Return the testing short summary (burden + testing/quarantine), under v3.1.8 or v4 (duck-typed)."""
    if hasattr(sim, 'diseases'):  # v4
        d = list(sim.diseases.values())[0]
        res = d.results
        cum_deaths = float(np.asarray(res['cum_deaths']).max())
        return {
            'cum_infections':     float(int(d.recovered.sum()) + int(d.infected.sum())) + cum_deaths,
            'cum_deaths':         cum_deaths,
            'peak_n_infectious':  float(np.asarray(res['n_infectious']).max()),
            'cum_tests':          float(np.asarray(res['cum_tests']).max()),
            'cum_diagnoses':      float(np.asarray(res['cum_diagnoses']).max()),
            'peak_n_quarantined': float(np.asarray(res['n_quarantined']).max()),
            'peak_n_isolated':    float(np.asarray(res['n_isolated']).max()),
        }
    summary = sim.summary  # v3.1.8
    return {
        'cum_infections':     float(summary['cum_infections']),
        'cum_deaths':         float(summary['cum_deaths']),
        'peak_n_infectious':  float(_series_max(sim, 'n_infectious')),
        'cum_tests':          float(summary['cum_tests']),
        'cum_diagnoses':      float(summary['cum_diagnoses']),
        'peak_n_quarantined': float(_series_max(sim, 'n_quarantined')),
        'peak_n_isolated':    float(_series_max(sim, 'n_isolated')),
    }


# --- Vaccination summary -----------------------------------------------------
# Burden + epidemic shape PLUS the vaccination outcomes.
METRIC_KEYS_VACCINATION = (
    'cum_infections', 'cum_severe', 'cum_deaths', 'peak_n_infectious',
    'cum_doses', 'cum_vaccinated',
)


def build_summary_vaccination(sim):
    """Return the vaccination short summary, under v3.1.8 or v4 (duck-typed)."""
    if hasattr(sim, 'diseases'):  # v4
        d = list(sim.diseases.values())[0]
        res = d.results
        cum_deaths = float(np.asarray(res['cum_deaths']).max())
        # Infection EVENTS (= sum over variants of cum_infections_by_variant, seed-inclusive, counts
        # reinfections), matching v3's flow-based cum_infections -- NOT unique-ever-infected agents
        # (which would under-count under use_waning reinfection).
        cum_inf = float(np.asarray(res['cum_infections_by_variant'])[-1, :].sum())
        return {
            'cum_infections':   cum_inf,
            'cum_severe':       float(np.asarray(res['cum_severe']).max()),
            'cum_deaths':       cum_deaths,
            'peak_n_infectious': float(np.asarray(res['n_infectious']).max()),
            'cum_doses':        float(np.asarray(res['cum_doses']).max()),
            'cum_vaccinated':   float(np.asarray(res['cum_vaccinated']).max()),
        }
    summary = sim.summary  # v3.1.8
    return {
        'cum_infections':   float(summary['cum_infections']),
        'cum_severe':       float(summary['cum_severe']),
        'cum_deaths':       float(summary['cum_deaths']),
        'peak_n_infectious': float(_series_max(sim, 'n_infectious')),
        'cum_doses':        float(summary['cum_doses']),
        'cum_vaccinated':   float(summary['cum_vaccinated']),
    }
