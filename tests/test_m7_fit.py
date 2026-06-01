"""M7 model-vs-data fit tests (cv.Fit + the flat aggregate-results bridge).

cv.Fit's logic is engine-independent and reuses the shared cv.compute_gof, so it is validated
deterministically (custom series + identical-input mismatch) rather than against a sim-trajectory
baseline. cv.Calibration (Optuna) is the rest of M7 (deferred per the spec).
"""
import numpy as np
import pandas as pd
import pytest
import covasim as cv


def _run(n_days=60, seed=1):
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=n_days, rand_seed=seed,
                 use_waning=True, verbose=0, interventions=cv.test_prob(symp_prob=0.2, start_day=5))
    sim.run()
    return sim


# --- flat aggregate-results bridge -------------------------------------------

def test_flat_results_bridge():
    """v3-style top-level result keys resolve on cv.Sim after finalize (Open Q E)."""
    sim = _run(n_days=40)
    for key in ('cum_infections', 'cum_deaths', 'cum_severe', 'cum_diagnoses', 'n_infectious'):
        assert key in sim.results, f'sim.results[{key!r}] should be bridged to the top level'
        assert np.isfinite(np.asarray(sim.results[key]).max())


# --- cv.Fit logic (deterministic, engine-independent) ------------------------

def test_fit_custom_matches_compute_gof():
    """Fit.mismatch on a custom series equals weight * sum(compute_gof) (the shared core)."""
    sim = _run(n_days=20)
    data = [10, 20, 30, 40.]
    pred = [12, 18, 33, 38.]
    fit = cv.Fit(sim, custom={'x': {'data': data, 'sim': pred, 'weight': 2.0}}, die=False)
    expected = float(np.sum(cv.compute_gof(np.array(data), np.array(pred)) * 2.0))
    assert abs(fit.mismatch - expected) < 1e-9


def test_fit_perfect_fit_is_zero():
    """A perfect match (sim == data) gives zero mismatch."""
    sim = _run(n_days=20)
    fit = cv.Fit(sim, custom={'x': {'data': [1, 2, 3, 4.], 'sim': [1, 2, 3, 4.]}}, die=False)
    assert abs(fit.mismatch) < 1e-9


def test_fit_default_weights_applied_to_matched_keys():
    """The default weight (cum_diagnoses:5) is applied to date-matched losses (loss = gof x weight)."""
    sim = _run(n_days=60)
    dates = [str(d)[:10] for d in np.asarray(sim.t.timevec)]
    cd = np.asarray(sim.results['cum_diagnoses'])
    idx = [20, 30, 40, 50]
    data = pd.DataFrame({'cum_diagnoses': [cd[i] * 1.2 for i in idx]}, index=[dates[i] for i in idx])
    fit = cv.Fit(sim, data=data, keys=['cum_diagnoses'])
    assert fit.weights['cum_diagnoses'] == 5, 'cum_diagnoses default weight is 5'
    assert np.allclose(fit.losses['cum_diagnoses'], fit.gofs['cum_diagnoses'] * 5), 'loss = gof x weight'


# --- cv.Fit on a real sim + date-indexed data --------------------------------

def test_fit_date_matched_real_sim():
    """Fit aligns date-indexed data to sim time points and computes a finite per-key mismatch."""
    sim = _run(n_days=60)
    dates = [str(d)[:10] for d in np.asarray(sim.t.timevec)]
    cd = np.asarray(sim.results['cum_diagnoses'])
    idx = [20, 30, 40, 50]
    data = pd.DataFrame({'cum_diagnoses': [cd[i] * 1.1 for i in idx]}, index=[dates[i] for i in idx])
    fit = cv.Fit(sim, data=data, keys=['cum_diagnoses'])
    assert fit.keys == ['cum_diagnoses']
    assert len(fit.pair['cum_diagnoses'].sim) == len(idx), 'all data dates matched sim dates'
    assert np.isfinite(fit.mismatch) and fit.mismatch > 0


def test_fit_auto_keys():
    """With no keys, Fit uses the cumulative keys present in BOTH the sim and the data."""
    sim = _run(n_days=40)
    dates = [str(d)[:10] for d in np.asarray(sim.t.timevec)]
    idx = [15, 25, 35]
    data = pd.DataFrame({'cum_severe': [np.asarray(sim.results['cum_severe'])[i] for i in idx],
                         'not_a_result': [1, 2, 3]}, index=[dates[i] for i in idx])
    fit = cv.Fit(sim, data=data)
    assert 'cum_severe' in fit.keys and 'not_a_result' not in fit.keys


def test_fit_requires_data():
    """Fit raises if given neither data nor custom series (die=True)."""
    sim = _run(n_days=20)
    with pytest.raises(RuntimeError):
        cv.Fit(sim)


# --- cv.Calibration (Optuna) -------------------------------------------------

def _calib_sim(rel_severe=1.0):
    import starsim as ss
    covid = cv.COVID(beta={'a': ss.probperday(0.016)}, init_prev=50, rel_severe_prob=rel_severe)
    return cv.Sim(pop_size=10000, pop_infected=50, pop_type='random', n_days=60, rand_seed=1,
                  verbose=0, diseases=covid)


@pytest.mark.slow
def test_calibration_converges():
    """A small calibration recovers a known parameter: fit rel_severe_prob to data made with 1.6.

    cum_severe increases monotonically with rel_severe_prob (fixed seed), so Optuna should move the
    starting guess (1.0) toward the truth (1.6) -- we assert it lands clearly above the start.
    """
    truth = _calib_sim(rel_severe=1.6); truth.run()
    dates = [str(d)[:10] for d in np.asarray(truth.t.timevec)]
    cs = np.asarray(truth.results['cum_severe'])
    idx = [30, 40, 50, 59]
    data = pd.DataFrame({'cum_severe': [cs[i] for i in idx]}, index=[dates[i] for i in idx])

    calib = cv.Calibration(_calib_sim(1.0), calib_pars=dict(rel_severe_prob=[1.0, 0.5, 2.5]),
                           data=data, total_trials=15, n_workers=1, weights={'cum_severe': 1.0})
    calib.calibrate()
    best = float(calib.best_pars['rel_severe_prob'])
    assert 1.2 < best < 2.2, f'calibration should move toward the truth 1.6, got {best:.2f}'
    assert calib.df is not None and len(calib.df) == 15, 'all trials recorded'
    # The best-fit mismatch should beat the starting guess (rel_severe_prob=1.0).
    start_fit = cv.Fit(_calib_sim(1.0).run(), data=data, keys=['cum_severe'])
    best_fit  = cv.Fit(_calib_sim(best).run(), data=data, keys=['cum_severe'])
    assert best_fit.mismatch < start_fit.mismatch, 'calibration reduces the mismatch vs the start'
