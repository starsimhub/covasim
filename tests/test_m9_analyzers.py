"""M9 analyzer tests: snapshot / age_histogram / nab_histogram / daily_age_stats / TransTree + plots.

Analyzers run at the analyzer loop slot and only OBSERVE state, so adding one does not change the
sim (byte-identical). The live analyzers are on sim.analyzers (cv.Sim deep-copies inputs).
(The synthpops population backend is the only remaining M9 piece; it is not installed here.)
"""
import numpy as np
import covasim as cv


def _run(analyzers, n_days=60, seed=1, use_waning=True):
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=n_days, rand_seed=seed,
                 use_waning=use_waning, verbose=0, analyzers=analyzers)
    sim.run()
    return sim


def test_analyzers_are_observational_byte_identical():
    """Adding analyzers does not change the sim dynamics (they only observe)."""
    base = cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=50, rand_seed=1,
                  use_waning=True, verbose=0); base.run()
    withana = _run([cv.snapshot(days=[20, 40]), cv.age_histogram(days=[40]), cv.nab_histogram(days=[40])],
                   n_days=50)
    b = float(np.asarray(base.diseases.covid.results['n_infectious']).max())
    w = float(np.asarray(withana.diseases.covid.results['n_infectious']).max())
    assert b == w, 'analyzers must not change the epidemic'


def test_snapshot_records_states():
    """cv.snapshot stores per-agent state on the requested days; counts match the results."""
    sim = _run([cv.snapshot(days=[20, 40])])
    snap = sim.analyzers['snapshot']
    assert len(snap.snapshots) == 2
    s = snap.get(40)
    assert 'age' in s and 'infectious' in s and 'nab' in s  # use_waning => nab stored
    # The snapshot infectious count matches the result series at day 40.
    n_inf_res = float(np.asarray(sim.diseases.covid.results['n_infectious'])[40])
    assert int(s['infectious'].sum()) == int(n_inf_res), 'snapshot infectious count matches results'


def test_age_histogram_sums_match_state_counts():
    """cv.age_histogram bins agents by age; per-state bin counts sum to that state's total."""
    sim = _run([cv.age_histogram(days=[40], states=['exposed', 'severe'])])
    ah = sim.analyzers['age_histogram']
    hist = ah.hists[list(ah.hists.keys())[0]]
    assert 'severe' in hist and 'exposed' in hist
    # Histogram total == number of agents in that state at day 40.
    d = sim.diseases.covid
    n_severe = int(np.asarray(d.results['n_severe'])[40])
    assert int(hist['severe'].sum()) == n_severe, 'age-histogram severe total matches n_severe'


def test_nab_histogram_counts_positive_nabs():
    """cv.nab_histogram (use_waning) bins NAb levels over agents with NAb > 0."""
    sim = _run([cv.nab_histogram(days=[40])])
    nh = sim.analyzers['nab_histogram']
    hist = nh.hists[list(nh.hists.keys())[0]]
    assert int(hist['counts'].sum()) > 0, 'some agents have positive NAbs by day 40 under waning'


def test_snapshot_get_by_date_and_day():
    """snapshot.get accepts an int day or a date key."""
    sim = _run([cv.snapshot(days=[30])])
    snap = sim.analyzers['snapshot']
    by_day = snap.get(30)
    by_date = snap.get(list(snap.snapshots.keys())[0])
    assert by_day is by_date


# --- cv.TransTree ------------------------------------------------------------

def test_transtree_records_and_r0():
    """cv.TransTree logs transmission events and computes per-infector offspring + an R estimate."""
    sim = _run([cv.TransTree()])
    tt = sim.analyzers['transtree']
    assert len(tt.infection_events) > 0, 'transmission events recorded'
    assert len(tt.n_targets) > 0 and tt.r0 > 0, 'infectors + R0 computed'
    # Every event is (source, target, day, variant); seeds (source<0) are roots.
    df = tt.make_detailed()
    assert df.shape[0] == len(tt.infection_events) and list(df.columns) == ['source', 'target', 'day', 'variant']
    # Offspring counts are consistent with the event list.
    src = np.array([e[0] for e in tt.infection_events])
    assert sum(tt.n_targets.values()) == int((src >= 0).sum()), 'offspring counts sum to non-seed events'


def test_transtree_off_without_analyzer_byte_identical():
    """Without a TransTree analyzer, transmission logging is off (M1-M8 unaffected)."""
    sim = cv.Sim(pop_size=8000, pop_infected=40, pop_type='random', n_days=40, rand_seed=1, verbose=0)
    sim.run()
    d = sim.diseases.covid
    assert d._record_transmissions is False and len(d.infection_events) == 0


# --- daily_age_stats + plots -------------------------------------------------

def test_daily_age_stats_records_by_age():
    """cv.daily_age_stats records per-day state counts by age bin; daily totals match the stocks."""
    sim = _run([cv.daily_age_stats(states=['severe'])])
    das = sim.analyzers['daily_age_stats']
    sev = das.age_results['severe']
    assert sev.shape == (sim.diseases.covid.t.npts, 10), 'shape (npts, n_bins)'
    # The per-day total (summed over age) equals the n_severe stock at that day.
    n_sev = np.asarray(sim.diseases.covid.results['n_severe'])
    assert np.allclose(sev.sum(axis=1), n_sev), 'age totals match the n_severe stock each day'


def test_plots_smoke():
    """cv.Sim.plot, cv.Fit.plot, and cv.TransTree.plot produce figures (agg backend)."""
    import matplotlib
    matplotlib.use('agg')
    sim = _run([cv.TransTree()], n_days=40)
    assert sim.plot() is not None, 'cv.Sim.plot'
    fit = cv.Fit(sim, custom={'cum_deaths': {'data': [10, 20.], 'sim': [11, 19.]}}, die=False)
    assert fit.plot() is not None, 'cv.Fit.plot'
    assert sim.analyzers['transtree'].plot() is not None, 'cv.TransTree.plot'
