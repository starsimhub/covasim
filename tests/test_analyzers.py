"""Analyzer tests: snapshot / age_histogram / nab_histogram / daily_age_stats / TransTree + plots.

Analyzers run at the analyzer loop slot and only OBSERVE state, so adding one does not change the
sim (byte-identical). The live analyzers are on sim.analyzers (cv.Sim deep-copies inputs).
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
    """cv.snapshot stores full-length per-agent state on the requested days; counts match the results."""
    sim = _run([cv.snapshot(days=[20, 40])])
    snap = sim.analyzers['snapshot']
    assert len(snap.snapshots) == 2
    s = snap.get(40)
    assert 'age' in s and 'infectious' in s and 'nab' in s and 'date_exposed' in s
    # The snapshot infectious count matches the result series at day 40.
    n_inf_res = float(np.asarray(sim.diseases.covid.results['n_infectious'])[40])
    assert s.count('infectious') == int(n_inf_res), 'snapshot infectious count matches results'
    # As in v3, arrays are full length and indexed by agent, including agents who have died.
    s20 = snap.get(20)
    assert len(s) == len(s20) == 20000, 'snapshots are full length on every day'
    n_dead = int(np.asarray(sim.diseases.covid.results['cum_deaths'])[40])
    assert s.count('dead') == n_dead and not s.alive[s.dead].any(), 'dead agents are kept, flagged dead'


def test_age_histogram_counts_defined_dates():
    """cv.age_histogram counts, per age bin, the agents for whom each state's date is defined (as in v3)."""
    sim = _run([cv.age_histogram(days=[40], states=['exposed', 'severe']), cv.snapshot(days=[40])])
    ah = sim.analyzers['age_histogram']
    hist = ah.get(40)
    assert 'severe' in hist and 'exposed' in hist
    people = sim.analyzers['snapshot'].get(40)
    for state in ['exposed', 'severe']:
        expected = np.histogram(people.age[people.defined(f'date_{state}')], bins=ah.edges)[0]
        assert np.array_equal(hist[state], expected), f'age-histogram {state} matches the snapshot'
    # Everyone who has been infected by day 40 is in the transmission log, and vice versa
    log = cv.analysis.make_infection_log(sim)
    n_infected = len(set(e['target'] for e in log if e['date'] <= 40))
    assert int(hist['exposed'].sum()) == n_infected, 'exposed histogram matches the transmission log'


def test_age_histogram_post_run():
    """sim.make_age_histogram() / cv.age_histogram(sim=sim) record the last day of a run sim."""
    sim = _run([], n_days=30)
    agehist = cv.age_histogram(sim=sim)
    assert list(agehist.hists.keys()) == [sim.date(30)]


def test_nab_histogram_counts_positive_nabs():
    """cv.nab_histogram (use_waning) bins log10 NAb levels over agents with NAb > 0."""
    sim = _run([cv.nab_histogram(days=[40])])
    nh = sim.analyzers['nab_histogram']
    hist = nh.hists[0]
    assert int(hist['n'].sum()) > 0, 'some agents have positive NAbs by day 40 under waning'
    import matplotlib; matplotlib.use('agg')
    assert nh.plot() is not None, 'nab_histogram has its own .plot() (not the generic one)'


def test_snapshot_get_by_date_and_day():
    """snapshot.get accepts an int day or a date key."""
    sim = _run([cv.snapshot(days=[30])])
    snap = sim.analyzers['snapshot']
    by_day = snap.get(30)
    by_date = snap.get(list(snap.snapshots.keys())[0])
    assert by_day is by_date


# --- cv.TransTree ------------------------------------------------------------

def test_transtree_records_and_r0():
    """sim.make_transtree() works after a normal run: every infection is logged, with sources + layers."""
    sim = _run([])
    tt = sim.make_transtree(to_networkx=True)
    cum_inf = int(np.asarray(sim.results['cum_infections'])[-1])
    assert len(tt) == cum_inf, 'one log entry per infection, including seed infections'
    layers = set(e['layer'] for e in tt.infection_log)
    assert 'seed_infection' in layers and layers - {'seed_infection', 'importation'} <= set(sim.layer_keys())
    # Offspring counts are consistent with the event list.
    assert len(tt.transmissions) == sum(e['source'] is not None for e in tt.infection_log)
    assert tt.r0() > 0 and len(tt.detailed) == 20000 and len(tt.df) == len(tt.transmissions)


def test_transtree_as_analyzer():
    """cv.TransTree() can also be attached as an analyzer, giving the same tree as making it afterwards."""
    sim = _run([cv.TransTree()], n_days=40)
    tt1 = sim.analyzers['transtree']
    tt2 = cv.TransTree(sim)
    assert len(tt1) > 0 and tt1.infection_log == tt2.infection_log


# --- daily_age_stats + plots -------------------------------------------------

def test_daily_age_stats_records_by_age():
    """cv.daily_age_stats records the new counts by age bin each day (as in v3); totals match the results."""
    sim = _run([cv.daily_age_stats(states=['severe', 'dead'])])
    das = sim.analyzers['daily_age_stats']
    assert len(das.age_results) == sim.npts, 'one entry per day'
    df = das.to_df()
    assert df.shape[0] == sim.npts * 10, 'one row per day and age bin'
    cum_deaths = np.asarray(sim.results['cum_deaths'])[-1]
    assert df.new_dead.sum() == cum_deaths, 'new deaths by age sum to the total deaths'


def test_plots_smoke():
    """cv.Sim.plot, cv.Fit.plot, and cv.TransTree.plot produce figures (agg backend)."""
    import matplotlib
    matplotlib.use('agg')
    sim = _run([cv.TransTree()], n_days=40)
    assert sim.plot() is not None, 'cv.Sim.plot'
    fit = cv.Fit(sim, custom={'cum_deaths': {'data': [10, 20.], 'sim': [11, 19.]}}, die=False)
    assert fit.plot() is not None, 'cv.Fit.plot'
    assert sim.analyzers['transtree'].plot() is not None, 'cv.TransTree.plot'
