"""M8 MultiSim / Scenarios / parallel tests.

cv.MultiSim runs a sim across seeds and reduces to a median + quantile band; cv.Scenarios compares
named parameter-set scenarios; cv.parallel/multi_run/single_run are the run helpers. All wrap
ss.MultiSim/ss.parallel but reduce over the per-seed COVID results themselves (ss.MultiSim.reduce
does not handle Covasim's bridged/nested results).
"""
import numpy as np
import covasim as cv


def _base(n_days=40, pop_size=8000, pop_infected=50, **kw):
    return cv.Sim(pop_size=pop_size, pop_infected=pop_infected, pop_type='random', n_days=n_days,
                  verbose=0, **kw)


# --- cv.MultiSim -------------------------------------------------------------

def test_multisim_runs_and_reduces():
    """cv.MultiSim runs N seeds and reduces to a median trajectory + 10/90 band."""
    msim = cv.MultiSim(_base(), n_runs=5).run(verbose=0).reduce(quantiles=[0.1, 0.9])
    assert len(msim.sims) == 5
    ni = msim.results['n_infectious']
    assert (ni.best <= ni.high + 1e-9).all() and (ni.best >= ni.low - 1e-9).all(), 'median within band'
    assert (ni.high - ni.low).max() > 0, 'band has width (seeds differ)'
    assert 'cum_severe' in msim.results, 'multiple result keys reduced'


def test_multisim_seeds_differ():
    """The per-seed runs are genuinely different (distinct seeds)."""
    msim = cv.MultiSim(_base(), n_runs=4).run(verbose=0)
    peaks = [float(np.asarray(s.diseases.covid.results['n_infectious']).max()) for s in msim.sims]
    assert len(set(peaks)) > 1, 'seeds should give different trajectories'


def test_multisim_mean_vs_median():
    """mean() and median() both produce a best trajectory + band."""
    msim = cv.MultiSim(_base(), n_runs=5).run(verbose=0)
    mean_r = msim.mean().results['cum_severe'].best.copy()
    med_r = msim.median().results['cum_severe'].best.copy()
    assert np.isfinite(mean_r).all() and np.isfinite(med_r).all()


# --- run helpers -------------------------------------------------------------

def test_multi_run_and_single_run():
    sims = cv.multi_run(_base(n_days=30), n_runs=3, verbose=0)
    assert len(sims) == 3 and all(s.diseases.covid.results is not None for s in sims)
    one = cv.single_run(_base(n_days=20))
    assert one.diseases.covid.results is not None


def test_parallel_distinct_sims():
    """cv.parallel runs a list of distinct sims and returns a cv.MultiSim."""
    s1 = _base(n_days=25, pop_infected=30)
    s2 = _base(n_days=25, pop_infected=120)  # more seeds -> bigger epidemic
    msim = cv.parallel([s1, s2], verbose=0)
    assert len(msim.sims) == 2
    p1 = float(np.asarray(msim.sims[0].diseases.covid.results['n_infectious']).max())
    p2 = float(np.asarray(msim.sims[1].diseases.covid.results['n_infectious']).max())
    assert p2 > p1, 'more initial infections -> larger epidemic'


# --- cv.Scenarios ------------------------------------------------------------

def test_scenarios_comparison():
    """cv.Scenarios runs named scenarios; a vaccination scenario has fewer infections than baseline."""
    scens = {
        'baseline':  {'name': 'No intervention', 'pars': {}},
        'vaccinate': {'name': 'Pfizer', 'pars': {'use_waning': True,
                      'interventions': cv.vaccinate_num('pfizer', num_doses=1000, sequence='age')}},
    }
    scn = cv.Scenarios(basepars=dict(pop_size=8000, pop_infected=50, pop_type='random', n_days=70, verbose=0),
                       scenarios=scens, n_runs=3).run(verbose=0)
    base = scn.results['baseline'].results['cum_infections'].best[-1]
    vacc = scn.results['vaccinate'].results['cum_infections'].best[-1]
    assert vacc < base, f'vaccination scenario should have fewer infections: base={base:.0f} vacc={vacc:.0f}'
    assert scn.results['vaccinate'].label == 'Pfizer'


def test_multisim_plot_smoke():
    """cv.MultiSim.plot produces a figure (the UQ demo path)."""
    import matplotlib
    matplotlib.use('agg')
    msim = cv.MultiSim(_base(n_days=30), n_runs=3).run(verbose=0)
    fig = msim.plot(keys=['n_infectious', 'cum_severe'])
    assert fig is not None and len(fig.axes) == 2
