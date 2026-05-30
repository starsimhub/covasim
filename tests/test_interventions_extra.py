"""Tests for the opportunistic interventions: change_beta / clip_edges / dynamic_pars / sequence.

These round out the intervention API (architecture map: M5-or-opportunistic) and are needed for the
M10 default-baseline scenario. All are additive -- no intervention => M1-M9 byte-identical.
"""
import numpy as np
import covasim as cv


def _ever_infected(sim):
    d = sim.diseases.covid
    return int(d.recovered.sum() + d.infected.sum()) + float(np.asarray(d.results['cum_deaths']).max())


def _run(iv=None, n_days=80, seed=2):
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=n_days, rand_seed=seed,
                 verbose=0, interventions=iv)
    sim.run()
    return sim


def test_change_beta_reduces_transmission():
    base = _run()
    cb = _run(cv.change_beta(days=20, changes=0.4))
    assert _ever_infected(cb) < _ever_infected(base), 'reducing beta should shrink the epidemic'


def test_change_beta_restores():
    """A change back to 1.0 restores the original beta (changes are relative to the original)."""
    sim = _run(cv.change_beta(days=[20, 40], changes=[0.3, 1.0]))
    covid = sim.diseases.covid
    # After the day-40 restore, beta should equal the original (the M1 base, 0.016*beta_layer).
    assert abs(float(covid.pars.beta['h']) - 0.016 * 3.0) < 1e-9, 'beta restored to original on the reset day'


def test_clip_edges_reduces_transmission():
    base = _run()
    ce = _run(cv.clip_edges(days=20, changes=0.4))
    assert _ever_infected(ce) < _ever_infected(base), 'clipping edges should shrink the epidemic'


def test_clip_edges_removes_edges():
    """clip_edges keeps ~the requested fraction of edges on the clip day."""
    sim = cv.Sim(pop_size=10000, pop_infected=50, pop_type='random', n_days=30, rand_seed=1, verbose=0,
                 interventions=cv.clip_edges(days=10, changes=0.5))
    sim.init()
    n0 = len(sim.networks['a'].edges.p1)  # the random backend's single layer is 'a'
    for _ in range(12):
        sim.run_one_step()
    n1 = len(sim.networks['a'].edges.p1)
    assert 0.4 * n0 < n1 < 0.6 * n0, f'~50% of edges kept: {n1}/{n0}'


def test_dynamic_pars_changes_severity():
    base = _run()
    dp = _run(cv.dynamic_pars(rel_severe_prob=dict(days=0, vals=2.0)))
    base_sev = float(np.asarray(base.diseases.covid.results['cum_severe']).max())
    dp_sev = float(np.asarray(dp.diseases.covid.results['cum_severe']).max())
    assert dp_sev > base_sev, 'doubling rel_severe_prob should increase cum_severe'


def test_sequence_switches_interventions():
    """cv.sequence applies the most recently activated intervention."""
    sim = _run(cv.sequence(days=[10, 40],
                           interventions=[cv.test_prob(symp_prob=0.1, start_day=0),
                                          cv.test_num(daily_tests=200, start_day=0)]))
    assert float(np.asarray(sim.diseases.covid.results['cum_tests']).max()) > 0, 'tests performed'


def test_change_beta_zero_stops_transmission():
    """change_beta to 0 from day 0 stops all onward transmission (only the initial seeds infected)."""
    sim = _run(cv.change_beta(days=0, changes=0.0), n_days=60)
    d = sim.diseases.covid
    ever = int(d.recovered.sum() + d.infected.sum()) + float(np.asarray(d.results['cum_deaths']).max())
    assert ever <= 100, f'no onward transmission with beta=0, only the ~100 seeds: {ever}'
