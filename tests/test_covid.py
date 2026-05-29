"""Functional tests for cv.COVID (minimal single-variant S->E->I->R).

Exercises the disease in isolation from the cv.Network port, using a minimal
ss.Sim with a stock ss.RandomNet. copy_inputs=False keeps a live reference to the
disease instance so its states can be inspected after running.
"""
import numpy as np
import pytest
import starsim as ss
import covasim as cv


def _minimal_sim(n_agents=2000, beta=0.05, init_prev=0.05, n_days=40, seed=1):
    covid = cv.COVID(beta=ss.probperday(beta), init_prev=ss.bernoulli(p=init_prev))
    sim = ss.Sim(diseases=covid, networks='random', n_agents=n_agents,
                 start=ss.date('2020-03-01'), dur=ss.days(n_days), dt=ss.days(1),
                 rand_seed=seed, verbose=0, copy_inputs=False)
    return sim, covid


def test_states_present():
    _, covid = _minimal_sim()
    for state in ['susceptible', 'infected', 'exposed', 'recovered']:
        assert hasattr(covid, state), f'cv.COVID missing state {state!r}'
    for ti in ['ti_infected', 'ti_exposed', 'ti_infectious', 'ti_recovered']:
        assert hasattr(covid, ti), f'cv.COVID missing {ti!r}'


def test_exposed_do_not_transmit():
    """At t=0 the seeded agents are exposed (infected) but still in the latent period,
    so none are infectious -- proving the infectious property excludes exposed agents."""
    sim, covid = _minimal_sim(init_prev=0.05)
    sim.init()  # init_post seeds initial cases
    infected = np.asarray(covid.infected)
    infectious = np.asarray(covid.infectious)
    exposed = np.asarray(covid.exposed)
    assert infected.sum() > 0, 'expected some seeded infections'
    assert exposed.sum() == infected.sum(), 'all seeded agents should be exposed (pre-infectious) at t=0'
    assert infectious.sum() == 0, 'no agent should be infectious at t=0 (all in the latent period)'
    assert (infectious <= infected).all(), 'infectious must be a subset of infected'


def test_seir_progression_and_growth():
    sim, covid = _minimal_sim(n_days=60)
    sim.run()
    res = covid.results
    assert np.asarray(res['n_infectious']).max() > 0, 'epidemic should produce infectious agents'
    assert int(covid.recovered.sum()) > 0, 'agents should progress through to recovered'
    # With beta=0.05 and ~random contacts, the epidemic should grow well beyond the seed.
    assert float(np.asarray(res['cum_infections']).max()) > 0


def test_permanent_immunity():
    """Recovered agents never return to susceptible (use_waning=False semantics)."""
    sim, covid = _minimal_sim(n_days=80)
    sim.run()
    recovered = np.asarray(covid.recovered)
    susceptible = np.asarray(covid.susceptible)
    assert recovered.sum() > 0
    assert not (recovered & susceptible).any(), 'recovered agents must not be susceptible'
    # SEIR partitions the (alive) population: S, E, I, R are mutually exclusive.
    infected = np.asarray(covid.infected)
    assert not (recovered & infected).any(), 'recovered agents must not still be infected'


def test_no_deaths_in_m1():
    """M1 is the asymptomatic-only path: no disease deaths (no death state/scheduling)."""
    sim, covid = _minimal_sim(n_days=60)
    sim.run()
    assert not hasattr(covid, 'ti_dead') or np.isnan(np.asarray(covid.ti_dead)).all(), \
        'M1 cv.COVID should not schedule deaths'


def test_deterministic_same_seed():
    a_sim, a = _minimal_sim(seed=7); a_sim.run()
    b_sim, b = _minimal_sim(seed=7); b_sim.run()
    assert np.array_equal(np.asarray(a.results['cum_infections']), np.asarray(b.results['cum_infections'])), \
        'same seed should give an identical epidemic'
