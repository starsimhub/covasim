"""M1 cv.Sim / cv.People assembly tests + continuous-runnability invariant."""
import numpy as np
import pytest
import starsim as ss
import covasim as cv


def _cum_infections(sim):
    return float(np.asarray(sim.diseases.covid.results['cum_infections']).max())


def test_sim_runs_random():
    sim = cv.Sim(pop_size=2000, pop_infected=20, pop_type='random', n_days=40, rand_seed=1)
    sim.run()
    assert sim.results is not None


def test_sim_runs_hybrid():
    sim = cv.Sim(pop_size=2000, pop_infected=20, pop_type='hybrid', n_days=40, rand_seed=1)
    sim.run()
    assert sim.results is not None


def test_default_sim_runs():
    # Continuous-runnability invariant: a bare cv.Sim().run() returns results.
    sim = cv.Sim(pop_size=1000, n_days=20)
    sim.run()
    assert sim.results is not None


def test_pop_infected_exact_seed():
    # Exactly pop_infected agents are infected at t=0 (Open question E: exact-count seed).
    sim = cv.Sim(pop_size=2000, pop_infected=25, pop_type='random', n_days=1, rand_seed=2)
    sim.init()
    assert int(sim.diseases.covid.infected.sum()) == 25


def test_epidemic_grows():
    sim = cv.Sim(pop_size=5000, pop_infected=50, pop_type='random', n_days=60, rand_seed=1)
    sim.run()
    # With beta=0.016/contact and ~20 contacts/day the epidemic should grow well beyond the seed.
    assert _cum_infections(sim) > 50


def test_unsupported_pop_type_raises():
    with pytest.raises(ValueError):
        cv.Sim(pop_type='synthpops')


def test_override_diseases_kwarg():
    # Passing diseases= short-circuits the default assembly.
    covid = cv.COVID(beta=ss.probperday(0.02), init_prev=ss.bernoulli(p=0.02))
    sim = cv.Sim(pop_size=1000, pop_type='random', n_days=10, diseases=covid)
    sim.run()
    assert sim.results is not None


def test_pop_scale_scales_extensive_results():
    """pop_scale multiplies extensive (scale=True) results but leaves intensive ones unchanged.

    Same seed -> identical agent-level dynamics, only the result scaling differs.
    """
    base = cv.Sim(pop_size=10_000, pop_infected=20, pop_type='random', n_days=60, rand_seed=1)
    base.run()
    scaled = cv.Sim(pop_size=10_000, pop_infected=20, pop_type='random', n_days=60, rand_seed=1, pop_scale=10)
    scaled.run()
    rb, rs = base.diseases.covid.results, scaled.diseases.covid.results
    cum_b = float(np.asarray(rb['cum_infections']).max())
    cum_s = float(np.asarray(rs['cum_infections']).max())
    assert cum_s == pytest.approx(10 * cum_b, rel=1e-6), 'extensive results should scale by pop_scale'
    prev_b = float(np.asarray(rb['prevalence']).max())
    prev_s = float(np.asarray(rs['prevalence']).max())
    assert prev_s == pytest.approx(prev_b, rel=1e-6), 'intensive results (prevalence) must be unchanged by pop_scale'


def test_deterministic_same_seed():
    a = cv.Sim(pop_size=2000, pop_infected=20, pop_type='hybrid', n_days=40, rand_seed=8); a.run()
    b = cv.Sim(pop_size=2000, pop_infected=20, pop_type='hybrid', n_days=40, rand_seed=8); b.run()
    assert _cum_infections(a) == _cum_infections(b)
