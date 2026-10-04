"""cv.Sim / cv.People assembly tests + continuous-runnability invariant."""
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
    # Exactly pop_infected agents are infected at t=0 (exact-count seed, as in v3).
    sim = cv.Sim(pop_size=2000, pop_infected=25, pop_type='random', n_days=1, rand_seed=2)
    sim.init()
    assert int(sim.diseases.covid.infected.sum()) == 25


def test_epidemic_grows():
    sim = cv.Sim(pop_size=5000, pop_infected=50, pop_type='random', n_days=60, rand_seed=1)
    sim.run()
    # With beta=0.016/contact and ~20 contacts/day the epidemic should grow well beyond the seed.
    assert _cum_infections(sim) > 50

    # As in Starsim, the per-agent arrays on sim.people only include agents who are alive, except for dead and date_dead
    ppl = sim.people
    n_dead = sim.summary['cum_deaths']
    assert n_dead > 0 # Agents have died
    for arr in [ppl, ppl.age, ppl.exposed, ppl.date_exposed]:
        assert len(arr) == 5000 - n_dead
    for arr in [ppl.dead, ppl.date_dead, ppl.age.raw]:
        assert len(arr) == 5000 # All agents ever created
    assert ppl.dead.sum() == n_dead # v3: people.dead counts the agents who have died
    assert len(ppl.age[cv.true(ppl.dead)]) == n_dead # Indexing by UID works for agents who have died
    assert np.array_equal(cv.true(ppl.dead), np.nonzero(sim.diseases.covid.dead.raw[:5000])[0]) # Indices are UIDs, as in v3


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
    base = cv.Sim(pop_size=10_000, pop_infected=20, pop_type='random', n_days=60, rand_seed=1, rescale=False)
    base.run()
    scaled = cv.Sim(pop_size=10_000, pop_infected=20, pop_type='random', n_days=60, rand_seed=1, pop_scale=10, rescale=False)
    scaled.run()
    rb, rs = base.diseases.covid.results, scaled.diseases.covid.results
    cum_b = float(np.asarray(rb['cum_infections']).max())
    cum_s = float(np.asarray(rs['cum_infections']).max())
    assert cum_s == pytest.approx(10 * cum_b, rel=1e-6), 'extensive results should scale by pop_scale'
    prev_b = float(np.asarray(rb['prevalence']).max())
    prev_s = float(np.asarray(rs['prevalence']).max())
    assert prev_s == pytest.approx(prev_b, rel=1e-6), 'intensive results (prevalence) must be unchanged by pop_scale'


def test_dynamic_rescaling():
    """With dynamic rescaling, the scale rises from 1 to pop_scale, giving a similar epidemic to a full-sized population"""
    full = cv.Sim(pop_size=50_000, pop_infected=100, n_days=60, rand_seed=1, verbose=0).run()
    scaled = cv.Sim(pop_size=5_000, pop_infected=100, n_days=60, rand_seed=1, verbose=0, pop_scale=10, rescale=True).run()
    scale = scaled.rescale_vec
    assert scale[0] == 1 and scale[-1] == 10, f'the scale should rise from 1 to 10, not {scale[0]} to {scale[-1]}'
    cum_f = full.results['cum_infections'][-1]
    cum_s = scaled.results['cum_infections'][-1]
    assert cum_s == pytest.approx(cum_f, rel=0.3), f'cumulative infections should be similar with dynamic rescaling ({cum_s} vs. {cum_f})'


def test_deterministic_same_seed():
    a = cv.Sim(pop_size=2000, pop_infected=20, pop_type='hybrid', n_days=40, rand_seed=8); a.run()
    b = cv.Sim(pop_size=2000, pop_infected=20, pop_type='hybrid', n_days=40, rand_seed=8); b.run()
    assert _cum_infections(a) == _cum_infections(b)
