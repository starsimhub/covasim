"""Tests for the post-v4.0 confirmed-gap fixes (tutorial validation follow-up).

Covers: r_eff result, the contact-tracing quarantine fix (trace_time=0 used to be a silent no-op),
the new quarantine flow results, and background n_imports seeding (inert by default).
"""
import numpy as np
import covasim as cv


def _traced_sim(trace_time, seed=1):
    tp = cv.test_prob(symp_prob=0.5, asymp_prob=0.05, symp_quar_prob=1.0, asymp_quar_prob=1.0, start_day=5)
    ct = cv.contact_tracing(trace_probs=0.8, trace_time=trace_time, start_day=8)
    sim = cv.Sim(pop_size=8000, pop_infected=100, pop_type='hybrid', n_days=45, rand_seed=seed,
                 verbose=0, interventions=[tp, ct])
    sim.run()
    return sim


def test_quarantine_fires_with_default_trace_time():
    """Regression: contact_tracing with the DEFAULT trace_time=0 must actually quarantine people.

    Previously same-day quarantine requests were popped before the intervention scheduled them
    (loop-ordering bug), so n_quarantined stayed 0 with trace_time=0.
    """
    sim = _traced_sim(trace_time=0)
    r = sim.diseases.covid.results
    assert float(np.asarray(r['n_quarantined']).max()) > 0, 'trace_time=0 must quarantine contacts'
    assert float(np.asarray(r['new_quarantined']).sum()) > 0
    assert float(np.asarray(r['cum_quarantined']).max()) > 0


def test_quarantine_results_consistent():
    """cum_quarantined is the cumulative sum of new_quarantined; test_yield is in [0, ~]."""
    sim = _traced_sim(trace_time=1)
    r = sim.diseases.covid.results
    cumq = np.asarray(r['cum_quarantined'])
    assert np.allclose(cumq, np.cumsum(np.asarray(r['new_quarantined'])))
    ty = np.asarray(r['test_yield'])
    assert np.all(ty >= 0) and ty.max() > 0


def test_r_eff_present_and_sensible():
    """r_eff is computed, bridged to sim.results, and starts above 1 for a growing epidemic."""
    sim = cv.Sim(pop_size=10000, pop_infected=100, pop_type='hybrid', n_days=60, rand_seed=1, verbose=0)
    sim.run()
    r = sim.diseases.covid.results
    assert 'r_eff' in r and 'r_eff' in sim.results, 'r_eff present + bridged'
    reff = np.asarray(r['r_eff'])
    assert reff.max() > 1.0, 'a growing epidemic has R_eff > 1 early'
    assert np.isfinite(reff).all()


def test_n_imports_inert_by_default():
    """n_imports defaults to 0 -> byte-identical to a sim with no importation."""
    a = cv.Sim(pop_size=5000, pop_infected=20, n_days=30, rand_seed=1, verbose=0); a.run()
    b = cv.Sim(pop_size=5000, pop_infected=20, n_days=30, rand_seed=1, verbose=0, n_imports=0); b.run()
    ai = float(np.asarray(a.diseases.covid.results['cum_infections']).max())
    bi = float(np.asarray(b.diseases.covid.results['cum_infections']).max())
    assert ai == bi, 'n_imports=0 must be inert'


def test_n_imports_seeds_epidemic():
    """n_imports>0 seeds background infections each day; an epidemic grows from zero initial cases."""
    sim = cv.Sim(pop_size=5000, pop_infected=0, n_days=40, rand_seed=1, verbose=0, n_imports=5)
    sim.run()
    r = sim.diseases.covid.results
    assert float(np.asarray(r['n_imports']).sum()) > 0, 'imports recorded'
    assert float(np.asarray(r['cum_infections']).max()) > 0, 'imports seed an epidemic from 0 initial cases'


def test_n_imports_via_dynamic_pars():
    """cv.dynamic_pars(n_imports=...) drives the background importation rate mid-run."""
    sim = cv.Sim(pop_size=5000, pop_infected=10, n_days=40, rand_seed=1, verbose=0,
                 interventions=cv.dynamic_pars(n_imports=dict(days=20, vals=10)))
    sim.run()
    assert float(np.asarray(sim.diseases.covid.results['n_imports']).sum()) > 0
