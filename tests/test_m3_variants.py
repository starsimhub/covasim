"""M3 multi-variant + cross-immunity tests for cv.COVID / cv.variant / cv.CrossImmunity.

Grows across M3 Tasks 1-4. Task 1 (this batch) covers the variant SCAFFOLDING on the
single cv.COVID module while keeping nv==1 byte-identical to M2:

  - the 12-key 2D by_variant results sub-dict exists with the right shape/dtype;
  - at nv==1, the by_variant[0] stocks equal the aggregate stocks (the design-spec
    "results for free" invariant);
  - the scalar *_variant tags follow their lifecycle (set when exposed/infectious,
    rolled to recovered_variant on recovery, cleared on death);
  - the 2D immunity arrays are allocated and all-zero (no effect at nv==1);
  - births are rejected (the arrays are not growth-aware in M3).
"""
import numpy as np
import pytest
import starsim as ss
import covasim as cv


def _arr(x):
    return np.asarray(x)


def _run(pop_type='random', pop_size=8000, pop_infected=30, n_days=60, seed=1):
    sim = cv.Sim(pop_size=pop_size, pop_infected=pop_infected, pop_type=pop_type,
                 n_days=n_days, rand_seed=seed, verbose=0)
    sim.run()
    return sim


# --- by_variant results scaffolding (nv==1) ----------------------------------

BY_VARIANT_KEYS = (
    'prevalence_by_variant', 'incidence_by_variant',
    'new_infections_by_variant', 'cum_infections_by_variant',
    'new_symptomatic_by_variant', 'cum_symptomatic_by_variant',
    'new_severe_by_variant', 'cum_severe_by_variant',
    'new_infectious_by_variant', 'cum_infectious_by_variant',
    'n_exposed_by_variant', 'n_infectious_by_variant',
)


def test_by_variant_results_shape_and_dtype():
    """The 12-key by_variant sub-dict exists, is 2D (nv, npts), and float dtype."""
    sim = _run(n_days=30)
    d = sim.diseases.covid
    assert d.nv == 1 and d.variant_map == {0: 'wild'}
    vres = d.results['variant']
    assert tuple(vres.keys()) == BY_VARIANT_KEYS, 'exact 12-key by_variant contract'
    npts = d.t.npts
    for k in BY_VARIANT_KEYS:
        a = _arr(vres[k])
        assert a.shape == (1, npts), f'{k} must be 2D (nv, npts), got {a.shape}'
        assert a.dtype.kind == 'f', f'{k} must be float dtype (v3 result_float)'


def test_nv1_by_variant_equals_aggregate():
    """At nv==1 the variant-0 stocks equal the aggregate stocks (host exclusivity is structural)."""
    sim = _run()
    r = sim.diseases.covid.results
    vres = r['variant']
    assert np.array_equal(_arr(r['n_infectious']), _arr(vres['n_infectious_by_variant'])[0]), \
        'n_infectious_by_variant[0] must equal aggregate n_infectious at nv==1'
    assert np.array_equal(_arr(r['n_exposed']), _arr(vres['n_exposed_by_variant'])[0]), \
        'n_exposed_by_variant[0] must equal aggregate n_exposed at nv==1'
    # cum_*_by_variant is the cumsum of new_*_by_variant along time.
    for stem in ('infections', 'symptomatic', 'severe', 'infectious'):
        new = _arr(vres[f'new_{stem}_by_variant'])[0]
        cum = _arr(vres[f'cum_{stem}_by_variant'])[0]
        assert np.allclose(cum, np.cumsum(new)), f'cum_{stem}_by_variant must be cumsum(new)'


def test_immunity_arrays_allocated_zero():
    """The 2D immunity arrays are allocated (nv, n_raw) and all-zero (no effect at nv==1)."""
    sim = _run(n_days=20)
    d = sim.diseases.covid
    n_raw = len(d.rel_sus.raw)
    for name in ('sus_imm', 'symp_imm', 'sev_imm'):
        a = getattr(d, name)
        assert a is not None and a.shape == (1, n_raw), f'{name} shape (nv, n_raw)'
        assert not a.any(), f'{name} must be all-zero at nv==1 (no cross-immunity yet)'


# --- scalar variant-tag lifecycle --------------------------------------------

def test_variant_tags_set_on_exposed_and_infectious():
    """Exposed agents carry a finite exposed_variant; infectious carry infectious_variant (==0 at nv==1)."""
    sim = _run(n_days=40)
    d = sim.diseases.covid
    exp = d.exposed.uids
    inf = d.infectious.uids
    if len(exp):
        ev = _arr(d.exposed_variant[exp])
        assert np.isfinite(ev).all() and (ev == 0).all(), 'all exposed are wild (variant 0) at nv==1'
    if len(inf):
        iv = _arr(d.infectious_variant[inf])
        assert np.isfinite(iv).all() and (iv == 0).all(), 'all infectious are wild (variant 0) at nv==1'


def test_recovered_variant_tagged_on_recovery():
    """Recovered agents have recovered_variant set and the active-infection tags cleared."""
    sim = _run(n_days=80)
    d = sim.diseases.covid
    rec = (d.recovered & (~d.infected)).uids
    assert len(rec) > 0, 'expected recoveries by day 80'
    rv = _arr(d.recovered_variant[rec])
    assert np.isfinite(rv).all() and (rv == 0).all(), 'recovered_variant set to wild at nv==1'
    # Agents who recovered (and are not currently re-infected) have no active exposed/infectious tag.
    assert np.isnan(_arr(d.infectious_variant[rec])).all(), 'infectious_variant cleared at recovery'


def test_step_die_clears_variant_tags():
    """step_die clears the three scalar variant tags to NaN (adversary punch-list #3)."""
    covid = cv.COVID(init_prev=None)
    sim = ss.Sim(people=cv.People(2000), diseases=covid, networks='random',
                 start=ss.date('2020-03-01'), dur=ss.days(1), dt=ss.days(1),
                 rand_seed=1, verbose=0, copy_inputs=False)
    sim.init()
    uids = sim.people.auids[:500]
    covid.set_prognoses(uids)
    assert np.isfinite(_arr(covid.exposed_variant[uids])).all(), 'exposed_variant set by set_prognoses'
    victims = uids[:100]
    covid.infectious_variant[victims] = 0.0
    covid.recovered_variant[victims] = 0.0
    covid.step_die(victims)
    v = np.asarray(victims)
    assert np.isnan(_arr(covid.exposed_variant)[v]).all(), 'exposed_variant cleared on death'
    assert np.isnan(_arr(covid.infectious_variant)[v]).all(), 'infectious_variant cleared on death'
    assert np.isnan(_arr(covid.recovered_variant)[v]).all(), 'recovered_variant cleared on death'


def test_births_rejected_in_m3():
    """The 2D immunity arrays are not growth-aware, so births must be rejected (adversary #10)."""
    covid = cv.COVID(init_prev=ss.bernoulli(p=0.02))
    sim = ss.Sim(people=cv.People(1000), diseases=covid, networks='random',
                 demographics=ss.Births(birth_rate=20),
                 start=ss.date('2020-03-01'), dur=ss.days(5), dt=ss.days(1),
                 rand_seed=1, verbose=0, copy_inputs=False)
    with pytest.raises(NotImplementedError):
        sim.init()
