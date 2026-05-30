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


# === Task 2: infect() override + variant-aware set_prognoses + cv.variant ===

def _cum_inf_traj(sim):
    """The aggregate cum_infections trajectory (state-based, seed-inclusive)."""
    d = sim.diseases.covid
    return np.asarray(d.results['cum_infections'])


def test_nv1_infect_override_byte_identical_to_stock():
    """At nv==1 the overridden infect() is byte-identical to stock ss.Infection.infect().

    Compared on the SAME class name ('covid') so the trans_rng CRN seed is unchanged -- any
    difference would be a genuine divergence in the override (adversary punch-list #1).
    """
    saved = cv.COVID.infect
    try:
        s_over = cv.Sim(pop_size=20000, pop_infected=50, pop_type='random', n_days=80, rand_seed=3, verbose=0)
        s_over.run()
        over = np.asarray(s_over.diseases.covid.results['n_infectious'])
        cv.COVID.infect = ss.Infection.infect  # restore stock single-beta infect on the same class
        s_stock = cv.Sim(pop_size=20000, pop_infected=50, pop_type='random', n_days=80, rand_seed=3, verbose=0)
        s_stock.run()
        stock = np.asarray(s_stock.diseases.covid.results['n_infectious'])
    finally:
        cv.COVID.infect = saved
    assert np.array_equal(over, stock), 'nv==1 infect() override must equal stock infect byte-for-byte'


def test_variant_registration():
    """cv.variant grows the single module's variant axis with stable indices (wild stays 0)."""
    sim = cv.Sim(pop_size=4000, pop_infected=20, pop_type='random', n_days=5, rand_seed=1, verbose=0,
                 variants=[cv.variant('alpha', days=2), cv.variant('delta', days=3)])
    sim.init()
    d = sim.diseases.covid
    assert d.nv == 3
    assert d.variant_map == {0: 'wild', 1: 'alpha', 2: 'delta'}
    assert set(d.variant_pars) == {'wild', 'alpha', 'delta'}
    assert d.sus_imm.shape[0] == 3, '2D immunity arrays sized to nv'
    # The default alpha rel_beta (1.67) is registered verbatim from get_variant_pars.
    assert d.variant_pars['alpha']['rel_beta'] == cv.get_variant_pars(variant='alpha')['rel_beta']


def test_two_variant_run_independent_draws():
    """A 2-variant run completes without a DistNotReadyError and both variants infect agents."""
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=70, rand_seed=2, verbose=0,
                 variants=cv.variant('alpha', days=10, n_imports=30))
    sim.run()  # no DistNotReadyError
    d = sim.diseases.covid
    ci = np.asarray(d.results['variant']['cum_infections_by_variant'])
    assert ci[0, -1] > 0, 'wild infects'
    assert ci[1, -1] > 0, 'alpha infects (independent per-variant transmission draws)'


def test_per_variant_rel_beta_scales_transmission():
    """A higher per-variant rel_beta yields a larger epidemic (same seed, t0 introduction)."""
    def run(rel_beta):
        v = cv.variant({'rel_beta': rel_beta, 'label': 'test'}, days=0, n_imports=50)
        sim = cv.Sim(pop_size=20000, pop_infected=0, pop_type='random', n_days=70, rand_seed=1,
                     verbose=0, variants=v)
        sim.run()
        ci = np.asarray(sim.diseases.covid.results['variant']['cum_infections_by_variant'])
        return ci[1, -1]  # the 'test' variant is index 1 (wild=0, seeded 0)
    hi = run(2.5)
    lo = run(0.4)
    assert hi > lo, f'higher rel_beta must transmit more: rel_beta=2.5 -> {hi}, 0.4 -> {lo}'


def test_midrun_import_bumps_n_imports_on_the_right_day():
    """Mid-run apply() seeds exactly n_imports on the introduction day, recorded in n_imports."""
    day, n_imp = 20, 25
    sim = cv.Sim(pop_size=20000, pop_infected=50, pop_type='random', n_days=40, rand_seed=1, verbose=0,
                 variants=cv.variant('delta', days=day, n_imports=n_imp))
    sim.run()
    n_imports = np.asarray(sim.diseases.covid.results['n_imports'])
    assert n_imports[:day].sum() == 0, 'no imports before the introduction day'
    assert n_imports[day] == n_imp, f'exactly {n_imp} imports on day {day}, got {n_imports[day]}'
    assert n_imports.sum() == n_imp, 'no spurious imports on other days'


def test_string_variant_sugar_introduces_at_t0():
    """A bare string/dict variant is introduced at t0 (days=0)."""
    sim = cv.Sim(pop_size=8000, pop_infected=20, pop_type='random', n_days=30, rand_seed=1, verbose=0,
                 variants='beta')
    sim.run()
    d = sim.diseases.covid
    assert d.variant_map == {0: 'wild', 1: 'beta'}
    ci = np.asarray(d.results['variant']['cum_infections_by_variant'])
    assert ci[1, -1] > 0, 'the beta variant (introduced at t0) infects agents'


# === Task 3: cv.CrossImmunity connector + reinfection ===

def _multivariant_sim(seed=2, n_days=100):
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='random', n_days=n_days, rand_seed=seed,
                 verbose=0, variants=cv.variant('beta', days=20, n_imports=30))
    sim.run()
    return sim


def test_connector_auto_attached_and_flag_set():
    """A multi-variant cv.Sim auto-attaches cv.CrossImmunity and flips cross_immunity_active on."""
    sim = _multivariant_sim()
    assert 'crossimmunity' in sim.connectors, 'CrossImmunity auto-attached when nv>1'
    assert isinstance(sim.connectors.crossimmunity, cv.CrossImmunity)
    assert sim.diseases.covid.cross_immunity_active is True


def test_nv1_no_connector_permanent_immunity():
    """At nv==1 no connector is attached, recovered stay immune, and behavior is unchanged."""
    sim = cv.Sim(pop_size=8000, pop_infected=30, pop_type='random', n_days=80, rand_seed=1, verbose=0)
    sim.run()
    assert len(sim.connectors) == 0, 'no connector at nv==1'
    d = sim.diseases.covid
    assert d.cross_immunity_active is False
    rec = (d.recovered & d.susceptible).uids
    assert len(rec) == 0, 'no recovered agent is susceptible again at nv==1 (permanent immunity)'


def test_crossimmunity_matrix_written_to_imm_arrays():
    """The connector writes sus_imm = matrix[target, recovered_variant] for ever-recovered agents."""
    sim = _multivariant_sim()
    d = sim.diseases.covid
    matrix = sim.connectors.crossimmunity.matrix
    ti = d.ti
    rec = (d.ti_recovered <= ti).uids
    assert len(rec) > 0, 'expected ever-recovered agents'
    ru = np.asarray(rec)
    src_v = np.asarray(d.recovered_variant[rec]).astype(int)
    for v in range(d.nv):
        expected = matrix[v, src_v]
        assert np.allclose(d.sus_imm[v, ru], expected), f'sus_imm[{v}] must equal matrix[{v}, recovered_variant]'
        assert np.allclose(d.symp_imm[v, ru], expected), 'symp_imm written from the same matrix (axis C: all three)'
        assert np.allclose(d.sev_imm[v, ru], expected), 'sev_imm written from the same matrix'


def test_same_variant_reinfection_is_zero():
    """Same-variant protection is the diagonal (1.0) => same-variant reinfection is exactly 0.

    Documented M3 divergence from v3's NAb numerics (calc_VE(nab*1.0) < 1 permits a tiny same-variant
    reinfection). The static matrix gives matrix[v,v]==1.0 => rel_sus*(1-1)=0 => no same-variant reinfection.
    """
    sim = _multivariant_sim()
    d = sim.diseases.covid
    rec = (d.ti_recovered <= d.ti).uids
    src_v = np.asarray(d.recovered_variant[rec]).astype(int)
    ru = np.asarray(rec)
    # The protection against the OWN recovered variant must be exactly 1.0 (full) for every ever-recovered agent.
    own_imm = d.sus_imm[src_v, ru]
    assert np.allclose(own_imm, 1.0), 'same-variant sus_imm must be 1.0 (no same-variant reinfection)'


def test_reinfection_enabled_under_cross_immunity():
    """Cross-immunity active => reinfection occurs (total infections can exceed the population)."""
    sim = _multivariant_sim()
    d = sim.diseases.covid
    ci = np.asarray(d.results['variant']['cum_infections_by_variant'])
    total = ci[:, -1].sum()
    assert total > 20000, f'reinfection should push total infections above pop_size, got {total}'


def test_cross_immunity_reduces_heterologous_reinfection():
    """The cross-immunity MATRIX gates reinfection: stronger off-diagonal => fewer 2nd-variant cases.

    Isolates cross-immunity from rel_beta by using ONE custom variant (fixed rel_beta) and overriding
    only the cross-immunity matrix off-diagonal. High off-diagonal (wild-recovered are well protected
    against the new variant) suppresses the second variant; low off-diagonal lets it escape and spread.
    """
    def second_variant_total(offdiag):
        v = cv.variant({'rel_beta': 1.5, 'label': 'X'}, days=25, n_imports=30)
        matrix = np.array([[1.0, offdiag], [offdiag, 1.0]])  # matrix[target, source]
        conn = cv.CrossImmunity(immunity=matrix)
        sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='random', n_days=100, rand_seed=5,
                     verbose=0, variants=v, connectors=conn)
        sim.run()
        ci = np.asarray(sim.diseases.covid.results['variant']['cum_infections_by_variant'])
        return ci[1, -1]
    strong = second_variant_total(0.9)  # strong cross-protection => 2nd variant suppressed
    weak   = second_variant_total(0.1)  # weak cross-protection  => 2nd variant escapes, spreads
    assert weak > strong, f'weaker cross-immunity must permit more reinfection: weak={weak}, strong={strong}'


def test_host_exclusivity_one_seir_chain():
    """Each agent runs exactly ONE SEIR chain: infectious agents carry a single, consistent variant tag."""
    sim = cv.Sim(pop_size=10000, pop_infected=80, pop_type='random', n_days=40, rand_seed=3, verbose=0,
                 variants=cv.variant('delta', days=5, n_imports=40))
    sim.run()
    d = sim.diseases.covid
    inf = d.infectious.uids
    if len(inf):
        ev = np.asarray(d.exposed_variant[inf])
        iv = np.asarray(d.infectious_variant[inf])
        # infectious_variant is a single valid index, and equals exposed_variant (one chain, one variant).
        assert np.isfinite(iv).all(), 'every infectious agent has a single finite infectious_variant'
        assert np.array_equal(ev, iv), 'exposed_variant == infectious_variant for infectious agents (one chain)'
        assert set(np.unique(iv)).issubset({0, 1}), 'variant tags are valid indices'
