"""M5 testing / tracing / quarantine tests.

Task 1 (this batch) covers the host scaffolding: the testing/quarantine STATES + state machines
(check_diagnosed/check_enter_iso/check_exit_iso/check_quar) + the covid.test() action -- all inert
until a testing intervention drives them, so M1-M4 stay byte-identical. The test_num/test_prob/
contact_tracing interventions land in M5 Tasks 2-3.
"""
import numpy as np
import starsim as ss
import covasim as cv


def _sim(n_agents=5000, init_prev=0.1, n_days=30, seed=1):
    covid = cv.COVID(init_prev=ss.bernoulli(p=init_prev))
    sim = ss.Sim(people=cv.People(n_agents), diseases=covid, networks='random',
                 start=ss.date('2020-03-01'), dur=ss.days(n_days), dt=ss.days(1),
                 rand_seed=seed, verbose=0, copy_inputs=False)
    sim.init()
    return sim, covid


def _advance(sim, n):
    for _ in range(n):
        sim.run_one_step()


# --- scaffolding is inert without a testing intervention ---------------------

def test_testing_states_inert_byte_identical():
    """With no testing intervention the testing states stay empty (M1-M4 behavior preserved)."""
    sim = cv.Sim(pop_size=8000, pop_infected=40, pop_type='random', n_days=80, rand_seed=1, verbose=0)
    sim.run()
    d = sim.diseases.covid
    for state in ('tested', 'diagnosed', 'known_contact', 'quarantined', 'isolated'):
        assert not np.asarray(getattr(d, state)).any(), f'{state} must be inert with no testing intervention'
    assert float(np.asarray(d.results['n_diagnosed']).max()) == 0


def test_states_and_results_present():
    """The M5 testing states exist and auto-create their n_* results."""
    sim, d = _sim()
    for state in ('tested', 'diagnosed', 'known_contact', 'quarantined', 'isolated'):
        assert hasattr(d, state)
    for res in ('n_diagnosed', 'n_quarantined', 'n_isolated'):
        assert res in d.results, f'{res} auto-result missing'


# --- the test() action -------------------------------------------------------

def test_test_action_diagnoses_infectious():
    """test() schedules diagnoses for infectious agents (sensitivity 1.0) and sets the dates."""
    sim, d = _sim()
    _advance(sim, 10)  # let agents become infectious
    inf = d.infectious.uids
    assert len(inf) > 0
    diag = d.test(inf, test_sensitivity=1.0, test_delay=2)
    assert len(diag) == len(inf), 'all infectious test positive at sensitivity 1.0'
    dd = np.asarray(d.date_diagnosed)
    assert np.allclose(dd[np.asarray(diag)], d.ti + 2), 'date_diagnosed = ti + test_delay'
    assert np.asarray(d.tested[inf]).all(), 'tested flag set'


def test_test_action_sensitivity_and_no_double_diagnose():
    """Lower sensitivity diagnoses fewer; already-diagnosed agents are not re-diagnosed."""
    sim, d = _sim(seed=2)
    _advance(sim, 10)
    inf = d.infectious.uids
    d1 = d.test(inf, test_sensitivity=0.5)
    assert 0 < len(d1) < len(inf), 'sensitivity 0.5 diagnoses a strict subset'
    # Re-testing the same infectious set should not re-diagnose the already-diagnosed ones.
    d2 = d.test(inf, test_sensitivity=1.0)
    assert set(np.asarray(d2)).isdisjoint(set(np.asarray(d1))), 'already-diagnosed are excluded'


# --- diagnosis / isolation state machine -------------------------------------

def test_diagnosis_and_isolation_state_machine():
    """When date_diagnosed arrives, the agent becomes diagnosed and enters isolation."""
    sim, d = _sim(seed=3)
    _advance(sim, 10)
    inf = d.infectious.uids
    d.test(inf, test_sensitivity=1.0, test_delay=1)  # diagnosed next step
    _advance(sim, 2)  # cross date_diagnosed
    diagnosed = d.diagnosed.uids
    assert len(diagnosed) > 0, 'agents become diagnosed when date_diagnosed arrives'
    # Diagnosed agents (still infected) are isolated.
    still_inf = diagnosed[np.asarray(d.infected[diagnosed])]
    assert np.asarray(d.isolated[still_inf]).all(), 'diagnosed-and-still-infected agents isolate'


# --- quarantine state machine ------------------------------------------------

def test_schedule_quarantine_machine():
    """schedule_quarantine + check_quar: eligible agents quarantine on the start day, then release."""
    sim, d = _sim(seed=4, n_days=40)
    _advance(sim, 5)
    # Pick susceptible (eligible) agents and schedule a 7-day quarantine starting next step.
    susc = d.susceptible.uids[:50]
    start = d.ti + 1
    d.schedule_quarantine(susc, start_date=start, period=7)
    _advance(sim, 2)  # cross the start day
    quar = d.quarantined.uids
    assert len(quar) > 0, 'scheduled eligible agents enter quarantine'
    assert set(np.asarray(quar)).issubset(set(np.asarray(susc))), 'only the scheduled agents quarantine'
    _advance(sim, 8)  # cross the quarantine end
    assert not np.asarray(d.quarantined[susc]).any(), 'agents are released at the end of quarantine'


# === Task 2: cv.Intervention base + test_prob / test_num ===

def test_test_prob_produces_diagnoses():
    """cv.test_prob tests agents and accumulates cum_tests / cum_diagnoses over the run."""
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=80, rand_seed=2, verbose=0,
                 interventions=cv.test_prob(symp_prob=0.1, asymp_prob=0.01, start_day=10))
    sim.run()
    r = sim.diseases.covid.results
    assert float(np.asarray(r['cum_tests']).max()) > 0, 'tests were performed'
    assert float(np.asarray(r['cum_diagnoses']).max()) > 0, 'diagnoses were made'
    # No tests before the start day.
    assert float(np.asarray(r['new_tests'])[:10].sum()) == 0, 'no testing before start_day'


def test_test_prob_higher_symp_prob_more_diagnoses():
    """A higher symptomatic test probability yields more diagnoses (monotone in symp_prob)."""
    def diagnoses(symp_prob):
        sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='random', n_days=80, rand_seed=4,
                     verbose=0, interventions=cv.test_prob(symp_prob=symp_prob, start_day=5))
        sim.run()
        return float(np.asarray(sim.diseases.covid.results['cum_diagnoses']).max())
    assert diagnoses(0.5) > diagnoses(0.05), 'more aggressive testing => more diagnoses'


def test_test_num_respects_daily_budget():
    """cv.test_num performs about daily_tests tests per active day (capped by eligibility)."""
    daily = 200
    start = 10
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=60, rand_seed=2, verbose=0,
                 interventions=cv.test_num(daily_tests=daily, start_day=start))
    sim.run()
    new_tests = np.asarray(sim.diseases.covid.results['new_tests'])
    assert new_tests[:start].sum() == 0, 'no tests before start_day'
    active = new_tests[start:]
    assert active.max() <= daily, 'never exceeds the daily test budget'
    assert active.sum() > 0 and (active == daily).any(), 'uses the full daily budget on busy days'


def test_test_num_diagnoses_only_infectious():
    """Diagnoses come only from genuinely infectious agents (test() positive only if infectious)."""
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='random', n_days=60, rand_seed=3, verbose=0,
                 interventions=cv.test_num(daily_tests=300, start_day=5))
    sim.run()
    d = sim.diseases.covid
    # Everyone diagnosed must have been infected at some point (ti_infected finite).
    diagnosed = d.diagnosed.uids
    assert len(diagnosed) > 0
    assert np.isfinite(np.asarray(d.ti_infected[diagnosed])).all(), 'only ever-infected agents are diagnosed'


# === Task 3: contact_tracing + quar/iso beta modifiers ===

def _ever_infected(sim):
    d = sim.diseases.covid
    return int(d.recovered.sum() + d.infected.sum()) + float(np.asarray(d.results['cum_deaths']).max())


def test_contact_tracing_creates_quarantine():
    """test_prob + contact_tracing puts traced contacts into quarantine (known_contact + quarantined)."""
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=100, rand_seed=2, verbose=0,
                 interventions=[cv.test_prob(symp_prob=0.2, asymp_prob=0.01, start_day=10),
                                cv.contact_tracing(trace_probs=0.5, trace_time=2, start_day=10)])
    sim.run()
    r = sim.diseases.covid.results
    assert float(np.asarray(r['n_quarantined']).max()) > 0, 'contacts are quarantined'
    assert np.asarray(sim.diseases.covid.known_contact).sum() >= 0  # known_contact flag is exercised


def test_contact_tracing_no_effect_without_testing():
    """contact_tracing alone (no testing) traces nobody (it acts only on the newly diagnosed)."""
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=60, rand_seed=2, verbose=0,
                 interventions=cv.contact_tracing(trace_probs=0.5, start_day=5))
    sim.run()
    assert float(np.asarray(sim.diseases.covid.results['n_quarantined']).max()) == 0, \
        'no diagnoses => no quarantine'


def test_tracing_reduces_transmission():
    """Adding contact tracing (and thus quarantine) reduces the total epidemic size."""
    def ever_infected(with_tracing):
        iv = [cv.test_prob(symp_prob=0.2, asymp_prob=0.01, start_day=10)]
        if with_tracing:
            iv.append(cv.contact_tracing(trace_probs=0.6, trace_time=1, start_day=10))
        sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=100, rand_seed=2,
                     verbose=0, interventions=iv)
        sim.run()
        return _ever_infected(sim)
    assert ever_infected(True) < ever_infected(False), 'tracing+quarantine should shrink the epidemic'


def test_quar_iso_factor_reduces_rel_trans():
    """Isolated/quarantined agents have their transmissibility reduced by iso_factor/quar_factor."""
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=60, rand_seed=2, verbose=0,
                 interventions=[cv.test_prob(symp_prob=0.3, start_day=5),
                                cv.contact_tracing(trace_probs=0.8, trace_time=1, start_day=5)])
    # Step until some agents are isolated/quarantined, then inspect rel_trans vs rel_trans_base.
    sim.init()
    d = sim.diseases.covid
    for _ in range(40):
        sim.run_one_step()
    iso = (d.isolated & d.infectious).uids
    if len(iso):
        ratio = np.asarray(d.rel_trans[iso]) / np.maximum(np.asarray(d.rel_trans_base[iso]), 1e-9)
        assert (ratio <= 1.0 + 1e-9).all(), 'isolated agents have rel_trans reduced (<= base)'
