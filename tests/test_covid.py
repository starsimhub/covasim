"""Functional + structural tests for cv.COVID.

M1 covered the minimal S->E->I->R; M2 adds the full prognosis tree (symptomatic ->
severe -> critical -> dead, with recovery off each stage). The natural-history
*structural invariants* are tested by calling set_prognoses directly and inspecting
the pre-scheduled timers (removal-free: dead agents are removed from the live arrays
once they die, so post-run state counts can't see them). Transmission is exercised
via a minimal ss.Sim with a stock ss.RandomNet.
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


def _inited_for_prognoses(n_agents=10_000, seed=1):
    """An initialized (not run) sim whose disease we drive via set_prognoses directly.

    Uses cv.People so ages follow Covasim's distribution (with elderly) -- otherwise
    ss.People's default uniform 0-60 ages give almost no severe/critical/dead agents.
    """
    covid = cv.COVID(init_prev=None)  # no auto-seeding; we call set_prognoses manually
    sim = ss.Sim(people=cv.People(n_agents), diseases=covid, networks='random',
                 start=ss.date('2020-03-01'), dur=ss.days(1), dt=ss.days(1),
                 rand_seed=seed, verbose=0, copy_inputs=False)
    sim.init()
    return sim, covid


def _arr(x):
    return np.asarray(x)


# --- basic functional tests --------------------------------------------------

def test_states_present():
    _, covid = _minimal_sim()
    for state in ['susceptible', 'infected', 'exposed', 'symptomatic', 'severe', 'critical', 'recovered', 'dead']:
        assert hasattr(covid, state), f'cv.COVID missing state {state!r}'
    for ti in ['ti_infected', 'ti_exposed', 'ti_infectious', 'ti_symptomatic', 'ti_severe', 'ti_critical', 'ti_recovered', 'ti_dead']:
        assert hasattr(covid, ti), f'cv.COVID missing {ti!r}'


def test_exposed_do_not_transmit():
    sim, covid = _minimal_sim(init_prev=0.05)
    sim.init()
    infected = _arr(covid.infected)
    infectious = _arr(covid.infectious)
    exposed = _arr(covid.exposed)
    assert infected.sum() > 0
    assert exposed.sum() == infected.sum(), 'all seeded agents are exposed (pre-infectious) at t=0'
    assert infectious.sum() == 0, 'no agent infectious at t=0 (all in the latent period)'
    assert (infectious <= infected).all()


def test_full_progression_occurs():
    """A run produces symptomatic, severe and critical agents (the new M2 stages)."""
    sim = cv.Sim(pop_size=20_000, pop_infected=50, pop_type='random', n_days=120, rand_seed=1)
    sim.run()
    d = sim.diseases.covid
    # Survivors who reached each stage (dead agents are removed, so check 'ever reached' via timers on survivors).
    assert (~np.isnan(_arr(d.ti_symptomatic))).sum() > 0, 'expected symptomatic agents'
    assert (~np.isnan(_arr(d.ti_severe))).sum() > 0, 'expected severe agents'
    assert (~np.isnan(_arr(d.ti_critical))).sum() > 0, 'expected critical agents'


def test_deaths_occur():
    """The full tree produces deaths (dead agents are removed, so the population shrinks)."""
    pop_size = 20_000
    sim = cv.Sim(pop_size=pop_size, pop_infected=50, pop_type='random', n_days=120, rand_seed=1)
    sim.run()
    assert len(sim.people) < pop_size, 'expected some COVID deaths (population should shrink)'


def test_permanent_immunity():
    sim, covid = _minimal_sim(n_days=80)
    sim.run()
    recovered = _arr(covid.recovered)
    susceptible = _arr(covid.susceptible)
    assert recovered.sum() > 0
    assert not (recovered & susceptible).any(), 'recovered agents must not be susceptible (no waning in M2)'


def test_deterministic_same_seed():
    a_sim, a = _minimal_sim(seed=7); a_sim.run()
    b_sim, b = _minimal_sim(seed=7); b_sim.run()
    assert np.array_equal(_arr(a.results['cum_infections']), _arr(b.results['cum_infections'])), \
        'same seed should give an identical epidemic'


# --- prognosis-tree structural invariants (via direct set_prognoses) ---------

def test_prognosis_timer_ordering():
    """Scheduled stage timers are monotonically ordered: inf <= sym <= sev <= crit <= dead."""
    sim, d = _inited_for_prognoses()
    uids = sim.people.auids[:5000]
    d.set_prognoses(uids)
    tinf = _arr(d.ti_infectious); tsym = _arr(d.ti_symptomatic)
    tsev = _arr(d.ti_severe); tcrit = _arr(d.ti_critical); tdead = _arr(d.ti_dead)
    u = np.asarray(uids)
    assert (~np.isnan(tinf[u])).all(), 'every infected agent gets ti_infectious'
    for earlier, later in [(tinf, tsym), (tsym, tsev), (tsev, tcrit), (tcrit, tdead)]:
        m = ~np.isnan(later[u])  # only where the later stage is scheduled
        assert (earlier[u][m] <= later[u][m]).all(), 'stage timers must be non-decreasing'


def test_asymptomatic_and_mild_cannot_die():
    """Only agents who reach the critical stage can be scheduled to die."""
    sim, d = _inited_for_prognoses()
    uids = sim.people.auids[:5000]
    d.set_prognoses(uids)
    u = np.asarray(uids)
    tdead = _arr(d.ti_dead); tcrit = _arr(d.ti_critical); tsym = _arr(d.ti_symptomatic)
    dead = ~np.isnan(tdead[u])
    assert dead.sum() > 0, 'expected some scheduled deaths in 5000 infections'
    assert (~np.isnan(tcrit[u][dead])).all(), 'every scheduled death must have passed through critical'
    # Asymptomatic agents (no ti_symptomatic) must recover, never die.
    asymp = np.isnan(tsym[u])
    assert (~np.isnan(_arr(d.ti_recovered)[u][asymp])).all(), 'asymptomatic agents must be scheduled to recover'
    assert np.isnan(tdead[u][asymp]).all(), 'asymptomatic agents must never be scheduled to die'


def test_recovery_xor_death():
    """Each infected agent is scheduled for exactly one of recovery or death."""
    sim, d = _inited_for_prognoses()
    uids = sim.people.auids[:5000]
    d.set_prognoses(uids)
    u = np.asarray(uids)
    has_rec = ~np.isnan(_arr(d.ti_recovered)[u])
    has_dead = ~np.isnan(_arr(d.ti_dead)[u])
    assert not (has_rec & has_dead).any(), 'no agent may have BOTH a recovery and a death scheduled'
    assert (has_rec | has_dead).all(), 'every infected agent must have a recovery or a death scheduled'


def test_step_die_resets_flags():
    """step_die clears all disease flags and sets dead=True for the given uids."""
    sim, d = _inited_for_prognoses()
    uids = sim.people.auids[:1000]
    d.set_prognoses(uids)
    # Force some flags on, then kill them
    victims = uids[:100]
    d.symptomatic[victims] = True
    d.severe[victims] = True
    d.step_die(victims)
    for state in ['infected', 'exposed', 'symptomatic', 'severe', 'critical', 'recovered', 'susceptible']:
        assert not _arr(getattr(d, state))[np.asarray(victims)].any(), f'{state} should be cleared on death'
    assert _arr(d.dead)[np.asarray(victims)].all(), 'dead flag should be set'
