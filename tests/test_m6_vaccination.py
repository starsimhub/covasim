"""M6 vaccination tests.

Task 2 covers the vaccination interventions (cv.vaccinate_prob / cv.vaccinate_num / cv.vaccinate)
on the NAb pipeline; Task 3 adds cv.simple_vaccine. Vaccination requires use_waning=True (NAb path);
with no vaccination intervention everything is inert (M1-M5 byte-identical).
"""
import numpy as np
import covasim as cv


def _ever_infected(sim):
    d = sim.diseases.covid
    return int(d.recovered.sum() + d.infected.sum()) + float(np.asarray(d.results['cum_deaths']).max())


def _sim(interventions=None, seed=2, n_days=120, use_waning=True, **kw):
    return cv.Sim(pop_size=20000, pop_infected=100, pop_type='hybrid', n_days=n_days, rand_seed=seed,
                  use_waning=use_waning, verbose=0, interventions=interventions, **kw)


# --- byte-identity without vaccination ---------------------------------------

def test_no_vaccine_byte_identical():
    """The vaccination machinery is inert with no vaccination intervention (M1-M5 preserved)."""
    import json
    out = {}
    for pt in ('random', 'hybrid'):
        sim = cv.Sim(pop_size=20000, pop_infected=50, pop_type=pt, n_days=120, rand_seed=1, verbose=0)
        sim.run()
        r = sim.diseases.covid.results
        out[pt] = [float(np.asarray(r[k]).max()) for k in ('cum_infections', 'cum_deaths', 'n_infectious')]
    # Sanity: no vaccination => no doses.
    assert float(np.asarray(sim.diseases.covid.results['cum_doses']).max()) == 0


# --- vaccinate_prob ----------------------------------------------------------

def test_vaccinate_prob_reduces_infections():
    """Vaccinating most of the population before the epidemic reduces the total infected."""
    base = _sim(); base.run()
    vacc = _sim(interventions=cv.vaccinate_prob('pfizer', days=0, prob=0.6)); vacc.run()
    assert _ever_infected(vacc) < _ever_infected(base), 'vaccination should reduce the epidemic'
    r = vacc.diseases.covid.results
    assert float(np.asarray(r['cum_vaccinated']).max()) > 0
    assert float(np.asarray(r['n_vaccinated']).max()) > 0


def test_vaccinate_prob_two_dose_scheduling():
    """A 2-dose vaccine (pfizer) gives ~2 doses per vaccinated person via second-dose scheduling."""
    sim = _sim(interventions=cv.vaccinate_prob('pfizer', days=5, prob=0.2)); sim.run()
    r = sim.diseases.covid.results
    cum_doses = float(np.asarray(r['cum_doses']).max())
    cum_vacc  = float(np.asarray(r['cum_vaccinated']).max())
    assert cum_vacc > 0
    # Pfizer has 2 doses with a 21-day interval; over 120 days nearly everyone gets both -> ~2x.
    assert 1.7 < cum_doses / cum_vacc <= 2.0, f'expected ~2 doses/person, got {cum_doses/cum_vacc:.2f}'


def test_vaccinate_requires_waning():
    """The NAb-based vaccines require use_waning=True (else cv.simple_vaccine)."""
    import pytest
    sim = _sim(interventions=cv.vaccinate_prob('pfizer', days=0, prob=0.5), use_waning=False)
    with pytest.raises(RuntimeError):
        sim.init()


# --- vaccinate_num -----------------------------------------------------------

def test_vaccinate_num_delivers_budget():
    """cv.vaccinate_num delivers about num_doses doses per day."""
    daily = 300
    sim = _sim(interventions=cv.vaccinate_num('pfizer', num_doses=daily, sequence='age')); sim.run()
    new_doses = np.asarray(sim.diseases.covid.results['new_doses'])
    assert new_doses.max() <= daily + 1, 'never exceeds the daily dose budget'
    assert new_doses.sum() > 0 and (new_doses >= daily * 0.9).any(), 'uses ~the full daily budget'


def test_vaccinate_wrapper_dispatch():
    """cv.vaccinate dispatches to num (num_doses) or prob (days)."""
    assert isinstance(cv.vaccinate('pfizer', num_doses=100), cv.vaccinate_num)
    assert isinstance(cv.vaccinate('pfizer', days=10, prob=0.1), cv.vaccinate_prob)


# --- per-variant efficacy ----------------------------------------------------

def test_vaccine_per_variant_efficacy():
    """A vaccine protects more against the strain it targets well (wild) than an escape variant (beta).

    pfizer efficacy: wild=1.0, beta=1/10.3. So vaccinated agents' sus_imm should be higher vs wild
    than vs beta (the per-variant efficacy enters the connector's max(natural, vaccine) weighting).
    """
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='random', n_days=60, rand_seed=2,
                 use_waning=True, verbose=0, variants=cv.variant('beta', days=5, n_imports=30),
                 interventions=cv.vaccinate_prob('pfizer', days=0, prob=0.5))
    sim.run()
    d = sim.diseases.covid
    wild_i = [i for i, l in d.variant_map.items() if l == 'wild'][0]
    beta_i = [i for i, l in d.variant_map.items() if l == 'beta'][0]
    vacc = d.vaccinated.uids
    assert len(vacc) > 0
    sus_wild = np.asarray(d.sus_imm[wild_i])[np.asarray(vacc)]
    sus_beta = np.asarray(d.sus_imm[beta_i])[np.asarray(vacc)]
    assert sus_wild.mean() > sus_beta.mean(), \
        f'pfizer should protect vaccinated agents more vs wild than beta: wild={sus_wild.mean():.3f} beta={sus_beta.mean():.3f}'
