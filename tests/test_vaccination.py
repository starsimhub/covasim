"""
Tests for the vaccination interventions

See also test_immunity.py, which has more detailed tests of vaccine efficacy.
"""

#%% Imports and settings
import pytest
import sciris as sc
import covasim as cv

do_plot = 1
cv.options.set(interactive=False) # Assume not running interactively

base_pars = dict(
    pop_size = 5000,
    pop_infected = 50,
    pop_type = 'hybrid',
    n_days = 90,
    verbose = 0,
)


#%% Define the tests

def test_vaccinate(do_plot=False):
    sc.heading('Testing vaccination by probability and by number')

    # Vaccinate by probability: more than half of people are vaccinated on the first day
    prob = 0.6
    base = cv.Sim(base_pars, label='No vaccine').run()
    sim  = cv.Sim(base_pars, interventions=cv.vaccinate_prob('pfizer', days=0, prob=prob), label='Vaccine').run()
    assert base.summary['cum_doses'] == 0
    assert sim.summary['cum_vaccinated'] == pytest.approx(prob*base_pars['pop_size'], rel=0.05)
    assert sim.summary['cum_doses'] == 2*sim.summary['cum_vaccinated'] # Pfizer has two doses
    assert sim.summary['cum_infections'] < base.summary['cum_infections']

    # Vaccinate by number: a fixed number of doses each day, oldest people first
    num_doses = 100
    sim2 = cv.Sim(base_pars, interventions=cv.vaccinate_num('pfizer', num_doses=num_doses, sequence='age')).run()
    ppl = sim2.people
    assert sim2.results['new_doses'][:].max() == num_doses
    assert ppl.age[ppl.vaccinated].min() > ppl.age[~ppl.vaccinated].max()

    # cv.vaccinate() chooses between the two
    assert isinstance(cv.vaccinate('pfizer', num_doses=num_doses), cv.vaccinate_num)
    assert isinstance(cv.vaccinate('pfizer', days=10, prob=prob),  cv.vaccinate_prob)

    # These vaccines need waning immunity
    with pytest.raises(RuntimeError):
        cv.Sim(base_pars, use_waning=False, interventions=cv.vaccinate_prob('pfizer', days=0, prob=prob)).initialize()

    if do_plot:
        cv.MultiSim([base, sim]).plot(to_plot=['cum_infections', 'cum_doses'])

    return sim


def test_vaccine_targeting():
    sc.heading('Testing vaccine targeting')

    # Vaccinate only people aged 65 and over
    def over_65(sim):
        return dict(inds=cv.true(sim.people.age >= 65), vals=0.9)

    sim = cv.Sim(base_pars, n_days=30, interventions=cv.vaccinate_prob('pfizer', days=20, subtarget=over_65)).run()
    ppl = sim.people
    assert ppl.vaccinated.sum() > 0
    assert ppl.age[ppl.vaccinated].min() >= 65 # With a subtarget and no prob, only the people subtargeted are vaccinated

    # Pfizer gives more protection against the wild type than against the beta variant
    beta = cv.variant('beta', days=5, n_imports=30)
    sim2 = cv.Sim(base_pars, n_days=40, variants=beta, interventions=cv.vaccinate_prob('pfizer', days=0, prob=0.5)).run()
    ppl = sim2.people
    wild_imm, beta_imm = ppl.sus_imm[ppl.vaccinated].mean(axis=0) # One column for each variant
    assert wild_imm > beta_imm > 0

    return sim


def test_simple_vaccine():
    sc.heading('Testing simple vaccine')

    # The simple vaccine changes susceptibility directly, so it doesn't need waning immunity or NAbs
    prob = 0.6
    pars = sc.mergedicts(base_pars, use_waning=False)
    base = cv.Sim(pars).run()
    sim  = cv.Sim(pars, interventions=cv.simple_vaccine(days=0, prob=prob, rel_sus=0.2, rel_symp=0.5)).run()
    assert sim.summary['cum_vaccinated'] == pytest.approx(prob*pars['pop_size'], rel=0.05)
    assert sim.summary['cum_infections'] < base.summary['cum_infections']
    assert sim.people.peak_nab.max() == 0

    return sim


def test_historical_vaccination():
    sc.heading('Testing vaccination before the start of the sim')

    # NAbs have waned more for people who were vaccinated a year ago than a month ago
    nabs = sc.objdict()
    for label,day in dict(year=-360, month=-30).items():
        sim = cv.Sim(base_pars, n_days=5, interventions=cv.historical_vaccinate_prob('pfizer', days=[day], prob=0.5)).run()
        ppl = sim.people
        assert sim.results['new_vaccinated'][0] == ppl.vaccinated.sum() # Historical doses are counted on the first day
        assert sim.results['new_doses'][0] == 2*ppl.vaccinated.sum() # Second doses are given before the start too (Pfizer has two doses)
        nabs[label] = ppl.nab[ppl.vaccinated].mean()
    assert 0 < nabs.year < nabs.month

    # Compliance applies to each dose, and second doses due after the start are given during the sim
    sim = cv.Sim(base_pars, n_days=30, interventions=cv.historical_vaccinate_prob('pfizer', days=[-10], prob=1.0, compliance=[1.0, 0.0])).run()
    assert sim.people.doses.max() == 1 # No one takes their second dose
    sim = cv.Sim(base_pars, n_days=30, interventions=cv.historical_vaccinate_prob('pfizer', days=[-10], prob=1.0)).run()
    assert sim.people.doses.min() == 2 # The second dose is due on day 11

    return sim



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    sim1 = test_vaccinate(do_plot=do_plot)
    sim2 = test_vaccine_targeting()
    sim3 = test_simple_vaccine()
    sim4 = test_historical_vaccination()

    sc.toc(T)
    print('Done.')
