"""
Tests for single simulations
"""

#%% Imports and settings
import os
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import covasim as cv

do_plot = 1
do_save = 0
cv.options.set(interactive=False) # Assume not running interactively


#%% Define the tests

def test_microsim():
    sc.heading('Minimal sim test')

    sim = cv.Sim()
    pars = {
        'pop_size': 10,
        'pop_infected': 1,
        'n_days': 10,
        'contacts': 2,
        }
    sim.update_pars(pars)
    sim.run()
    sim.disp()
    sim.summarize(full=True)
    sim.brief()
    assert sim.compute_summary(t=5, output=True)['cum_infections'] == sim.results['cum_infections'][5] # The summary is for the requested day
    assert 'Simulation summary' in sim.summarize(output=True) # As in v3, the printed summary

    return sim


def test_sim(do_plot=False, do_save=False): # If being run via pytest, turn off
    sc.heading('Basic sim test')

    # Settings
    seed = 1
    verbose = 1

    # Create and run the simulation
    sim = cv.Sim()
    sim.set_seed(seed)
    sim.run(verbose=verbose)

    # Optionally plot
    if do_plot:
        sim.plot(do_save=do_save)

    return sim


def test_fileio():
    sc.heading('Test file saving')

    json_path = 'test_covasim.json'
    xlsx_path = 'test_covasim.xlsx'
    sim_path  = 'test_covasim.sim'

    # Create and run the simulation
    sim = cv.Sim()
    sim['n_days'] = 20
    sim['pop_size'] = 1000
    sim.run(verbose=0)

    # Create objects
    json = sim.to_json()
    xlsx = sim.to_excel()
    print(xlsx)

    # Save files
    sim.to_json(json_path)
    sim.to_excel(xlsx_path)
    sim.save(sim_path)

    # Check that a loaded sim has the same results
    sim2 = cv.load(sim_path)
    assert sim2.summary == sim.summary

    for path in [json_path, xlsx_path, sim_path]:
        print(f'Removing {path}')
        os.remove(path)

    return json


def test_sim_data(do_plot=False):
    sc.heading('Data test')

    pars = dict(
        pop_size = 2000,
        start_day = '2020-02-25',
        )

    # Create and run the simulation
    sim = cv.Sim(pars=pars, datafile=os.path.join(sc.thisdir(__file__), 'example_data.csv'))
    sim.run()

    # Optionally plot
    if do_plot:
        sim.plot()

    return sim


def test_dynamic_resampling(do_plot=False): # If being run via pytest, turn off
    sc.heading('Test dynamic resampling')

    pop_size = 1000
    sim = cv.Sim(pop_size=pop_size, rescale=1, pop_scale=1000, n_days=180, rescale_factor=2)
    sim.run()

    # Optionally plot
    if do_plot:
        sim.plot()

    # Create and run a basic simulation
    assert sim.results['cum_infections'][-1] > pop_size  # infections at the end of sim should be much more than internal pop

    # Without dynamic rescaling, numbers of people are multiplied by pop_scale, but proportions aren't
    pars = dict(pop_size=pop_size, n_days=30, rescale=False, verbose=0)
    s1 = cv.Sim(pars).run()
    s2 = cv.Sim(pars, pop_scale=10).run()
    assert s2.summary['cum_infections'] == s1.summary['cum_infections']*10
    assert s2.summary['prevalence'] == s1.summary['prevalence']

    return sim


def test_people():
    sc.heading('Test people')

    pop_size = 5000
    pop_infected = 100
    sim = cv.Sim(pop_size=pop_size, pop_infected=pop_infected, n_days=60, verbose=0)
    sim.initialize()
    ppl = sim.people
    assert ppl.exposed.sum() == pop_infected # Exactly this many people are infected to start with...
    assert ppl.infectious.sum() == 0 # ...but they aren't infectious yet

    sim.run()
    n_dead = sim.summary['cum_deaths']
    assert n_dead > 0

    # As in Starsim, people who have died are removed from the arrays, except for people.dead and people.date_dead
    assert len(ppl) == len(ppl.age) == len(ppl.exposed) == pop_size - n_dead
    assert len(ppl.dead) == len(ppl.date_dead) == pop_size
    assert len(ppl.age.raw) == pop_size # Use .raw to include people who have died
    assert ppl.dead.sum() == n_dead
    assert len(ppl.age[cv.true(ppl.dead)]) == n_dead # Indexing by UID works for everyone

    # Ages depend on the location
    ages = sc.objdict()
    for location in ['japan', 'bangladesh']:
        ages[location] = cv.Sim(pop_size=pop_size, location=location, verbose=0).initialize().people.age.mean()
    assert ages.japan > ages.bangladesh

    return sim


def test_results():
    sc.heading('Test results')

    n_imports = 5
    sim = cv.Sim(pop_size=5000, pop_infected=0, n_imports=n_imports, n_days=60, verbose=0)
    sim.run()
    res = sim.results

    # With no initial infections, the epidemic starts from the imported infections
    assert np.isclose(res['n_imports'][:].mean(), n_imports, rtol=0.3) # n_imports is the average number per day
    assert res['cum_infections'][-1] > res['n_imports'][:].sum()
    assert res['r_eff'][:].max() > 1 # The epidemic is growing

    # Each stage of disease has fewer people than the one before
    assert res['cum_infections'][-1] > res['cum_symptomatic'][-1] > res['cum_severe'][-1] > res['cum_critical'][-1] > res['cum_deaths'][-1] > 0

    # Cumulative results are the sum of the new ones
    for key in ['infections', 'severe', 'deaths']:
        assert np.allclose(res[f'cum_{key}'], np.cumsum(res[f'new_{key}']))

    return sim


def test_starsim():
    sc.heading('Test using Starsim modules in a Covasim sim')

    # Starsim modules can be added to a sim, e.g. births
    pop_size = 1000
    sim = cv.Sim(pop_size=pop_size, n_days=30, demographics=ss.Births(birth_rate=ss.peryear(200)), verbose=0)
    sim.run()
    assert sim.people.n_uids > pop_size # People have been born

    # The COVID module can be created directly, with Starsim parameters
    covid = cv.COVID(beta=ss.probperday(0.02), init_prev=ss.bernoulli(p=0.02))
    sim2 = cv.Sim(pop_size=pop_size, n_days=30, diseases=covid, verbose=0)
    sim2.run()
    assert sim2.summary['cum_infections'] > 0

    # Starsim time arguments, and sim-level COVID arguments that can't be combined with a user-supplied disease
    assert cv.Sim(n_agents=pop_size, dur=ss.years(1))['n_days'] == 365 # Converted to days
    with pytest.raises(ValueError):
        cv.Sim(diseases=cv.COVID(), pop_infected=50) # Would otherwise be silently ignored

    # sim.new() takes the v3 names of the Starsim settings
    sim3 = sim2.new(pop_size=2*pop_size, end_day='2020-03-11').init()
    assert len(sim3.people) == 2*pop_size and sim3['n_days'] == 10

    return sim



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    sim0 = test_microsim()
    sim1 = test_sim(do_plot=do_plot, do_save=do_save)
    json = test_fileio()
    sim2 = test_sim_data(do_plot=do_plot)
    sim3 = test_dynamic_resampling(do_plot=do_plot)
    sim4 = test_people()
    sim5 = test_results()
    sim6 = test_starsim()

    sc.toc(T)
    print('Done.')
