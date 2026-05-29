'''
Tests for single simulations
'''

#%% Imports and settings
import os
import pytest
import sciris as sc
import covasim as cv

do_plot = 1
do_save = 0
cv.options.set(interactive=False) # Assume not running interactively


#%% Define the tests

def test_parsobj():
    sc.heading('Testing parameters object')

    pars1 = {'a':1, 'b':2}
    parsobj = cv.ParsObj(pars1)

    # Once created, you cannot directly add new keys to a parsobj, and a nonexistent key works like a dict
    with pytest.raises(KeyError): parsobj['c'] = 3
    with pytest.raises(KeyError): parsobj['c']

    # Only a dict is allowed
    with pytest.raises(TypeError):
        pars2 = ['a', 'b']
        cv.ParsObj(pars2)

    return parsobj


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
    sim.summarize()
    sim.brief()

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

    for path in [json_path, xlsx_path]:
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
    return sim



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    pars = test_parsobj()
    sim0 = test_microsim()
    sim1 = test_sim(do_plot=do_plot, do_save=do_save)
    json = test_fileio()
    sim2 = test_sim_data(do_plot=do_plot)
    sim3 = test_dynamic_resampling(do_plot=do_plot)

    sc.toc(T)
    print('Done.')
