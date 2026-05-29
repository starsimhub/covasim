'''
Tests for run options (multisims and scenarios)
'''

#%% Imports and settings
import os
import numpy as np
import sciris as sc
import covasim as cv

do_plot = 1
do_save = 0
debug   = 1
verbose = 0
pop_size = 500
cv.options.set(interactive=False) # Assume not running interactively


#%% Define the tests

def test_singlerun():
    sc.heading('Single run test')

    iterpars = {'beta': 0.035}
    sim = cv.Sim(verbose=verbose)
    sim['n_days'] = 20
    sim['pop_size'] = 1000
    sim = cv.single_run(sim=sim, **iterpars)

    return sim


def test_multirun(do_plot=do_plot): # If being run via pytest, turn off
    sc.heading('Multirun test')

    n_days = 60

    # Method 1 -- Note: this runs 3 simulations, not 3x3!
    iterpars = {'beta': [0.015, 0.025, 0.035],
                'iso_factor': [0.1, 0.5, 0.9],
                }
    sim = cv.Sim(n_days=n_days, pop_size=pop_size)
    sims = cv.multi_run(sim=sim, iterpars=iterpars, verbose=verbose)

    # Method 2 -- run a list of sims
    simlist = []
    for i in range(len(iterpars['beta'])):
        sim = cv.Sim(n_days=n_days, pop_size=pop_size, beta=iterpars['beta'][i], iso_factor=iterpars['iso_factor'][i])
        simlist.append(sim)
    sims2 = cv.multi_run(sim=simlist, verbose=verbose)

    # Method 3 -- shortcut for parallelization
    s1 = cv.Sim(n_days=n_days, pop_size=pop_size)
    s2 = s1.copy()
    s1,s2 = cv.parallel(s1, s2).sims
    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    # Run in serial for debugging
    cv.multi_run(sim=cv.Sim(n_days=n_days, pop_size=pop_size), n_runs=2, parallel=False)

    if do_plot:
        for sim in sims + sims2:
            sim.plot()

    return sims


def test_multisim_reduce(do_plot=do_plot): # If being run via pytest, turn off
    sc.heading('Combine results test')

    n_runs = 3
    pop_infected = 10

    sim = cv.Sim(pop_size=pop_size, pop_infected=pop_infected)
    msim = cv.MultiSim(sim, n_runs=n_runs, noise=0.1)
    msim.run(verbose=verbose, reduce=True)

    if do_plot:
        msim.plot()

    return msim


def test_multisim_combine(do_plot=do_plot): # If being run via pytest, turn off
    sc.heading('Combine results test')

    n_runs = 3
    pop_infected = 10

    print('Running first sim...')
    sim = cv.Sim(pop_size=pop_size, pop_infected=pop_infected, verbose=verbose)
    msim = cv.MultiSim(sim)
    msim.run(n_runs=n_runs, keep_people=True)
    sim1 = msim.combine(output=True)
    assert sim1['pop_size'] == pop_size*n_runs

    print('Running second sim, results should be similar but not identical (stochastic differences)...')
    sim2 = cv.Sim(pop_size=pop_size*n_runs, pop_infected=pop_infected*n_runs)
    sim2.run(verbose=verbose)

    if do_plot:
        msim.plot()
        sim2.plot()

    return msim


def test_multisim_advanced():
    sc.heading('Advanced multisim options')

    # Settings
    msim_path = 'msim_test.msim'

    # Creat the sims/msims
    sims = sc.objdict()
    for i in range(4):
        sims[f's{i}'] = cv.Sim(label=f'Sim {i}', pop_size=pop_size, beta=0.01*i)

    m1 = cv.MultiSim(sims=[sims.s0, sims.s1])
    m2 = cv.MultiSim(sims=[sims.s2, sims.s3])

    # Test methods
    m1.init_sims()
    m1.run()
    m2.run(reduce=True)
    m1.reduce()
    m1.mean()
    m1.median()
    m1.shrink()
    m1.disp()
    m1.summarize()
    m1.brief()

    # Check save/load
    m1.save(msim_path)
    m1b = cv.MultiSim.load(msim_path)
    assert np.allclose(m1.summary[:], m1b.summary[:], rtol=0, atol=0, equal_nan=True)
    os.remove(msim_path)

    # Check merging/splitting
    merged1 = cv.MultiSim.merge(m1, m2)
    merged2 = cv.MultiSim.merge([m1, m2], base=True)
    m1c, m2c = merged1.split()
    m1d, m2d = merged1.split(chunks=[2,2])

    return merged1, merged2


def test_simple_scenarios(do_plot=do_plot):
    sc.heading('Simple scenarios test')
    basepars = {'pop_size':pop_size}

    json_path = 'scen_test.json'
    xlsx_path = 'scen_test.xlsx'

    scens = cv.Scenarios(basepars=basepars)
    scens.run(verbose=verbose)
    if do_plot:
        scens.plot()
    scens.to_json(json_path)
    scens.to_excel(xlsx_path)
    scens.disp()
    scens.summarize()
    scens.brief()

    for path in [json_path, xlsx_path]:
        print(f'Removing {path}')
        os.remove(path)

    return scens


def test_complex_scenarios(do_plot=do_plot, do_save=False, fig_path=None):
    sc.heading('Test impact of reducing delay time for finding contacts of positives')

    n_runs = 3
    base_pars = {
      'pop_size': pop_size,
      'pop_type': 'hybrid',
      }

    base_sim = cv.Sim(base_pars) # create sim object
    base_sim['n_days'] = 50
    base_sim['beta'] = 0.03 # Increase beta

    n_people = base_sim['pop_size']
    npts = base_sim.npts

    # Define overall testing assumptions
    testing_prop = 0.1 # Assumes we could test 10% of the population daily (way too optimistic!!)
    daily_tests = [testing_prop*n_people]*npts # Number of daily tests

    # Define the scenarios
    scenarios = {
        'lowtrace': {
            'name': 'Poor contact tracing',
            'pars': {
                'quar_factor': {'h': 1, 's': 0.5, 'w': 0.5, 'c': 0.25},
                'quar_period': 7,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 0, 's': 0, 'w': 0, 'c': 0},
                        trace_time  = {'h': 1, 's': 7,   'w': 7,   'c': 7})]
            }
        },
        'modtrace': {
            'name': 'Moderate contact tracing',
            'pars': {
                'quar_factor': {'h': 0.75, 's': 0.25, 'w': 0.25, 'c': 0.1},
                'quar_period': 10,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.5, 'c': 0.1},
                        trace_time  = {'h': 0,  's': 3,  'w': 3,   'c': 8})]
            }
        },
        'hightrace': {
            'name': 'Fast contact tracing',
            'pars': {
                'quar_factor': {'h': 0.5, 's': 0.1, 'w': 0.1, 'c': 0.1},
                'quar_period': 14,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.8, 'c': 0.2},
                        trace_time  = {'h': 0, 's': 1,   'w': 1,   'c': 5})]
            }
        },
        'alltrace': {
            'name': 'Same-day contact tracing',
            'pars': {
                'quar_factor': {'h': 0.0, 's': 0.0, 'w': 0.0, 'c': 0.0},
                'quar_period': 21,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 1, 's': 1, 'w': 1, 'c': 1},
                        trace_time  = {'h': 0, 's': 1, 'w': 1, 'c': 2})]
            }
        },
    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)
    scens.compare()

    if do_plot:
        to_plot = [
            'cum_infections',
            'cum_recoveries',
            'new_infections',
            'cum_severe',
            'n_quarantined',
            'new_quarantined'
        ]
        fig_args = dict(figsize=(24,16))
        scens.plot(do_save=do_save, to_plot=to_plot, fig_path=fig_path, n_cols=2, fig_args=fig_args)

    return scens


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    sim1   = test_singlerun()
    sims2  = test_multirun(do_plot=do_plot)
    msim1  = test_multisim_reduce(do_plot=do_plot)
    msim2  = test_multisim_combine(do_plot=do_plot)
    m1,m2  = test_multisim_advanced()
    scens1 = test_simple_scenarios(do_plot=do_plot)
    scens2 = test_complex_scenarios(do_plot=do_plot)

    sc.toc(T)
    print('Done.')
