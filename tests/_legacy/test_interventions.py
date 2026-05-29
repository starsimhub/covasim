'''
Tests covering all the built-in interventions, mostly taken
from the intervention's docstrings.

See also test_immunity.py for tests of vaccines.
'''

#%% Housekeeping

import sciris as sc
import numpy as np
import pylab as pl
import covasim as cv
import pytest

verbose = -1
do_plot = 0 # Whether to plot when run interactively
cv.options.set(interactive=False) # Assume not running interactively
csv_file  = sc.thispath() / 'example_data.csv'


def test_all_interventions(do_plot=False):
    ''' Test all interventions supported by Covasim '''
    sc.heading('Testing default interventions')

    # Default parameters, using the random layer
    pars = sc.objdict(
        pop_size     = 1e3,
        scaled_pop   = 10e3,
        pop_infected = 10,
        n_days       = 90,
        verbose      = verbose,
    )
    hpars = sc.mergedicts(pars, {'pop_type':'hybrid'}) # Some, but not all, tests require layers
    rsim = cv.Sim(pars)
    hsim = cv.Sim(hpars)

    def make_sim(which='r', interventions=None):
        ''' Helper function to avoid having to recreate the sim each time '''
        if   which == 'r': sim = sc.dcp(rsim)
        elif which == 'h': sim = sc.dcp(hsim)
        sim['interventions'] = interventions
        sim.initialize()
        return sim


    #%% Define the interventions

    # 1. Dynamic pars
    i1a = cv.test_prob(start_day=5, symp_prob=0.3)
    i1b = cv.dynamic_pars({'beta':{'days':[40, 50], 'vals':[0.005, 0.015]}, 'rel_death_prob':{'days':30, 'vals':2.0}}) # Starting day 30, make diagnosed people stop transmitting


    # 2. Sequence
    i2 = cv.sequence(days=[15, 30, 45], interventions=[
                        cv.test_num(daily_tests=[20]*pars.n_days),
                        cv.test_prob(symp_prob=0.0),
                        cv.test_prob(symp_prob=0.2),
                    ])


    # 3. Change beta
    i3a = cv.change_beta([30, 50], [0.0, 1.0], layers='h')
    i3b = cv.change_beta([30, 40, 60], [0.0, 1.0, 0.5])


    # 4. Clip edges -- should match the change_beta scenarios -- note that intervention i07 was removed
    i4a = cv.clip_edges([30, 50], [0.0, 1.0], layers='h')
    i4b = cv.clip_edges([30, 40, 60], [0.0, 1.0, 0.5])


    # 5. Test number
    i5 = cv.test_num(daily_tests=[100, 100, 100, 0, 0, 0]*(pars.n_days//6))


    # 6. Test probability
    i6 = cv.test_prob(symp_prob=0.1)


    # 7. Contact tracing
    i7a = cv.test_prob(start_day=20, symp_prob=0.01, asymp_prob=0.0, symp_quar_prob=1.0, asymp_quar_prob=1.0, test_delay=0)
    i7b = cv.contact_tracing(start_day=20, trace_probs=dict(h=0.9, s=0.7, w=0.7, c=0.3), trace_time=dict(h=0, s=1, w=1, c=3))


    # 8. Combination, with dynamically set days
    def check_inf(interv, sim, thresh=10, close_day=18):
        days = close_day if sim.people.infectious.sum()>thresh else np.nan
        return days

    i8a = cv.clip_edges(days=check_inf, changes=0.0, layers='s') # Close schools
    i8b = cv.clip_edges(days=[20, 32, 45], changes=[0.7, 0.3, 0.9], layers=['w', 'c']) # Reduce work and community
    i8c = cv.test_prob(start_day=38, symp_prob=0.01, asymp_prob=0.0, symp_quar_prob=1.0, asymp_quar_prob=1.0, test_delay=2) # Start testing for TTQ
    i8d = cv.contact_tracing(start_day=40, trace_probs=dict(h=0.9, s=0.7, w=0.7, c=0.3), trace_time=dict(h=0, s=1, w=1, c=3)) # Start tracing for TTQ

    # 9. Vaccine
    i9a = cv.simple_vaccine(days=20, prob=1.0, rel_sus=1.0, rel_symp=0.0)
    i9b = cv.simple_vaccine(days=50, prob=1.0, rel_sus=0.0, rel_symp=0.0)

    #%% Create the simulations
    sims = sc.objdict()
    sims.dynamic      = make_sim('r', [i1a, i1b])
    sims.sequence     = make_sim('r', i2)
    sims.change_beta1 = make_sim('h', i3a)
    sims.clip_edges1  = make_sim('h', i4a) # Roughly equivalent to change_beta1
    sims.change_beta2 = make_sim('r', i3b)
    sims.clip_edges2  = make_sim('r', i4b) # Roughly equivalent to change_beta2
    sims.test_num     = make_sim('r', i5)
    sims.test_prob    = make_sim('r', i6)
    sims.tracing      = make_sim('h', [i7a, i7b])
    sims.combo        = make_sim('h', [i8a, i8b, i8c, i8d])
    sims.vaccine      = make_sim('r', [i9a, i9b])

    # Run the simualations
    for key,sim in sims.items():
        sim.label = key
        sim.run()

    # Test intervention retrieval methods
    sim = sims.combo
    ce1, ce2 = sim.get_interventions(cv.clip_edges)
    ce, tp = sim.get_interventions([0,2])
    inds = sim.get_interventions(cv.clip_edges, as_inds=True) # Returns [0,1]
    assert inds == [0,1]
    sim.get_interventions('summary') # Prints a summary

    # Test other intervention methods
    ce.disp()
    ce.shrink()

    #%% Plotting
    if do_plot:
        for sim in sims.values():
            print(f'Running {sim.label}...')
            sim.plot()
            fig = pl.gcf()
            try:
                fig.axes[0].set_title(f'Simulation: {sim.label}')
            except:
                pass

    return


def test_data_interventions():
    sc.heading('Testing data interventions and other special cases')

    # Create sim
    sim = cv.Sim(pop_size=100, n_days=60, datafile=csv_file, verbose=verbose)

    # Intervention conversion
    ce = cv.InterventionDict(**{'which': 'clip_edges', 'pars': {'days': [10, 30], 'changes': [0.5, 1.0]}})
    print(ce)
    with pytest.raises(sc.KeyNotFoundError):
        cv.InterventionDict(**{'which': 'invalid', 'pars': {'days': 10, 'changes': 0.5}})

    # Test numbers and contact tracing
    swab_dict = {'dist':'uniform', 'par1':1, 'par2':3}
    tp1 = cv.test_prob(0.05, start_day=5, end_day=15, ili_prev=0.1, swab_delay=swab_dict, subtarget={'inds': lambda sim: cv.true(sim.people.age>50), 'vals': 0.3})
    tn1 = cv.test_num(10, start_day=3, end_day=20, ili_prev=0.1, swab_delay=swab_dict)
    tn2 = cv.test_num(daily_tests='data', quar_policy=[0,5], subtarget={'inds': lambda sim: cv.true(sim.people.age>50), 'vals': 1.2})
    ct = cv.contact_tracing(presumptive=True, capacity=5)

    # Create and run
    sim['interventions'] = [ce, tp1, tn1, tn2, ct]
    sim.run()

    return


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    test_all_interventions(do_plot=do_plot)
    test_data_interventions()

    sc.toc(T)
    print('Done.')