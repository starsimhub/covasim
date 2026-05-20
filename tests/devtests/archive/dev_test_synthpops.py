'''
Test different population options
'''

#%% Imports and settings
import pytest
import pylab as pl
import sciris as sc
import covasim as cova
try:
    import synthpops as sp
except:
    print('Synthpops import failed, you are not going to be able to run this')

doplot = 1


#%% Define the tests

def test_import():
    sc.heading('Testing imports')

    assert cova.requirements.available['synthpops'] == True
    import synthpops as sp
    print(sp.datadir)

    return


def test_pop_options(doplot=False): # If being run via pytest, turn off
    sc.heading('Basic populations tests')

    popchoices = ['random', 'bayesian']
    if sp.config.full_data_available:
        popchoices.append('data')

    basepars = {
        'n': 3000,
        'n_infected': 10,
        'contacts': 20,
        'n_days': 90
        }

    sims = sc.objdict()
    for popchoice in popchoices:
        sc.heading(f'Running {popchoice}')
        sims[popchoice] = cova.Sim()
        sims[popchoice].update_pars(basepars)
        sims[popchoice]['usepopdata'] = popchoice
        sims[popchoice].run()

    if doplot:
        for key,sim in sims.items():
            sim.plot()
            try:
                pl.gcf().axes[0].set_title(f'Counts: {key}')
            except:
                pass

    return sims


def test_interventions(doplot=False): # If being run via pytest, turn off
    sc.heading('Test interventions')

    popchoice = 'bayesian'
    intervs = ['none', 'all', 'HSWR', 'SWR', 'H']
    interv_days = [21]

    basepars = {
        'n': 10000,
        'n_infected': 100,
        'n_days': 60,
        'usepopdata': popchoice,
        }

    def interv_func(sim, t, interv, interv_days):
        if t in interv_days:
            print(f'Applying custom intervention/change on day {t}...')
            if   interv == 'none':   sim['beta'] *= 1.0
            elif interv == 'all':    sim['beta'] *= 0.1
            else:
                for key in interv: sim['beta_pop'][key] = 0
        return sim

    # Create the base sim and initialize (since slow)
    base_sim = cova.Sim()
    base_sim.update_pars(basepars)
    base_sim.initialize()

    # Run the sims
    sims = sc.objdict()
    for interv in intervs:
        sc.heading(f'Running {interv}')
        interv_lambda = lambda sim,t: interv_func(sim=sim, t=t, interv=interv, interv_days=interv_days)
        sims[interv] = sc.dcp(base_sim)
        sims[interv]['interv_func'] = interv_lambda
        sims[interv].run(initialize=False) # Since already initialized

    if doplot:
        for key,sim in sims.items():
            sim.plot()
            pl.gcf().axes[0].set_title(f'Counts: {key}')

    return sims


def test_simple_interv(doplot=False): # If being run via pytest, turn off
    sc.heading('Test simple intervention')

    def close_schools(sim, t):
        if t == 10:
            print(f'Closing schools on day {t}...')
            sim['beta_pop']['S'] = 0
        return sim

    basepars = {
        'n':           2000,
        'n_infected':  100,
        'n_days':      60,
        'interv_func': close_schools,
        'usepopdata':  'bayesian',
        }

    sim = cova.Sim()
    sim.update_pars(basepars)
    sim.run()

    if doplot:
        sim.plot()

    return sim



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    test_import()
    sims1 = test_pop_options(doplot=doplot)
    sims2 = test_interventions(doplot=doplot)
    sims3 = test_simple_interv(doplot=doplot)

    sc.toc()


print('Done.')
