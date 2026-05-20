'''
Cruise-ship specific tests
'''

#%% Imports and settings
import pytest
import sciris as sc
import cruise_ship as cova

doplot = 1


#%% Define the tests

def test_parsobj():
    pars1 = {'a':1, 'b':2}
    parsobj = cova.ParsObj(pars1)

    # Once created, you cannot directly add new keys to a parsobj, and a nonexistent key works like a dict
    with pytest.raises(KeyError): parsobj['c'] = 3
    with pytest.raises(KeyError): parsobj['c']

    # Only a dict is allowed
    with pytest.raises(TypeError):
        pars2 = ['a', 'b']
        cova.ParsObj(pars2)

    return parsobj


def test_sim(doplot=False): # If being run via pytest, turn off

    # Settings
    seed = 1
    verbose = 1

    # Create and run the simulation
    sim = cova.Sim()
    sim.set_seed(seed)
    sim.run(verbose=verbose)

    # Optionally plot
    if doplot:
        sim.plot()

    return sim




#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    parsobj = test_parsobj()
    sim     = test_sim(doplot=doplot)

    sc.toc()


print('Done.')
