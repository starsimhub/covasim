'''
Test different population options
'''

#%% Imports and settings
import pylab as pl
import sciris as sc
import covasim as cv


doplot = 1


#%% Define the tests

def test_pop_options(doplot=False): # If being run via pytest, turn off
    sc.heading('Basic populations tests')

    # Define population choices and betas
    popchoices = {'random':0.015, 'hybrid':0.015, 'synthpops':0.020}

    basepars = {
        'pop_size': 10000,
        'pop_infected': 20,
        'n_days': 90,
        }

    sims = sc.objdict()
    for popchoice,beta in popchoices.items():
        sc.heading(f'Running {popchoice}')
        sims[popchoice] = cv.Sim()
        sims[popchoice].update_pars(basepars, pop_type=popchoice, beta=beta)
        sims[popchoice].run()

    if doplot:
        for key,sim in sims.items():
            sim.plot()
            try:
                pl.gcf().axes[0].set_title(f'Counts: {key}')
            except:
                pass

    return sims



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    sims = test_pop_options(doplot=doplot)

    sc.toc()


print('Done.')
