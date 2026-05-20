'''
Simple script for running Covasim
'''

import sciris as sc
import covasim as cv

# Run options
do_plot = 1
verbose = 1
interv  = 1

# Configure the sim -- can also just use a normal dictionary
pars = sc.objdict(
    pop_size     = 10000,    # Population size
    pop_infected = 10,       # Number of initial infections
    n_days       = 120,      # Number of days to simulate
    rand_seed    = 1,        # Random seed
    pop_type     = 'hybrid', # Population to use -- "hybrid" is random with household, school,and work structure
)

# Optionally add an intervention
if interv:
    pars.interventions = cv.change_beta(days=45, changes=0.5)

# Make, run, and plot the sim
sim = cv.Sim(pars=pars)
sim.initialize()
sim.run(verbose=verbose)
if do_plot:
    sim.plot()
