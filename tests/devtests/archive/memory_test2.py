import sciris as sc
import covasim as cv

cv.check_version('0.27.12')

pars = sc.objdict(
    pop_size     = 100e3, # Population size
    pop_infected = 1,     # Number of initial infections
    n_days       = 60,   # Number of days to simulate
    rand_seed    = 1,     # Random seed
    pop_type     = 'random',
    verbose      = 0,
)

sim = cv.Sim(pars=pars)

sc.tic()
sim.initialize()
sc.toc()

#%% Version 0.27.11
'''
>>> import sciris as sc
>>> import covasim as cv
Covasim 0.27.11 (2020-04-17) — © 2020 by IDM
>>> sim = cv.Sim(pop_size=1e6)
>>> sc.tic(); sim.initialize(); sc.toc()
1587187959.2373662
Initializing sim with 1e+06 people for 60 days
Elapsed time: 230 s
'''

#%% Version 0.27.12
'''
>>> import sciris as sc
>>> import covasim as cv
Covasim 0.27.12 (2020-04-17) — © 2020 by IDM
>>> sim = cv.Sim(pop_size=1e6)
>>> sc.tic(); sim.initialize(); sc.toc()
1587191076.600737
Initializing sim with 1e+06 people for 60 days
Elapsed time: 12.7 s
'''

