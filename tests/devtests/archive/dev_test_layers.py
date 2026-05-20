'''
Explore different layer options
'''

#%% Basic setup

import covasim as cv
import sciris as sc

do_plot = True

basepars = sc.objdict(
    pop_size     = 2000, # Population size
    pop_infected = 10,     # Number of initial infections
    n_days       = 90,   # Number of days to simulate
    rand_seed    = 1,     # Random seed
    verbose      = 0,
)


#%% Default
pars = sc.objdict(
    pop_type     = 'random',
    use_layers   = False,
)

sim1 = cv.Sim(pars=sc.mergedicts(basepars, pars))
sim1.run(do_plot=do_plot)


#%% With default layers
pars = sc.objdict(
    pop_type     = 'random',
    use_layers   = True,
)

sim2 = cv.Sim(pars=sc.mergedicts(basepars, pars))
sim2.run(do_plot=do_plot)


#%% With layers customized to be identical to default
pars = sc.objdict(
    pop_type     = 'random',
    use_layers   = True,
    contacts     = {'a': 20},  # Number of contacts per person per day -- 'a' for 'all'
    beta_layer   = {'a': 1.0}, # Per-population beta weights; relative
    quar_eff     = {'a': 0.3}, # Multiply beta by this factor for people who know they've been in contact with a positive, even if they haven't been diagnosed yet
)

sim3 = cv.Sim(pars=sc.mergedicts(basepars, pars))
sim3.run(do_plot=do_plot)


#%% With layers customized to be the same as default
pars = sc.objdict(
    pop_type     = 'random',
    use_layers   = True,
    contacts     = {'h': 20, 's': 0, 'w': 0, 'c': 0},  # Number of contacts per person per day -- 'a' for 'all'
    beta_layer   = {'h': 1.0, 's': 0, 'w': 0, 'c': 0}, # Per-population beta weights; relative
    quar_eff     = {'h': 0.3, 's': 0, 'w': 0, 'c': 0}, # Multiply beta by this factor for people who know they've been in contact with a positive, even if they haven't been diagnosed yet
)

sim4 = cv.Sim(pars=sc.mergedicts(basepars, pars))
sim4.run(do_plot=do_plot)


#%% Results

sc.heading('Numbers of contacts')

for label,sim in {'Default':sim1, 'With layers':sim2, 'Identical':sim3, 'Almost-identical':sim4}.items():
    for key in sim.people.layer_keys():
        print(f'{label}: layer {key} of length {len(sim.people.contacts[key])}')