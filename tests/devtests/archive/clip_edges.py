'''
Test the new implementation of clip_edges().
'''

import covasim as cv

pars = dict(
    n_days = 120,
    pop_type = 'hybrid',
    pop_infected = 100,
    pop_size = 50e3,
    )

s1 = cv.Sim(pars)
s2 = cv.Sim(pars)

# Create interventions
days    = [20, 40, 60, 80]
changes = [0.7, 0.5, 0.2, 1.0]
layers  = ['s','w','c']
cb = cv.change_beta(days=days, changes=changes, layers=layers)
ce = cv.clip_edges(days=days,  changes=changes, layers=layers)

s1['interventions'] = cb
s2['interventions'] = ce

msim = cv.MultiSim([s1, s2])
msim.run(verbose=0)
msim.plot(to_plot='overview')
