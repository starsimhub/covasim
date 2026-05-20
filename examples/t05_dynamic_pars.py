'''
Demonstrate dynamic parameters
'''

import covasim as cv

# Define the dynamic parameters
imports = cv.dynamic_pars(n_imports=dict(days=[15, 30], vals=[100, 0]))

# Create, run, and plot the simulations
sim1 = cv.Sim(label='Baseline')
sim2 = cv.Sim(interventions=imports, label='With imported infections')
msim = cv.parallel([sim1, sim2])
msim.plot()