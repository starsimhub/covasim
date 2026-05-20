'''
Test multisim plotting options
'''

import covasim as cv
import numpy as np
import sciris as sc


n = 100
betas = np.linspace(0.005, 0.030, n)
sims = []
for beta in betas:
    sim = cv.Sim(pop_size=1000, beta=beta, datafile='../example_data.csv')
    sims.append(sim)
msim = cv.MultiSim(sims)
msim.run(reseed=True)

# Demonstrate indices
msim.plot(inds=[10, 20, 30])

# Demonstrate and time lots of lines
sc.tic()
msim.plot()
sc.toc()

# Run with profiling
sc.profile(run=msim.plot, follow=cv.plotting.plot_sim)