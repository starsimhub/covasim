'''
Confirm prevalence and incidence calculations
'''

import covasim as cv
import pylab as pl

sim = cv.Sim(n_days=120, rescale=True, pop_scale=10)
sim.run()

to_plot = ['n_exposed', 'new_infections', 'prevalence', 'incidence']
sim.plot(to_plot=to_plot)

# Sanity checks
a = sim.results['new_infections'].values
b = sim.results['n_susceptible'].values
c = a/b

pl.figure()
pl.plot(a/a.mean())
pl.plot(b/b.mean())
pl.plot(c/c.mean())