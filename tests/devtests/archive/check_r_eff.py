'''
Check R_eff calculations
'''

import covasim as cv
import pylab as pl

sim = cv.Sim(n_days=90, rescale=True, pop_scale=10, pop_infected=10, rand_seed=2)
sim.run()
# sim.compute_r_eff(method='outcome')

sim.plot(to_plot='overview', n_cols=3)

# Sanity checks
a = sim.results['new_infections'].values
b = sim.results['n_infectious'].values
c = a/(b+1e-6)

pl.figure()
pl.plot(a/a.mean(), label='Number of infections (scaled)')
pl.plot(b/b.mean(), label='Number infectious (scaled)')
pl.plot(c/c.mean(), label='r_eff (scaled)')
pl.legend()