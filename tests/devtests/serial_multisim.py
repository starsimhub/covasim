'''
Test that parallel and serial MultiSims give the same results
'''

import numpy as np
import covasim as cv

m1 = cv.MultiSim(cv.Sim())
m1.run()
m1.reduce()

m2 = cv.MultiSim(cv.Sim())
m2.run(parallel=False)
m2.reduce()

assert np.all(m1.results['cum_infections'].values == m2.results['cum_infections'].values)

m1.plot(plot_sims=True)
m2.plot(plot_sims=True)