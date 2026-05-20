'''
Simplest Covasim usage example.
'''

import covasim as cv

debug = False # Set to True to raise exception on warnings
if debug:
    import warnings
    warnings.simplefilter("error")

sim = cv.Sim(pop_size=1000, pop_infected=10, n_days=90)
sim.run()
sim.plot()