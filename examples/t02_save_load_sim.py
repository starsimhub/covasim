'''
Demonstration of saving and resuming a partially-run sim
'''

import covasim as cv

sim_orig = cv.Sim(start_day='2020-04-01', end_day='2020-06-01', label='Load & save example')
sim_orig.run(until='2020-05-01')
sim_orig.save('my-half-finished-sim.sim')

sim = cv.load('my-half-finished-sim.sim')
sim['beta'] *= 0.3
sim.run()
sim.plot(to_plot=['new_infections', 'n_infectious', 'cum_infections'])