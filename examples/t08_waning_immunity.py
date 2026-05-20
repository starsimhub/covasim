'''
Illustrate waning immunity
'''

import numpy as np
import covasim as cv
import pylab as pl

# Create sims with and without waning immunity
sim_nowaning = cv.Sim(n_days=120, use_waning=False, label='No waning immunity')
sim_waning   = cv.Sim(n_days=120, label='Waning immunity')

# Now create an alternative sim with faster decay for neutralizing antibodies
sim_fasterwaning = cv.Sim(
    label='Faster waning immunity',
    n_days=120,
    nab_decay=dict(form='nab_growth_decay', growth_time=21, decay_rate1=0.07, decay_time1=47, decay_rate2=0.02, decay_time2=106)
)

if __name__ == '__main__':
    msim = cv.parallel(sim_nowaning, sim_waning, sim_fasterwaning)
    msim.plot()