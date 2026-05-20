'''
Try out sim.people.plot() method
'''

import covasim as cv

use_synthpops = False

if use_synthpops:
    import synthpops as sp
    sp.config.set_nbrackets(20)
    pop_type = 'synthpops'
else:
    pop_type = 'hybrid'

sim = cv.Sim(pop_size=20000, pop_type=pop_type, verbose=0)
sim.initialize()
sim.people.plot()
