import os
import sciris  as sc; sc
import covasim as cv; cv

popfile = 'example.pop'


sim = cv.Sim(pop_type='random')
sim.initialize(save_pop=True, popfile=popfile)
sim.initialize(load_pop=True, popfile=popfile)
sim.run()

sim2 = cv.Sim(pop_type='synthpops')
sim2.initialize(save_pop=True, popfile=popfile)
sim2.initialize(load_pop=True, popfile=popfile)
sim2.run()

sim3 = cv.Sim(pop_type='hybrid')
sim3.initialize(save_pop=True, popfile=popfile)
sim3.initialize(load_pop=True, popfile=popfile)
sim3.run()

os.remove(popfile)