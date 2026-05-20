'''
Compare multisim combine() with a comparable single-threaded simulation.
'''

import covasim as cv

m1 = cv.MultiSim(cv.Sim(pop_size=20000, pop_infected=200, label='Single multi-core run'), n_runs=5)
m1.run()
m1.combine()

m2 = cv.MultiSim(cv.Sim(pop_size=100000, pop_infected=1000, label='Multiple single-core runs'), n_runs=5)
m2.run()
m2.reduce()

cv.MultiSim([m1.base_sim, m2.base_sim]).plot()