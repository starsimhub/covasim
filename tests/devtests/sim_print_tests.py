'''
Illustrate different ways of printing a sim.
'''

import sciris as sc
import covasim as cv

sim = cv.Sim(label='sim1', verbose=0)
sim.run()

sc.heading('print(sim)')
print(sim)

sc.heading('sim.summarize()')
sim.summarize()

sc.heading('sim.brief()')
sim.brief()

sc.heading('msim.summarize()')
msim = cv.MultiSim(sim, verbose=0)
msim.run(reduce=True)
msim.summarize()