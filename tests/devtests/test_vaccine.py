'''
Demonstrate the vaccine intervention
'''

import covasim as cv
import sciris as sc
import numpy as np


# Basic usage
v1 = cv.vaccine(days=20, prob=1.0, rel_sus=1.0, rel_symp=0.0) # Prevent symptoms but not transmission
v2 = cv.vaccine(days=50, prob=1.0, rel_sus=0.0, rel_symp=0.0) # Prevent both
sim = cv.Sim(interventions=[v1,v2])
sim.run()
sim.plot()

# Age targeting
def target_over_65(sim, prob=1.0):
    ''' Subtarget '''
    inds = sc.findinds(sim.people.age>65)
    vals = prob*np.ones(len(inds))
    return {'inds':inds, 'vals':vals}

v3 = cv.vaccine(days=20, prob=0.0, rel_sus=0.5, rel_symp=0.2, subtarget=target_over_65) # Age targeting
sim2 = cv.Sim(label='Status quo', n_days=180)
sim3 = cv.Sim(label='Vaccinate over 65s', n_days=180, interventions=v3)
msim = cv.MultiSim([sim2, sim3])
msim.run()
msim.plot()

sim4 = cv.Sim(interventions=cv.vaccine(days=[10,20,30,40], prob=0.8, rel_sus=0.5, cumulative=[1, 0.5, 0.5, 0]))
sim4.run()
sim4.plot()