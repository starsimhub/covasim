''' No prognoses by age'''
import numpy as np
import covasim as cv
import covasim.parameters as cvp

sim = cv.Sim({'prog_by_age':False, 'prognoses':cvp.get_prognoses(False)})
sim.run()
for key in ['symp_prob', 'severe_prob', 'crit_prob', 'death_prob', 'rel_sus']:
    assert(len(np.unique(sim.people[key])) == 1)


sim = cv.Sim({'prog_by_age':False, 'prognoses':cvp.get_prognoses(False, cv.__version__)})
sim.run()

for key in ['symp_prob', 'severe_prob', 'crit_prob', 'death_prob', 'rel_sus']:
    assert(len(np.unique(sim.people[key])) == 1)