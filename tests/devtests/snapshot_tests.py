'''
Test the people snapshot analyzer.
'''

import covasim as cv

sim = cv.Sim(analyzers=cv.snapshot('2020-04-04', '2020-04-14'))
sim.run()
snapshot = sim['analyzers'][0]
people = snapshot.snapshots[0]            # Option 1
people = snapshot.snapshots['2020-04-04'] # Option 2
people = snapshot.get('2020-04-14')       # Option 3
people = snapshot.get(34)                 # Option 4
