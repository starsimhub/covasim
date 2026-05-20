'''
Test memory requirements of different simulations
'''

import covasim as cv

sim = cv.Sim()

multirun1 = cv.multi_run(sim, n_runs=50, keep_people=False) # Peak memory usage: ~100 MB
multirun2 = cv.multi_run(sim, n_runs=50, keep_people=True) # Peak memory usage: ~3 GB