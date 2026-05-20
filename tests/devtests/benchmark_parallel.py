'''
Perform a heavy test to test parallelization
'''

import sciris as sc
import covasim as cv

T = sc.timer()

sim = cv.Sim(n_agents=100e3, n_days=100)
msim = cv.MultiSim(base_sim=sim)
msim.run(n_runs=100)

T.toc()