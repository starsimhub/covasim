'''
Simple script for running the Covid-19 agent-based model
'''

# Benchmark the simulation

import sciris as sc
import covasim as cv

cv.check_version('0.27.10')

def make_run_sim():
    sim = cv.Sim(n_days=180, verbose=0)
    sim.init_people()
    sim.initialize()
    sim.run()
    del sim.people.contacts
    del sim.people
    del sim.popdict
    return sim

sim = make_run_sim()

to_profile = 'run' # Must be one of the options listed below...currently only 1

sc.mprofile(run=make_run_sim, follow=make_run_sim)


#%% Example output
'''
Line #    Mem usage    Increment   Line Contents
================================================
    29    250.1 MiB    250.1 MiB   def make_run_sim():
    30    250.1 MiB      0.0 MiB       sim = cv.Sim(n_days=180, verbose=0)
    31    261.9 MiB     11.8 MiB       sim.init_people()
    32    270.7 MiB      8.9 MiB       sim.initialize()
    33    270.8 MiB      0.1 MiB       sim.run()
    34    255.6 MiB      0.0 MiB       del sim.people.contacts
    35    253.3 MiB      0.0 MiB       del sim.people
    36    253.3 MiB      0.0 MiB       del sim.popdict
    37    253.3 MiB      0.0 MiB       return sim
'''