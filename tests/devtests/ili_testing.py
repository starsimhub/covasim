'''
Test the impact of including ILI
'''

import sciris as sc
import covasim as cv

pars = dict(
    n_days   = 90, # Not too long
    pop_size = 100e3, # Use a large population size
    beta     = 0.010, # More interesting when prevalence is lower
    pop_type = 'hybrid', # Doesn't matter
    )

# Define the interventions
start_day = 40 # Start day
tn1 = cv.test_num(start_day=start_day, daily_tests=100, symp_test=20, ili_prev=0.0)
tn2 = cv.test_num(start_day=start_day, daily_tests=100, symp_test=20, ili_prev=0.3) # Artificially high ILI prevalence
tp1 = cv.test_prob(start_day=start_day, symp_prob=0.1, asymp_prob=0.01, ili_prev=0.0)
tp2 = cv.test_prob(start_day=start_day, symp_prob=0.1, asymp_prob=0.01, ili_prev=0.3)

# Create the sims
s = sc.objdict()
s.tn1 = cv.Sim(pars, interventions=tn1, label='test_num, no ILI') # Expect 100 tests/day, high yield
s.tn2 = cv.Sim(pars, interventions=tn2, label='test_num, with ILI') # Expect 100 tests/day, low yield
s.tp1 = cv.Sim(pars, interventions=tp1, label='test_prob, no ILI') # Expect 1000 tests/day, high yield
s.tp2 = cv.Sim(pars, interventions=tp2, label='test_prob, with ILI') # Expect 2000 tests/day, low yield

# Run and plot the sims
msim = cv.MultiSim(s.values())
msim.run()
for sim in msim.sims:
    sim.plot(to_plot='overview', n_cols=3)
