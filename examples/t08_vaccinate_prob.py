'''
Illustrate simple vaccine usage
'''

import covasim as cv

# Create some base parameters
pars = dict(
    beta   = 0.015,
    n_days = 90,
)

# Define probability based vaccination
pfizer = cv.vaccinate_prob(vaccine='pfizer', days=20, prob=0.8)

# Create and run the sim
sim = cv.Sim(pars=pars, interventions=pfizer)
sim.run()
sim.plot(['new_infections', 'cum_infections', 'new_doses', 'cum_doses'])