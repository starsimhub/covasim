'''
Illustrate boosters
'''

import covasim as cv
import numpy as np
import pylab as pl

# Define base parameters
pars = dict(
    beta   = 0.015,
    n_days = 90,
)

# Define a function to specify the number of doses
def num_doses(sim):
    if sim.t < 50:
        return sim.t*10
    else:
        return 500

# Define the number of boosters
def num_boosters(sim):
    if sim.t < 50: # None over the first 50 days
        return 0
    else:
        return 50 # Then 100/day thereafter

# Define base vaccine
pfizer = cv.vaccinate_num(vaccine='pfizer', sequence='age', num_doses=num_doses)

# Only give boosters to people who have had 2 doses
booster_target  = {'inds': lambda sim: cv.true(sim.people.doses != 2), 'vals': 0}
booster = cv.vaccinate_num(vaccine='pfizer', sequence='age', subtarget=booster_target, booster=True, num_doses=num_boosters)

# Track doses
n_doses = []
n_doses_boosters = []

# Create simple sim
sim = cv.Sim(
    pars          = pars,
    interventions = pfizer,
)
sim.run()

# Create a sim with boosters
sim_booster = cv.Sim(
    pars          = pars,
    interventions = [pfizer, booster],
    label         = 'With booster',
    analyzers     = lambda sim: n_doses_boosters.append(sim.people.doses.copy())
)
sim_booster.run()

# Plot the sims with and without boosters together
cv.MultiSim([sim, sim_booster]).plot(to_plot=['cum_infections', 'cum_severe', 'cum_deaths','pop_nabs'])

# Plot doses again
pl.figure()
n_doses = np.array(n_doses_boosters)
fully_vaccinated = (n_doses == 2).sum(axis=1)
first_dose = (n_doses == 1).sum(axis=1)
boosted = (n_doses > 2).sum(axis=1)
pl.stackplot(sim.tvec, first_dose, fully_vaccinated, boosted)
pl.legend(['First dose', 'Fully vaccinated', 'Boosted'], loc='upper left');
pl.show()