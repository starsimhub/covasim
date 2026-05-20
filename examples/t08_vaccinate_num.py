'''
Illustrate sequence-based vaccination with a specified number of doses
'''

import covasim as cv
import numpy as np
import pylab as pl

# Define base parameters
pars = dict(
    beta   = 0.015,
    n_days = 90,
)

# Define the prioritization function
def prioritize_by_age(people):
    return np.argsort(-people.age)

# Record the number of doses each person has received each day so
# that we can plot the rollout in this example. Without a custom 
# analyzer, only the total number of doses will be recorded
n_doses = []

# Define sequence based vaccination
pfizer = cv.vaccinate_num(vaccine='pfizer', sequence=prioritize_by_age, num_doses=100)
sim = cv.Sim(
    pars=pars,
    interventions=pfizer,
    analyzers=lambda sim: n_doses.append(sim.people.doses.copy())
)
sim.run()

pl.figure()
n_doses = np.array(n_doses)
fully_vaccinated = (n_doses == 2).sum(axis=1)
first_dose = (n_doses == 1).sum(axis=1)
pl.stackplot(sim.tvec, first_dose, fully_vaccinated)
pl.legend(['First dose', 'Fully vaccinated'])
pl.show()