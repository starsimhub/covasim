'''
Demonstrate custom population properties
'''

import numpy as np
import sciris as sc
import covasim as cv

def protect_the_prime(sim):
    if sim.t == sim.day('2020-04-01'):
        are_prime = sim.people.prime
        sim.people.rel_sus[are_prime] = 0.0

pars = dict(
    pop_type = 'hybrid',
    pop_infected = 100,
    n_days = 90,
    verbose = 0,
)

# Default simulation
orig_sim = cv.Sim(pars, label='Default')

# Create the simulation
sim = cv.Sim(pars, label='Protect the prime', interventions=protect_the_prime)
sim.initialize() # Initialize to create the people array
sim.people.prime = np.array([sc.isprime(i) for i in range(len(sim.people))]) # Define whom to target

# Run and plot
if __name__ == '__main__':
    msim = cv.parallel([orig_sim, sim])
    msim.plot()