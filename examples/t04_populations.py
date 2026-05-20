'''
Illustrate different population options
'''

import covasim as cv

pars = dict(
    pop_size = 10_000, # Alternate way of writing 10000
    pop_type = 'hybrid',
    location = 'Bangladesh', # Case insensitive
)

sim = cv.Sim(pars)
sim.initialize() # Create people
sim.people.plot() # Show statistics of the people