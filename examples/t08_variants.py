'''
Illustrate multiple variants
'''

import covasim as cv

# Define three new variants: B117, B1351, and a custom-defined variant
alpha  = cv.variant('alpha', days=0, n_imports=10)
beta   = cv.variant('beta', days=0, n_imports=10)
custom = cv.variant(label='3x more transmissible', variant={'rel_beta': 3.0}, days=7, n_imports=10)

# Create the simulation
sim = cv.Sim(variants=[alpha, beta, custom], pop_infected=10, n_days=32)

# Run and plot
sim.run()
sim.plot('variant')