'''
Simple Covasim usage
'''

import covasim as cv

# Set the parameters of the simulation
pars = dict(
    pop_size = 50e3,
    pop_infected = 100,
    start_day = '2020-04-01',
    end_day = '2020-06-01',
)

# Run the simulation
sim = cv.Sim(pars)
sim.run()

# Plot the results
fig = sim.plot()