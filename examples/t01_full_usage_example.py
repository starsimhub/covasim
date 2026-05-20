'''
Full Covasim usage example, including a custom intervention
'''

import covasim as cv

# Custom intervention -- see Tutorial 5
def protect_elderly(sim):
    if sim.t == sim.day('2020-04-01'):
        elderly = sim.people.age>70
        sim.people.rel_sus[elderly] = 0.0

pars = dict(
    pop_size = 50e3, # Have 50,000 people total in the population
    pop_infected = 100, # Start with 100 infected people
    n_days = 90, # Run the simulation for 90 days
    verbose = 0, # Do not print any output
)

if __name__ == '__main__':
    
    # Running with multisims -- see Tutorial 3
    s1 = cv.Sim(pars, label='Default')
    s2 = cv.Sim(pars, interventions=protect_elderly, label='Protect the elderly')
    msim = cv.MultiSim([s1, s2])
    msim.run()
    fig = msim.plot(to_plot=['cum_deaths', 'cum_infections'])