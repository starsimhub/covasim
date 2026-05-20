'''
Demonstrate Plotly usage
'''

import covasim as cv

do_plot   = 0
do_plotly = 1
interv    = 0

# Configure the sim -- can also just use a normal dictionary
pars = dict(
    pop_size     = 5000, # Population size
    pop_infected = 10,     # Number of initial infections
    n_days       = 60,   # Number of days to simulate
    rand_seed    = 1,     # Random seed
    pop_type     = 'random',
)

# Optionally add an intervention
if interv:
    pars.interventions = cv.change_beta(days=45, changes=0.5) # Optionally add an intervention

# Make and run the sim
sim = cv.Sim(pars=pars)
sim.run()

if do_plot:
    print('Plotting with Matplotlib...')
    fig = sim.plot()

if do_plotly:
    print('Plotting with plotly...')
    fig1 = cv.plot_people(sim)
    fig1.show(renderer='browser')
    fig2 = cv.animate_people(sim)
    fig2.show(renderer='browser')