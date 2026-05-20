import covasim as cv

pop_size = 1000 # Default number of people
do_plot = True

# Random network using defaults
sim = cv.Sim(pop_size=pop_size)
sim.run(do_plot=do_plot)

# Random network with specified beta and people
sim = cv.Sim(beta=0.01, pop_size=2000)
sim.run(do_plot=do_plot)

# Default is to use age-specific progression. Here we modify the parameters to remove age variation
sim = cv.Sim(pop_size=pop_size, prog_by_age=False)
sim.run(do_plot=do_plot)

# Make and use a random population (including community transmission)
sim = cv.Sim(pop_size=pop_size, prog_by_age=False, use_layers=True)
sim.run(do_plot=do_plot)

# Add an intervention changing the overall beta
pars = dict(
    pop_size = pop_size,
    interventions = cv.interventions.change_beta(25, 0.5),
    )
sim = cv.Sim(pars)
sim.run(do_plot=do_plot)

# Change beta only for community contacts
pars = dict(
    pop_size = pop_size,
    use_layers = True,
    interventions = cv.interventions.change_beta(25, 0.0, 'c'),
    )
sim = cv.Sim(pars)
sim.run(do_plot=do_plot)

# Run sim using synthpops with community transmission
try:
    pars = dict(
        pop_size = 5000,
        pop_type = 'synthpops',
        use_layers = True,
        )
    sim = cv.Sim(pars)
    sim.run(do_plot=do_plot)
except Exception as E:
    print(f'Could not use synthpops: {str(E)})')
