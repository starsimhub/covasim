'''
Runs a fairly complex analysis using a single line of code. Equivalent to:

    pars = dict(
        pop_size     = 1e3,
        pop_infected = 10,
        pop_type     = 'hybrid',
        n_days       = 180,
    )
    cb = cv.change_beta([30, 50], [0.0, 1.0], layers=['w','c'])

    sim = cv.Sim(**pars, interventions=cb)
    sim.run()
    sim.plot(to_plot='seir')
'''
import covasim as cv

cv.Sim(pop_size=1e3, pop_infected=10, pop_type='hybrid', n_days=180, interventions=cv.change_beta([30, 50], [0.0, 1.0], layers=['w','c'])).run().plot(to_plot='seir')