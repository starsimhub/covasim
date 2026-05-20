'''
Test the fitting to data
'''

import covasim as cv

regenerate = 0
datafile = 'target_fit_data.xlsx'
# datafile = '../example_data.csv'

intervs = [cv.change_beta(days=40, changes=0.5), cv.test_prob(start_day=20, symp_prob=0.1, asymp_prob=0.01)] # Common interventions
pars = dict(
    pop_size      = 20000,    # Population size
    pop_infected  = 80,      # Number of initial infections -- use more for increased robustness
    pop_type      = 'hybrid', # Population to use -- "hybrid" is random with household, school,and work structure
    verbose       = 0,        # Don't print details of the run
    interventions = intervs,   # Include the most common interventions
    rand_seed     = 2,
    beta          = 0.017,
)

target_pars = dict(
    pop_infected  = 80,
    beta          = 0.017,
    rand_seed     = 294873,
)

if regenerate:
    target = cv.Sim(pars)
    target.update_pars(target_pars)
    target.run()
    target.to_excel(datafile)

sim = cv.Sim(pars, datafile=datafile)
sim.run()
fit = sim.compute_fit()
# fit = sim.compute_fit(keys=['cum_deaths', 'cum_diagnoses', 'new_infections', 'new_diagnoses'])
print('Mismatch: ', fit.mismatch)
fit.plot()