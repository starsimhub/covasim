''' Illustrate testing options '''

import covasim as cv

# Define the testing interventions
tn_data = cv.test_num('data')
tn_fixed = cv.test_num(daily_tests=500, start_day='2020-03-01')
tp = cv.test_prob(symp_prob=0.2, asymp_prob=0.001, start_day='2020-03-01')

# Define the default parameters
pars = dict(
    pop_size = 50e3,
    pop_infected = 100,
    start_day = '2020-02-01',
    end_day   = '2020-03-30',
)

# Define the simulations
sim1 = cv.Sim(pars, datafile='example_data.csv', interventions=tn_data, label='Number of tests from data')
sim2 = cv.Sim(pars, interventions=tn_fixed, label='Constant daily number of tests')
sim3 = cv.Sim(pars, interventions=tp, label='Probability-based testing')

# Run and plot results
msim = cv.parallel(sim1, sim2, sim3)
msim.plot(to_plot=['new_infections', 'new_tests', 'new_diagnoses', 'cum_diagnoses'])