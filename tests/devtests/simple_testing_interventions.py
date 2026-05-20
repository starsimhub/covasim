'''
Perform (semi) simple testing tests
'''

import covasim as cv

pars = dict(
    n_days = 180,
    )

sim = cv.Sim(pars)
sim.initialize()

# Set parameters
n_people = sim['pop_size']
n_tests  = 0.1 * n_people
delay = 5
start_day = 40

# Create interventions
tn = cv.test_num(daily_tests=n_tests, symp_test=1.0, start_day=start_day, test_delay=delay, label='Testing', subtarget={'inds':lambda sim: sim.people.age>50, 'vals':1.2})
tp = cv.test_prob(symp_prob=0.1, asymp_prob=0.1, start_day=start_day, test_delay=delay)

# Create scenarios
scenarios = {
    'test_num': {
        'name': 'test_num',
        'pars': {
            'interventions': tn
            }
    },
    'test_prob': {
        'name': 'test_prob',
        'pars': {
            'interventions': tp
        }
    },
}

scens = cv.Scenarios(sim=sim, scenarios=scenarios, metapars={'n_runs': 1})
scens.run(debug=True)
scens.plot(to_plot='overview', n_cols=3)