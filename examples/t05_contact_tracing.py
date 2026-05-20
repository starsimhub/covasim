''' Demonstrate contact tracing '''

import covasim as cv

# Define the testing and contact tracing interventions
tp = cv.test_prob(symp_prob=0.2, asymp_prob=0.001, symp_quar_prob=1.0, asymp_quar_prob=1.0)
ct = cv.contact_tracing(trace_probs=dict(h=1.0, s=0.5, w=0.5, c=0.3))

# Define the default parameters
pars = dict(
    pop_type      = 'hybrid',
    pop_size      = 50e3,
    pop_infected  = 100,
    start_day     = '2020-02-01',
    end_day       = '2020-05-01',
    interventions = [tp, ct],
)

# Create, run, and plot the simulation
sim = cv.Sim(pars)
sim.run()
sim.plot(to_plot=['new_infections', 'new_tests', 'new_diagnoses', 'cum_diagnoses', 'new_quarantined', 'test_yield'])