'''
Simplest Covasim usage example.
'''

import covasim as cv

start_day = 30


ones  = {k:1.0 for k in 'hswc'}
zeros = {k:0.0 for k in 'hswc'}

tp = {'h':0, 's':0, 'w':1, 'c':0}

interventions = [
    cv.test_prob(start_day=start_day, symp_prob=1.0, asymp_prob=1.0, test_delay=0.0),
    cv.contact_tracing(start_day=start_day, trace_probs=tp, trace_time=zeros),
]

pars = dict(
    pop_size=1000,
    pop_infected=10,
    pop_type='hybrid',
    n_days=90,
    interventions=interventions
    )

sim = cv.Sim(pars)
sim.initialize()
# sim.people.contacts['w']['beta'] *= 0


sim.run()
sim.plot()