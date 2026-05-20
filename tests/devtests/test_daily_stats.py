'''
Test daily stats analyzer
'''

import sciris as sc
import covasim as cv

T = sc.tic()

pars = dict(
    pop_type = 'hybrid',
    pop_infected = 100,
    n_days = 120,
    quar_factor = {k:0 for k in 'hswc'}
    )

tp = cv.test_prob(symp_prob=0.1, asymp_prob=0.01, symp_quar_prob=1, asymp_quar_prob=1)
ct = cv.contact_tracing(trace_probs=0.5)
sim = cv.Sim(pars, interventions=[tp, ct], analyzers=cv.daily_stats())
sim.run()
sim.plot(to_plot='overview')
ds = sim.get_analyzer()
ds.plot()

sc.toc(T)