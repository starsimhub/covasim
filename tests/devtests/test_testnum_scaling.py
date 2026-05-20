'''
Compare test_num and test_prob for different scaling options to ensure consistent results.
'''

import sciris as sc
import covasim as cv

pars = dict(
    pop_type = 'hybrid',
    n_days = 60,
    rand_seed = 1,
    verbose = 0,
)

p1 = dict(
    pop_size = 200000,
    pop_infected = 500,
    pop_scale = 1,
    rescale = False,
)

p2 = dict(
    pop_size = 20000,
    pop_infected = 500,
    pop_scale = 10,
    rescale = True,
)

p3 = dict(
    pop_size = 20000,
    pop_infected = 50,
    pop_scale = 10,
    rescale = False,
)


tnp = dict(start_day=0, daily_tests=1000)
tpp = dict(symp_prob=0.1, asymp_prob=0.01)

sims = []
for st in [1000]:
    tn = cv.test_num(**tnp, symp_test=st)
    tp = cv.test_prob(**tpp)
    ti = tn # Switch testing interventions here
    s1 = cv.Sim(pars, **p1, interventions=sc.dcp(ti), label=f'full, symp_test={st}')
    s2 = cv.Sim(pars, **p2, interventions=sc.dcp(ti), label=f'dynamic, symp_test={st}')
    s3 = cv.Sim(pars, **p3, interventions=sc.dcp(ti), label=f'static, symp_test={st}')
    sims.extend([s1, s2, s3])

msim = cv.MultiSim(sims)
sc.tic()
msim.run()
sc.toc()

for sim in msim.sims:
    sim.results['n_quarantined'][:] = sim.rescale_vec
    sim.results['n_quarantined'].name = 'Rescale vec'
    sim.results['new_quarantined'] = sim.results['rel_test_yield']

fig = msim.plot(to_plot='overview')
cv.maximize()