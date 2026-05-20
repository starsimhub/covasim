'''
Test the new quarantine policy argument.
'''

import covasim as cv
import sciris as sc

benchmark = False

# Set parameters
pars = dict(
    pop_size = 50e3,
    pop_type = 'hybrid',
    rand_seed = 1,
    n_days = 365,
    pop_infected = 10,
    beta = 0.016,
    verbose = 0.1,
    iso_factor = {k:0.0 for k in 'hswc'},
    quar_factor = {k:0.0 for k in 'hswc'},
)

# To change symptomatic probability
pars['prognoses'] = cv.get_prognoses()
pars['prognoses']['symp_probs'][:] = 1

# Define interventions
sd = 30
t = sc.objdict()
t.prob = cv.test_prob(start_day=sd, symp_prob=0.06, asymp_prob=0.0006, symp_quar_prob=1.0, asymp_quar_prob=1.0)
t.num = cv.test_num(start_day=sd, daily_tests=500, symp_test=10.0, quar_test=10.0)
ct = cv.contact_tracing(start_day=sd, trace_probs=1, trace_time=0)

# Run with all options for both testing interventions
if not benchmark:
    for test_type in ['num', 'prob']:
        sims = []
        for quar_policy in ['end', 'start', 'both', 'daily']:
            ti = sc.dcp(t[test_type])
            ti.quar_policy = quar_policy
            sim = cv.Sim(pars, interventions=[ti, sc.dcp(ct)], label=f'Policy: {quar_policy}')
            sims.append(sim)

        # Run the sims
        msim = cv.MultiSim(sims)
        msim.run()
        msim.plot(to_plot='overview', fig_args={'figsize':(35,20)}, interval=120)

# Do benchmarking
else:
    sim = cv.Sim(pars, interventions=[t.prob, ct])
    sim.initialize()
    sc.profile(run=sim.run, follow=ct.apply) # 99% of time is in sim.people.trace
    sc.profile(run=sim.run, follow=sim.people.trace) # 97% of time is in np.isin(self.contacts[lkey][k1], inds)
