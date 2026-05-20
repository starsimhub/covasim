'''
Demonstrate different transtree plotting options
'''

import covasim as cv
import sciris as sc

tstart = sc.tic()

plot_sim     = 0
verbose      = 0
simple       = 1
animated     = 1
animate_days = 1

iday = 15
pop_type = 'random'
lkeys = dict(
    random='a',
    hybrid='hswc'
    )
tp = cv.test_prob(start_day=iday, symp_prob=0.01, asymp_prob=0.0, symp_quar_prob=1.0, asymp_quar_prob=1.0, test_delay=0)
ct = cv.contact_tracing(trace_probs={k:1.0 for k in lkeys[pop_type]},
                          trace_time={k:0 for k in lkeys[pop_type]},
                          start_day=iday)

pars = dict(
    pop_size = 500,
    pop_type = pop_type,
    n_days = 60,
    contacts = 3,
    beta = 0.2,
    )

labels = ['No interventions', 'Testing only', 'Test + trace']
sims = sc.objdict()
sims.base  = cv.Sim(pars) # Baseline
sims.test  = cv.Sim(pars, interventions=tp) # Testing only
sims.trace = cv.Sim(pars, interventions=[tp, ct]) # Testing + contact tracing

tts = sc.objdict()
for key,sim in sims.items():
    sim.run()
    tts[key] = cv.TransTree(sim.people)
    if plot_sim:
        to_plot = cv.get_sim_plots()
        to_plot['Total counts']  = ['cum_infections', 'cum_diagnoses', 'cum_quarantined', 'n_quarantined']
        sim.plot(to_plot=to_plot)


#%% Plotting

if simple:
    for tt in tts.values():
        tt.plot()

if animated:
    for tt in tts.values():
        tt.animate(animate=False)

sc.toc(tstart)