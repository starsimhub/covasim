import covasim as cv

def plot_all(sim):
    import pylab as pl
    to_plot = ['cum_infections',
             'cum_diagnoses',
             'cum_deaths',
             'cum_quarantined',
             'new_infections',
             'new_tests',
             'new_diagnoses',
             'new_deaths',
             'new_quarantined',
             'n_infectious',
             'n_quarantined',
             'r_eff']
    sim.plot(to_plot=to_plot, n_cols=3, fig_args=dict(figsize=(40,20)))
    pl.axhline(y=1)
    return

pars = dict(
    pop_size = 20e3,
    pop_type = 'hybrid',
    rescale = False,
    n_days = 300,
    )

simple = False

if simple:

    intervs = [
        cv.test_prob(start_day=20, symp_prob=1.0, asymp_prob=0.0, symp_quar_prob=0.0, asymp_quar_prob=0.0, test_delay=1.0),
        cv.contact_tracing(start_day=20, trace_probs=1.0, trace_time=1.0, presumptive=True),
        ]

    sim = cv.Sim(pars, interventions=intervs)
    sim.run()
    plot_all(sim)

else:

    sims = []
    for aqp in [0, 1.0]:
        for pr in [False, True]:

            intervs = [
                cv.test_prob(start_day=10, symp_prob=0.1, asymp_prob=0.001, symp_quar_prob=aqp, asymp_quar_prob=aqp, test_delay=1.0, do_plot=False),
                cv.contact_tracing(start_day=10, trace_probs=dict(h=0.9, s=0.7, w=0.7, c=0.3), trace_time=2.0, presumptive=pr, do_plot=False),
                ]

            sim = cv.Sim(pars, interventions=intervs)
            sim.run()
            plot_all(sim)
            sims.append(sim)

    msim = cv.MultiSim(sims)
    msim.plot()