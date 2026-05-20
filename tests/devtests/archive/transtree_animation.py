'''
Animation of transmission tree plotting for no interventions, testing only,
and TTQ.
'''

import covasim as cv
import pylab as pl
import sciris as sc
import numpy as np


tstart = sc.tic()

plot_sim    = 0
verbose     = 0
animate     = 1
animate_all = 0

isday = 15
ieday = None
pop_type = 'hybrid'
lkeys = dict(
    random='a',
    hybrid='hswc'
    )
tp = cv.test_prob(start_day=isday, end_day=ieday, symp_prob=0.01, asymp_prob=0.0, symp_quar_prob=1.0, asymp_quar_prob=1.0, test_delay=0)
ct = cv.contact_tracing(trace_probs={k:1.0 for k in lkeys[pop_type]},
                          trace_time={k:0 for k in lkeys[pop_type]},
                          start_day=isday)

contacts = dict(
    random=10,
    hybrid=dict(h=2, s=2, w=2, c=2),
    )
beta = dict(
    random=0.6/contacts['random'],
    hybrid=0.1,
    )
pop_infected = dict(
    random=1,
    hybrid=10,
    )
pars = dict(
    pop_size = 800,
    pop_infected = pop_infected[pop_type],
    pop_type = pop_type,
    n_days = 90,
    contacts = contacts[pop_type],
    beta = beta[pop_type],
    rand_seed = 3248,
    )

labels = ['No interventions', 'Testing only', 'Test + trace']
sims = sc.objdict()
sims.base  = cv.Sim(pars) # Baseline
sims.test  = cv.Sim(pars, interventions=tp) # Testing only
sims.trace = cv.Sim(pars, interventions=[tp, ct]) # Testing + contact tracing

tts = sc.objdict()
for key,sim in sims.items():
    sim.run()
    tts[key] = cv.TransTree(sim.people).detailed
    if plot_sim:
        to_plot = cv.get_sim_plots()
        to_plot['Total counts']  = ['cum_infections', 'cum_diagnoses', 'cum_quarantined', 'n_quarantined']
        sim.plot(to_plot=to_plot)


#%% Plotting

colors = sc.vectocolor(sim.n, cmap='parula')

msize = 10
suscol = [0.5,0.5,0.5]
plargs = dict(lw=2, alpha=0.5)
idelay = 0.05
daydelay = 0.3
pl.rcParams['font.size'] = 18

F = sc.objdict()
T = sc.objdict()
D = sc.objdict()
Q = sc.objdict()

for key in sims.keys():

    tt = tts[key]

    frames = [list() for i in range(sim.npts)]
    tests  = [list() for i in range(sim.npts)]
    diags  = [list() for i in range(sim.npts)]
    quars  = [list() for i in range(sim.npts)]

    for i,entry in enumerate(tt):
        frame = sc.objdict()
        dq = sc.objdict()
        if entry:
            source = entry['source']
            target = entry['target']
            target_date = entry['date']
            if source:
                source_date = tt[source]['date']
            else:
                source = 0
                source_date = 0

            frame.x = [source_date, target_date]
            frame.y = [source, target]
            frame.c = colors[source] # colors[target_date]
            frame.e = True
            frames[target_date].append(frame)

            dq.t = target
            dq.d = target_date
            dq.c = colors[target]
            date_t = entry.t.date_tested
            date_d = entry.t.date_diagnosed
            date_q = entry.t.date_known_contact
            if ~np.isnan(date_t) and date_t<sim.npts: tests[int(date_t)].append(dq)
            if ~np.isnan(date_d) and date_d<sim.npts: diags[int(date_d)].append(dq)
            if ~np.isnan(date_q) and date_q<sim.npts: quars[int(date_q)].append(dq)
        else:
            frame.x = [0]
            frame.y = [i]
            frame.c = suscol
            frame.e = False
            frames[0].append(frame)

    F[key] = frames
    T[key] = tests
    D[key] = diags
    Q[key] = quars


# Actually plot
fig, axes = pl.subplots(figsize=(24,18), nrows=3, ncols=1)
pl.subplots_adjust(**{'left': 0.10, 'bottom': 0.05, 'right': 0.85, 'top': 0.97, 'wspace': 0.25, 'hspace': 0.25})

# Create the legend
ax = pl.axes([0.85, 0.05, 0.14, 0.9])
ax.axis('off')
lcol = colors[0]
pl.plot(np.nan, np.nan, '-', c=lcol, **plargs, label='Transmission')
pl.plot(np.nan, np.nan, 'o', c=lcol, markersize=msize, **plargs, label='Source')
pl.plot(np.nan, np.nan, '*', c=lcol, markersize=msize, **plargs, label='Target')
pl.plot(np.nan, np.nan, 'o', c=lcol, markersize=msize*2, fillstyle='none', **plargs, label='Tested')
pl.plot(np.nan, np.nan, 's', c=lcol, markersize=msize*1.2, **plargs, label='Diagnosed')
pl.plot(np.nan, np.nan, 'x', c=lcol, markersize=msize*2.0, label='Known contact')
pl.legend()


for day in range(sim.npts):
    for i,key in enumerate(sims.keys()):
        pl.sca(axes[i])
        sim = sims[key]
        pl.title(f'Simulation: {labels[i]}; day: {day}; infections: {sim.results["cum_infections"][day]}')
        pl.xlim([0, sim.npts])
        pl.ylim([0, sim.n])
        pl.xlabel('Day')
        pl.ylabel('Person')
        flist = F[key][day]
        tlist = T[key][day]
        dlist = D[key][day]
        qlist = Q[key][day]
        if verbose: print(i, flist)
        for f in flist:
            if verbose: print(f)
            pl.plot(f.x[0], f.y[0], 'o', c=f.c, markersize=msize, **plargs)
            pl.plot(f.x, f.y, '-', c=f.c, **plargs)
        if animate_all:
            pl.pause(idelay)
        for f in flist:
            if f.e:
                pl.plot(f.x[1], f.y[1], '*', c=f.c, markersize=msize, **plargs)
        if tlist:
            for dq in tlist:
                pl.plot(dq.d, dq.t, 'o', c=dq.c, markersize=msize*2, fillstyle='none', **plargs)
        if dlist:
            for dq in dlist:
                pl.plot(dq.d, dq.t, 's', c=dq.c, markersize=msize*1.2, **plargs)
        if qlist:
            for dq in qlist:
                pl.plot(dq.d, dq.t, 'x', c=dq.c, markersize=msize*2.0)
        pl.plot([0, day], [0.5, 0.5], c='k', lw=5)
    if animate:
        pl.pause(daydelay)

sc.toc(tstart)