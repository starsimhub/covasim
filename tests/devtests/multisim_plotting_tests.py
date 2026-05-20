'''
Demonstrate different MultiSim plotting options
'''

import covasim as cv

show_reduced    = 1
show_several    = 1
show_many       = 1
show_two_bounds = 1

sim = cv.Sim(verbose=0, pop_size=5000)

# Reduced plotting
if show_reduced:
    m1 = cv.MultiSim(sim)
    m1.run()
    m1.reduce()
    m1.plot()

# Separate plotting, several sims
if show_several:
    m2 = cv.MultiSim(sim)
    m2.run(n_runs=4)
    m2.plot()

# Separate plotting, lots of sims
if show_many:
    m3 = cv.MultiSim(sim)
    m3.run(n_runs=20)
    m3.plot()

# Show the plotting of multiple multisims
if show_two_bounds:
    sa = cv.Sim(verbose=0, beta=0.016, label='High beta')
    sb = cv.Sim(verbose=0, beta=0.015, label='Moderate beta')
    sc = cv.Sim(verbose=0, beta=0.014, label='Low beta')
    ma = cv.MultiSim(sa)
    mb = cv.MultiSim(sb)
    mc = cv.MultiSim(sc)
    ma.run(reduce=True)
    mb.run(reduce=True)
    mc.run(reduce=True)
    m4 = cv.MultiSim.merge(ma, mb, mc, base=True)
    m4.plot()