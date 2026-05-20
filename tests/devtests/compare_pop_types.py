'''
Compare different population types.
'''

import sciris as sc
import covasim as cv

pars= dict(
    pop_size = 30e3,
    pop_infected = 100,
    n_days = 120,
    rand_seed = 2938,
    )

nseeds = 5
m = sc.objdict()
for pt in ['random', 'hybrid', 'synthpops']:
    sim = cv.Sim(pars=pars, pop_type=pt, label=pt)
    msim = cv.MultiSim(sim, reseed=True, n_runs=5)
    msim.run()
    msim.reduce()
    m[pt] = msim

mm = cv.MultiSim.merge(m.values(), base=True)
mm.plot(to_plot='overview')