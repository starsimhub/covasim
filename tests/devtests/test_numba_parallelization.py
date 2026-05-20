'''
Test different parallelization options
'''

import covasim as cv
import sciris as sc

# Set the parallelization to use -- 0 = none, 1 = safe, 2 = rand
parallel = 1

pars = dict(
    pop_size = 1e6,
    n_days = 200,
    verbose = 0.1,
)

cv.options.set(numba_cache=0, numba_parallel=parallel)

parstr = f'Parallel={cv.options.numba_parallel}'
print('Initializing (always single core)')
sim = cv.Sim(**pars, label=parstr)
sim.initialize()

print(f'Running ({parstr})')
sc.tic()
sim.run()
sc.toc(label=parstr)