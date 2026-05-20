'''
Check 32 bit vs 64 bit. Change the value of default_precision in defaults.py,
then rerun.

A very simple test is

python -c 'import covasim as cv; sim = cv.Sim(pop_size=1e6); sim.run()'
'''


#%% Settings

import covasim as cv
import sciris as sc
import pandas as pd

popsizes = [20e3, 100e3, 500e3]
repeats = 3


#%% First check memory since it's quicker
def mrun():
    ''' Split these out  explicitly since memory is counted per line '''

    print('Dummy run to load any missing libraries')
    sim = cv.Sim(verbose=0)
    sim.run()

    popsize = popsizes[0]
    print(f'Working on {popsize} for {cv.defaults.default_int}...')
    sim = cv.Sim(pop_size=popsize, verbose=0)
    sim.run()

    popsize = popsizes[1]
    print(f'Working on {popsize} for {cv.defaults.default_int}...')
    sim = cv.Sim(pop_size=popsize, verbose=0)
    sim.run()

    popsize = popsizes[2]
    print(f'Working on {popsize} for {cv.defaults.default_int}...')
    sim = cv.Sim(pop_size=popsize, verbose=0)
    sim.run()

    return sim


sc.mprofile(mrun, mrun)


#%% Now check timings
timings = []
for popsize in popsizes:
    for r in range(repeats):
        print(f'Working on {popsize} for {cv.defaults.default_int}, iteration {r}...')
        T = sc.tic()
        sim = cv.Sim(pop_size=popsize, verbose=0)
        sim.run()
        out = sc.toc(T, output=True)
        timings.append({'popsize':popsize, 'elapsed':out})


df  = pd.DataFrame.from_dict(timings)
print(df)