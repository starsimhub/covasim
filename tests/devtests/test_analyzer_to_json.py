'''
Confirm that with default settings, all analyzers can be exported as JSONs.
'''

import sciris as sc
import covasim as cv

datafile = sc.thisdir(__file__, aspath=True).parent / 'example_data.csv'

# Create and runt he sim
sim = cv.Sim(analyzers=[cv.snapshot(days='2020-04-04'),
                        cv.age_histogram(),
                        cv.daily_age_stats(),
                        cv.daily_stats()],
             datafile=datafile)
sim.run()

# Compute extra analyzers
tt = sim.make_transtree()
fit = sim.compute_fit()

# Construct list of all analyzers
analyzers = sim['analyzers'] + [tt, fit]

# Make jsons
jsons = {}
for an in analyzers:
    print(f'Working on analyzer {an.label}...')
    jsons[an.label] = an.to_json()

# Compute memory
for k,json in jsons.items():
    sc.checkmem({k:json})

print('Done.')