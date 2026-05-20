'''
Check if states match
'''

import sciris as sc
import covasim as cv

states = [
        'susceptible',
        'naive',
        'exposed',
        'infectious',
        'symptomatic',
        'severe',
        'critical',
        'tested',
        'diagnosed',
        'recovered',
        'dead',
]

sim = cv.Sim()
sim.run()

d = sc.objdict()
for state in states:
    n_in = len(cv.true(sim.people[state]))
    n_out = len(cv.false(sim.people[state]))
    d[state] = n_in
    assert n_in + n_out == sim['pop_size']

print(sim.summary)
print(d)
assert d.naive + d.exposed + d.recovered + d.dead == sim['pop_size']
