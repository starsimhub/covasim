'''
Confirm that two populations generated and saved with the same seed match,
and otherwise don't.
'''

import covasim as cv
import numpy as np
import sciris as sc

T = sc.tic()

seeds = [55, 66]
pop_type = ['random', 'hybrid', 'synthpops'][1]

pars = dict(
    pop_type = pop_type,
    pop_size = 12345,
    )

lkmap = {'random':'a', 'hybrid':'w', 'synthpops':'w'}
lkey = lkmap[pop_type]

s1  = cv.Sim(pars, popfile='p1.pop', save_pop=True, rand_seed=seeds[0], label='p1')
s1b = cv.Sim(pars, popfile='p1b.pop', save_pop=True, rand_seed=seeds[0], label='p1b')
s2  = cv.Sim(pars, popfile='p2.pop', save_pop=True, rand_seed=seeds[1], label='p2')
s1.initialize()
s1b.initialize()
s2.initialize()

s1c = cv.Sim(pars, popfile='p1.pop', load_pop=True, rand_seed=seeds[0], label='p1c')
s1c.initialize()


def check_eq(a, b):
    assert np.allclose(a, b, rtol=0, atol=0, equal_nan=True)


for key in s1.people.keys():
    print(f'Checking key {key}...')
    check_eq(s1.people[key], s1b.people[key])
    check_eq(s1.people[key], s1c.people[key])


s1.initialize()
s1.run(verbose=0)
s1b.run(verbose=0)
s2.run(verbose=0)
s1c.run(verbose=0)

assert s1.summary == s1b.summary
assert s1.summary == s1c.summary
assert s1.summary != s2.summary
assert (s1.people.contacts[lkey]['p2'] == s1c.people.contacts[lkey]['p2']).all()
assert len(s1.people.contacts) == len(s1b.people.contacts)
assert len(s1.people.contacts) == len(s1c.people.contacts)
assert len(s1.people.contacts) != len(s2.people.contacts)

sc.toc(T)