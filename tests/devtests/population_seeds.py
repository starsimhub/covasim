'''
Check how much difference it matters to use a different population seed.
'''

import covasim as cv
import sciris as sc

T = sc.tic()

pop_type = 'synthpops'
pop_size = 10001
n_runs = 10

pars = dict(
    pop_type = pop_type,
    pop_size = pop_size,
    pop_infected = pop_size*0.01
    )

# Create and save the populations
s1p = cv.Sim(pars, popfile='p1.pop', save_pop=True, rand_seed=127834)
s2p = cv.Sim(pars, popfile='p2.pop', save_pop=True, rand_seed=575836)
s1p.initialize()
s2p.initialize()
assert len(s1p.people.contacts) != len(s2p.people.contacts)

# Create sims from loaded populations
rs1 = 210
rs2 = 982
s = sc.objdict()
s.s1 = cv.Sim(pars, popfile='p1.pop', load_pop=True, rand_seed=rs1, label=f'Pop. 1, {pop_type}, {pop_size:n}, seed {rs1}')
s.s3 = cv.Sim(pars, popfile='p1.pop', load_pop=True, rand_seed=rs2, label=f'Pop. 1, {pop_type}, {pop_size:n}, seed {rs2}')
s.s2 = cv.Sim(pars, popfile='p2.pop', load_pop=True, rand_seed=rs1, label=f'Pop. 2, {pop_type}, {pop_size:n}, seed {rs1}')
s.s4 = cv.Sim(pars, popfile='p2.pop', load_pop=True, rand_seed=rs2, label=f'Pop. 2, {pop_type}, {pop_size:n}, seed {rs2}')

# Run comparison
m = sc.objdict()
for k,sim in s.items():
    m[k] = cv.MultiSim(sim, n_runs=n_runs)
    m[k].run(reduce=True)
msim = cv.MultiSim.merge(m.values(), base=True)
msim.plot(to_plot='overview')

sc.toc(T)