'''
Compare simulating the entire population vs. dynamic rescaling vs. static rescaling.
'''

import sciris as sc
import covasim as cv

T = sc.tic()

p = sc.objdict() # Parameters
s = sc.objdict() # Sims
m = sc.objdict() # Multisims

# Interventions
tn = cv.test_num(start_day=40, daily_tests=1000, symp_test=10) # Test a number of people
tp = cv.test_prob(start_day=30, symp_prob=0.1, asymp_prob=0.01) # Test a number of people
cb = cv.change_beta(days=50, changes=0.5) # Change beta

# Properties that are shared across sims
basepop      = 10e3
popinfected  = 100
popscale1    = 10
popscale2    = 20 # Try a different population scale
which_interv = 2 # Which intervention to test

shared = sc.objdict(
    pop_type = 'hybrid',
    beta_layer = dict(h=50, s=0, w=0, c=0.3),
    n_days = 60,
    beta = 0.010,
    rand_seed = 20589,
    verbose = 0,
    rescale_factor = 5,
    # interventions = [cb, tn, tp],
)

# Simulate the entire population
p.entire = dict(
    pop_size     = basepop*popscale1,
    pop_infected = popinfected,
    pop_scale    = 1,
    rescale      = False,
)

# Simulate a small population with dynamic scaling
p.rescale = dict(
    pop_size     = basepop,
    pop_infected = popinfected,
    pop_scale    = popscale1,
    rescale      = True,
)

# Simulate a small population with static scaling
p.static = dict(
    pop_size     = basepop,
    pop_infected = popinfected//popscale1,
    pop_scale    = popscale1,
    rescale      = False,
)

# Simulate an extra large population
p.entire2 = dict(
    pop_size     = basepop*popscale2,
    pop_infected = popinfected,
    pop_scale    = 1,
    rescale      = False,
)

# Simulate a small population with dynamic scaling
p.rescale2 = dict(
    pop_size     = basepop,
    pop_infected = popinfected,
    pop_scale    = popscale2,
    rescale      = True,
)

# Simulate a small population with static scaling
p.static2 = dict(
    pop_size     = basepop,
    pop_infected = popinfected//popscale2,
    pop_scale    = popscale2,
    rescale      = False,
)


# Create and run the sims
keys = p.keys()

for key in keys:
    p[key].update(shared)

for key in keys:
    s[key] = cv.Sim(pars=p[key], label=key)
    m[key] = cv.MultiSim(base_sim=s[key], n_runs=5)

for key in keys:
    print(f'Running {key}...')
    m[key].run()
    m[key].reduce()


# Plot
to_plot = {
    'Totals': ['cum_infections', 'cum_diagnoses'],
    'New': ['new_infections', 'new_diagnoses'],
    'Total tests': ['cum_tests'],
    'New tests': ['new_tests'],
    }
log_scale = ['Total tests']


for key in keys:
    m[key].plot(to_plot=to_plot, log_scale=log_scale)

bsims = [msim.base_sim for msim in m.values()]
mm = cv.MultiSim(sims=bsims)
mm.compare()

sc.toc(T)