'''
Check regression, specifically of testing and tracing interventions, against
earlier versions of Covasim verson.

Instructions:
    1. Copy to a new file (since this file will disappear when you check out the other branch)
    2. Set 'old' and 'new' version labels
    2. Check out an earlier Covasim version (e.g. "git checkout v1.6.1")
    3. Run
    4. Check out Covasim a later version (e.g. "git checkout v1.7.2" or "git checkout master")
    5. Run

At least between 1.6.1 and 1.7.2, should be a change in test_num but not in test_prob.
'''

import sciris as sc
import covasim as cv


# Modify these to compare different versions
vers = sc.objdict({
        'old':'1.6.1',
        'new':'1.7.2',
        })

# Setup -- exact values don't really matter

n_runs = 5

pars = dict(
    pop_type = 'hybrid',
    n_days = 60,
    rand_seed = 1,
    verbose = 0.01,
    pop_size = 50000,
    pop_infected = 500,
    pop_scale = 10,
    rescale = True,
)

tn = cv.test_num(start_day=10, daily_tests=2000, symp_test=100)
tp = cv.test_prob(start_day=10, symp_prob=0.1, asymp_prob=0.01)
ct = cv.contact_tracing(start_day=20, trace_probs={k:0.5 for k in 'hswc'})

# Run
for label,ti in {'test_num':tn, 'test_prob':tp}.items():

    ver = cv.__version__
    sim = cv.Sim(pars, interventions=sc.dcp([ti, ct]), label=f'Version {ver}, {label}')
    m1 = cv.MultiSim(sim)
    m1.run(n_runs=n_runs)
    m1.reduce()

    old_fn = f'intervention_regression_{label}_1.6.1.msim'
    if ver == vers.old:
        m1.save(filename=old_fn)
        print(f'Multisim saved as {old_fn}')
    elif ver == vers.new:
        m0 = cv.load(old_fn)
        msim = cv.MultiSim.merge([m0, m1], base=True)
        fig = msim.plot(to_plot='overview')
        cv.maximize()
    else:
        raise NotImplementedError(f'You have asked to compare versions {vers.old} and {vers.new}, but Covasim is version {ver}')


