'''
Example of running a nested multisim object
'''

import covasim as cv

n_sims = 3
betas = [0.012, 0.016, 0.018]

msims = []
for beta in betas:
    sims = []
    for s in range(n_sims):
        sim = cv.Sim(pop_size=10e3, beta=beta, rand_seed=s, label=f'Beta = {beta}')
        sims.append(sim)
    msim = cv.MultiSim(sims)
    msim.run()
    msim.mean()
    msims.append(msim)

merged = cv.MultiSim.merge(msims, base=True)
merged.plot(color_by_sim=True)