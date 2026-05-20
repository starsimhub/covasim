'''
Demonstrate different multisim plotting options.
'''

import covasim as cv

n_sims = 8

sims = []
labels = []
for i in range(n_sims):
    sim = cv.Sim(label='Sim' + '*'*i, rand_seed=i)
    sims.append(sim)
    labels.append(f'Sim label {i}')

msim = cv.MultiSim(sims)
msim.run()

msim.plot(color_by_sim=False)
msim.plot(color_by_sim=True)
msim.plot(color_by_sim=True, labels=labels)