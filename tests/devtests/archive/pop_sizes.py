'''
Explore how infections change as a function of population size
'''

#%% Run

import covasim as cv
import pylab as pl
import sciris as sc

sims = sc.objdict()

popsizes = [10e3, 20e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3, 90e3, 100e3]
popsizes += [p+ex for ex in [1e3, 2e3, 3e3, 4e3, 5e3] for p in popsizes]

results = []
keys = []
for p,psize in enumerate(popsizes):
    print(f'Running {psize} ({p+1}/{len(popsizes)}...')
    key = f'p{psize}'
    keys.append(key)
    sims[key] = cv.Sim(pop_size=psize,
                 n_days=30,
                 pop_infected=100,
                 rand_seed=25857+p*241,#, 29837*(p+298),
                 pop_type = 'random',
                 verbose=0,
                 )
    sims[key].run()
    results.append(sims[key].results['cum_infections'].values)


#%% Plotting

pl.figure(figsize=(18,6), dpi=200)
# pl.rcParams['font.size'] = 18

pl.subplot(1,3,1)
colors = sc.vectocolor(pl.log10(popsizes), cmap='parula')
for k,key in enumerate(keys):
    label = f'{int(float(key[1:]))/1000}k: {results[k][-1]:0.0f}'
    pl.plot(results[k], label=label, lw=3, color=colors[k])
    print(label)
# pl.legend()
pl.title('Total number of infections')
pl.xlabel('Day')
pl.ylabel('Number of infections')
sc.commaticks(axis='y')

pl.subplot(1,3,2)
for k,key in enumerate(keys):
    label = f'{int(float(key[1:]))/1000}k: {results[k][-1]/popsizes[k]*100:0.1f}'
    pl.plot(results[k]/popsizes[k]*100, label=label, lw=3, color=colors[k])
    print(label)
# pl.legend()
pl.title('Attack rate')
pl.xlabel('Day')
pl.ylabel('Attack rate (%)')

pl.subplot(1,3,3)
fres = [res[-1] for res in results]
pl.scatter(popsizes, fres, s=150, c=colors)
pl.title('Correlation')
pl.xlabel('Population size')
pl.ylabel('Number of infections')
sc.commaticks(axis='x')
sc.commaticks(axis='y')

print(pl.corrcoef(popsizes, fres))