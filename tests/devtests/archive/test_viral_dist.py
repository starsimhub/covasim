''' Test/example for changing from a constant viral
load to a simple time varying viral load.'''
import sciris as sc
import covasim as cv
import numpy as np

runs = 100
dist_const = np.zeros(0,dtype=np.int64)
dist_twolevel = np.zeros(0,dtype=np.int64)
dist_twolevel2 = np.zeros(0,dtype=np.int64)
base_pars = sc.objdict(
        n_days       = 12,   # Number of days to simulate
        asymp_factor = 1, # Multiply beta by this factor for asymptomatic cases
        diag_factor  = 1, # Multiply beta by this factor for diganosed cases -- baseline assumes complete isolation
        verbose = 0,
        pop_size = 1000,
        pop_infected = 1
)
base_pars['dur'] = {}
base_pars['dur']['exp2inf']  = {'dist':'normal_int', 'par1':4, 'par2':0} # Duration from exposed to infectious
base_pars['dur']['inf2sym']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from infectious to symptomatic
base_pars['dur']['sym2sev']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from symptomatic to severe symptoms
base_pars['dur']['sev2crit'] = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from severe symptoms to requiring ICU

# Duration parameters: time for disease recovery
base_pars['dur']['asym2rec'] = {'dist':'normal_int', 'par1':0,  'par2':0} # Duration for asymptomatics to recover
base_pars['dur']['mild2rec'] = {'dist':'normal_int', 'par1':8,  'par2':0} # Duration from mild symptoms to recovered
base_pars['dur']['sev2rec']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from severe symptoms to recovered - leads to mean total disease time of
base_pars['dur']['crit2rec'] = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from critical symptoms to recovered
base_pars['dur']['crit2die'] = {'dist':'normal_int', 'par1':0,  'par2':0} # Duration from critical symptoms to death

base_pars['OR_no_treat']     = 1.0  # Odds ratio for how much more likely people are to die if no treatment available
base_pars['rel_symp_prob']   = 2.0  # Scale factor for proportion of symptomatic cases
base_pars['rel_severe_prob'] = 0  # Scale factor for proportion of symptomatic cases that become severe
base_pars['rel_crit_prob']   = 0  # Scale factor for proportion of severe cases that become critical
base_pars['rel_death_prob']  = 0  # Scale factor for proportion of critical cases that result in death
base_pars['prog_by_age']     = False

for i in range(runs):
    # Configure the sim -- can also just use a normal dictionary
    pars = base_pars
    pars['rand_seed'] = i*np.random.rand()
    pars['viral_dist'] = {'frac_time':1, 'load_ratio':1, 'high_cap':100}
    print('Making sim ', i, '...')
    sim1 = cv.Sim(pars=pars)
    sim1.run()
    linelist = sim1.people.transtree.targets[0]
    targets = np.zeros(len(linelist), dtype=np.int64)
    for j in np.arange(len(linelist)):
        targets[j] = linelist[j]['target']
    time_temp = np.int64(np.array(sim1.people.date_exposed)[targets]) - np.int64(np.array(sim1.people.date_infectious[0]))
    dist_const = np.append(dist_const,time_temp)
    pars['rand_seed'] = i*np.random.rand()
    pars['viral_dist'] = {'frac_time':.5, 'load_ratio':4, 'high_cap':4}
    sim2 = cv.Sim(pars=pars)
    sim2.run()
    linelist = sim2.people.transtree.targets[0]
    targets = np.zeros(len(linelist), dtype=np.int64)
    for j in np.arange(len(linelist)):
        targets[j] = linelist[j]['target']
    time_temp = np.int64(np.array(sim2.people.date_exposed)[targets]) - np.int64(np.array(sim2.people.date_infectious[0]))
    dist_twolevel = np.append(dist_twolevel,time_temp)
    pars['rand_seed'] = i*np.random.rand()
    pars['viral_dist'] = {'frac_time':.125, 'load_ratio':10, 'high_cap':1}
    sim3 = cv.Sim(pars=pars)
    sim3.run()
    linelist = sim3.people.transtree.targets[0]
    targets = np.zeros(len(linelist), dtype=np.int64)
    for j in np.arange(len(linelist)):
        targets[j] = linelist[j]['target']
    time_temp = np.int64(np.array(sim3.people.date_exposed)[targets]) - np.int64(np.array(sim3.people.date_infectious[0]))
    dist_twolevel2 = np.append(dist_twolevel2,time_temp)

import matplotlib.pyplot as plt
hist3 = plt.hist(dist_twolevel2, bins=np.arange(-0.5, 10.5), density=True)
hist2 = plt.hist(dist_twolevel, bins=np.arange(-0.5, 10.5), density=True)
hist1 = plt.hist(dist_const, bins=np.arange(-0.5, 10.5), density=True)
plt.show()