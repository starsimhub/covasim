''' Test/example for changing from a constant viral
load to a simple time varying viral load.'''
import sciris as sc
import covasim as cv
import numpy as np
from scipy.special import kl_div

runs = 500
r0_const = np.zeros(runs)
r0_twolevel = np.zeros(runs)
r0_twolevel2 = np.zeros(runs)
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
base_pars['viral_dist'] = {'frac_time':.5, 'load_ratio':2, 'high_cap':4}


for i in range(runs):
    # Configure the sim -- can also just use a normal dictionary
    pars = base_pars
    pars['rand_seed'] = i*np.random.rand()
    pars['beta_dist']   = {'dist':'lognormal','par1':1, 'par2':0}
    print('Making sim ', i, '...')
    sim1 = cv.Sim(pars=pars)
    sim1.run()
    r0_const[i] = cv.TransTree(sim1.people).r0()
    pars['rand_seed'] = i*np.random.rand()
    pars['beta_dist']   = {'dist':'lognormal','par1':1, 'par2':.3}
    sim2 = cv.Sim(pars=pars)
    sim2.run()
    r0_twolevel[i] = cv.TransTree(sim2.people).r0()
    pars['rand_seed'] = i*np.random.rand()
    pars['beta_dist']   = {'dist':'lognormal','par1':1, 'par2':.5}
    sim3 = cv.Sim(pars=pars)
    sim3.run()
    r0_twolevel2[i] = cv.TransTree(sim3.people).r0()

print('R0 constant viral load: ', np.mean(r0_const), ' +- ', np.std(r0_const))
print('R0 two level viral load: ', np.mean(r0_twolevel), ' +- ', np.std(r0_twolevel))
print('R0 two level diff params: ', np.mean(r0_twolevel2), ' +- ', np.std(r0_twolevel2))

import matplotlib.pyplot as plt
hist1 = plt.hist(r0_const, bins=np.arange(-0.5, 10.5), density=True)
hist2 = plt.hist(r0_twolevel, bins=np.arange(-0.5, 10.5), density=True)
hist3 = plt.hist(r0_twolevel2, bins=np.arange(-0.5, 10.5), density=True)
plt.show()
# Test that the R0 did not change substantially, though the std is large
assert(abs(np.mean(r0_const)-np.mean(r0_twolevel))<np.std(r0_const))
assert(abs(np.mean(r0_const)-np.mean(r0_twolevel2))<np.std(r0_const))
assert(abs(np.mean(r0_twolevel)-np.mean(r0_twolevel2))<np.std(r0_twolevel))
# adding some sudo counts to distribution because a 0 in the second distribution
# where there isn't a 0 in the first distribution gives inf
hist1[0][hist1[0]==0] = 1e-10
hist1[0][:] = hist1[0]/sum(hist1[0])
hist2[0][hist2[0]==0] = 1e-10
hist2[0][:] = hist2[0]/sum(hist2[0])
hist3[0][hist3[0]==0] = 1e-10
hist3[0][:] = hist3[0]/sum(hist3[0])
# Test the KL diverage of the distributions of R0. Since std is large this is
# likely a stronger test than the above R0 comparisons.
assert(sum(kl_div(hist1[0], hist2[0]))<1)
assert(sum(kl_div(hist1[0], hist3[0]))<1)
assert(sum(kl_div(hist2[0], hist3[0]))<1)