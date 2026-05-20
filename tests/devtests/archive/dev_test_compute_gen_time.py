''' Test/example for calculating the generation time. '''
import covasim as cv

pars = {}

pars['n_days']       = 180
pars['pop_size']     = 5000
pars['pop_infected'] = 10
pars['rand_seed']    = 1
pars['asymp_factor'] = 1 # Multiply beta by this factor for asymptomatic cases
pars['diag_factor']  = 0 # Multiply beta by this factor for diganosed cases

# Duration parameters: time for disease recovery
pars['dur'] = {}
pars['dur']['exp2inf']  = {'dist':'normal_int', 'par1':4, 'par2':0} # Duration from exposed to infectious
pars['dur']['inf2sym']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from infectious to symptomatic
pars['dur']['sym2sev']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from symptomatic to severe symptoms
pars['dur']['sev2crit'] = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from severe symptoms to requiring ICU
pars['dur']['asym2rec'] = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration for asymptomatics to recover
pars['dur']['mild2rec'] = {'dist':'normal_int', 'par1':8, 'par2':0} # Duration from mild symptoms to recovered
pars['dur']['sev2rec']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from severe symptoms to recovered - leads to mean total disease time of
pars['dur']['crit2rec'] = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from critical symptoms to recovered
pars['dur']['crit2die'] = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from critical symptoms to death

pars['OR_no_treat']     = 1.0  # Odds ratio for how much more likely people are to die if no treatment available
pars['rel_symp_prob']   = 2.0  # Scale factor for proportion of symptomatic cases
pars['rel_severe_prob'] = 0.0  # Scale factor for proportion of symptomatic cases that become severe
pars['rel_crit_prob']   = 0.0  # Scale factor for proportion of severe cases that become critical
pars['rel_death_prob']  = 0.0  # Scale factor for proportion of critical cases that result in death
pars['prog_by_age']     = False



print('Testing true vs clinical (should use the same)...')
sim = cv.Sim(pars=pars)
sim.run()
sim.compute_gen_time()
assert(sim.results['gen_time']['true']==sim.results['gen_time']['clinical'])
assert(sim.results['gen_time']['true_std']==sim.results['gen_time']['clinical_std'])

print('Testing with asympotmatic and sympotmatic...')
pars['rel_symp_prob']   = 0.5  # Scale factor for proportion of symptomatic cases
pars['dur']['asym2rec'] = {'dist':'normal_int', 'par1':4,  'par2':0} # Duration for asymptomatics to recover
pars['dur']['mild2rec'] = {'dist':'normal_int', 'par1':8,  'par2':0} # Duration from mild symptoms to recovered
sim = cv.Sim(pars=pars)
sim.run()
sim.compute_gen_time()
print('true gen_time: ', sim.results['gen_time']['true'])
print('clinical gen_time: ', sim.results['gen_time']['clinical'])
assert(sim.results['gen_time']['true']<sim.results['gen_time']['clinical'])
