'''
Testing the new test_num intervention that using a delay from symptom onset
to swab. This starts by defining the original test_num from Covasim 1.3.2.

NOTE: this test relies on several data files that are not included in the repository.
It is included for completeness.
'''

import os
import covasim as cv
import pylab as pl
import numpy as np
import sciris as sc
from covasim import utils as cvu
from covasim.interventions import process_daily_data


do_save = False

datafile = '20200510_KingCounty_Covasim.csv'
swabfile = 'WA_Symptoms_to_Swab.csv'

if not os.path.exists(datafile) or not os.path.exists(swabfile):
    raise FileNotFoundError(f'Cannot find {datafile} or {swabfile}, this test is not available')


class test_num_old(cv.Intervention):
    '''
    Test a fixed number of people per day.

    Args:
        daily_tests (int or arr or dataframe/series): number of tests per day; if integer, use that number every day
        symp_test (float): odds ratio of a symptomatic person testing
        quar_test (float): probability of a person in quarantine testing
        subtarget (dict or func): subtarget intervention to people with particular indices (format: {'ind': array of indices, or function to return indices from the sim, 'vals': value(s) to apply}
        sensitivity (float): test sensitivity
        ili_prev (float or array or dataframe/series): Prevalence of influenza-like-illness symptoms in the population
        loss_prob (float): probability of the person being lost-to-follow-up
        test_delay (int): days for test result to be known
        start_day (int): day the intervention starts
        end_day (int): day the intervention ends
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = cv.test_num(daily_tests=[0.10*n_people]*npts)
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': sim.people.age>50, 'vals': 1.2}) # People over 50 are 20% more likely to test
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': lambda sim: sim.people.age>50, 'vals': 1.2}) # People over 50 are 20% more likely to test
    '''

    def __init__(self, daily_tests, symp_test=100.0, quar_test=1.0, subtarget=None,
                 ili_prev=None, sensitivity=1.0, loss_prob=0, test_delay=0,
                 start_day=0, end_day=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.daily_tests = daily_tests # Should be a list of length matching time
        self.symp_test   = symp_test   # Set probability of testing symptomatics
        self.quar_test   = quar_test
        self.subtarget   = subtarget  # Set any other testing criteria
        self.ili_prev    = ili_prev     # Should be a list of length matching time or a float or a dataframe
        self.sensitivity = sensitivity
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
        return


    def initialize(self, sim):
        ''' Fix the dates and number of tests '''

        # Handle days
        self.start_day   = sim.day(self.start_day)
        self.end_day     = sim.day(self.end_day)
        self.days        = [self.start_day, self.end_day]

        # Process daily data
        self.daily_tests = process_daily_data(self.daily_tests, sim, self.start_day)
        self.ili_prev    = process_daily_data(self.ili_prev,    sim, self.start_day)

        self.initialized = True

        return


    def apply(self, sim):

        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # Check that there are still tests
        rel_t = t - self.start_day
        if rel_t < len(self.daily_tests):
            n_tests = int(self.daily_tests[rel_t]/sim.rescale_vec[t])  # Number of tests for this day -- rescaled
            if not (n_tests and pl.isfinite(n_tests)): # If there are no tests today, abort early
                return
            else:
                sim.results['new_tests'][t] += n_tests
        else:
            return

        test_probs = np.ones(sim.n) # Begin by assigning equal testing probability to everyone

        # Handle symptomatic testing, taking into account prevalence of ILI symptoms
        symp_inds = cvu.true(sim.people.symptomatic)
        if self.ili_prev is not None:
            if rel_t < len(self.ili_prev):
                n_ili = int(self.ili_prev[rel_t] * sim['pop_size'])  # Number with ILI symptoms on this day
                ili_inds = cvu.choose(sim['pop_size'], n_ili) # Give some people some symptoms. Assuming that this is independent of COVID symptomaticity...
                symp_inds = np.unique(np.concatenate((symp_inds, ili_inds)),0)
        test_probs[symp_inds] *= self.symp_test

        # Handle quarantine testing
        quar_inds  = cvu.true(sim.people.quarantined)
        test_probs[quar_inds] *= self.quar_test

        # Handle any other user-specified testing criteria
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = cv.get_subtargets(self.subtarget, sim)
            test_probs[subtarget_inds] = test_probs[subtarget_inds]*subtarget_vals

        # Don't re-diagnose people
        diag_inds  = cvu.true(sim.people.diagnosed)
        test_probs[diag_inds] = 0.

        # Now choose who gets tested and test them
        test_inds = cvu.choose_w(probs=test_probs, n=n_tests, unique=False)
        sim.people.test(test_inds, self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)

        return

def single_sim_old(end_day='2020-05-10', rand_seed=1):

    pop_type  = 'hybrid'
    pop_size  = 225000
    pop_scale = 2.25e6/pop_size
    start_day = '2020-01-27'
    # end_day   = '2020-05-01'   # for calibration plots
    # end_day   = '2020-06-30'  # for projection plots


    pars = {
     'verbose': 0,
     'pop_size': pop_size,
     'pop_infected': 30,   # 300
     'pop_type': pop_type,
     'start_day': start_day,
     'n_days': (sc.readdate(end_day)-sc.readdate(start_day)).days,
     'pop_scale': pop_scale,
     'rescale': True,
     'beta': 0.015,
     'rand_seed': rand_seed,
    }

    sim = cv.Sim(pars, datafile=datafile)

    # Define beta interventions
    b_days = ['2020-03-04', '2020-03-12', '2020-03-23']
    b_ch = sc.objdict()
    b_ch.h = [1.00, 1.10, 1.20]
    b_ch.s = [1.00, 0.00, 0.00]
    b_ch.w = [0.60, 0.40, 0.25]
    b_ch.c = [0.60, 0.40, 0.25]

    # Define testing interventions
    daily_tests = sim.data['new_tests']
    test_kwargs = {'quar_test': 0, 'sensitivity': 1.0, 'test_delay': 0,
                   'loss_prob': 0}
    interventions = [
      test_num_old(daily_tests=daily_tests, symp_test=70,  start_day='2020-01-27', end_day=None, **test_kwargs),
        ]

    for lkey,ch in b_ch.items():
        interventions.append(cv.change_beta(days=b_days, changes=b_ch[lkey], layers=lkey))

    sim.update_pars(interventions=interventions)

    sim.initialize()

    # Define age susceptibility
    mapping = {
        0: 0.2,
        20: 0.9,
        40: 1.0,
        70: 2.5,
        80: 5.0,
        90: 10.0,
    }

    for age, val in mapping.items():
        sim.people.rel_sus[sim.people.age > age] = val

    return sim

def single_sim_new(end_day='2020-05-10', rand_seed=1, dist='lognormal', par1=10, par2=170):

    pop_type  = 'hybrid'
    pop_size  = 225000
    pop_scale = 2.25e6/pop_size
    start_day = '2020-01-27'
    # end_day   = '2020-05-01'   # for calibration plots
    # end_day   = '2020-06-30'  # for projection plots


    pars = {
     'verbose': 0,
     'pop_size': pop_size,
     'pop_infected': 30,   # 300
     'pop_type': pop_type,
     'start_day': start_day,
     'n_days': (sc.readdate(end_day)-sc.readdate(start_day)).days,
     'pop_scale': pop_scale,
     'rescale': True,
     'beta': 0.015,
     'rand_seed': rand_seed,
    }

    sim = cv.Sim(pars, datafile=datafile)

    # Define beta interventions
    b_days = ['2020-03-04', '2020-03-12', '2020-03-23']
    b_ch = sc.objdict()
    b_ch.h = [1.00, 1.10, 1.20]
    b_ch.s = [1.00, 0.00, 0.00]
    b_ch.w = [0.60, 0.40, 0.25]
    b_ch.c = [0.60, 0.40, 0.25]

    # Define testing interventions
    daily_tests = sim.data['new_tests']
    test_kwargs = {'quar_test': 0, 'sensitivity': 1.0, 'test_delay': 0,
                   'loss_prob': 0}
    interventions = [
      cv.test_num(daily_tests=daily_tests, symp_test=70,  start_day='2020-01-27', end_day=None,
                  swab_delay_dist={'dist':dist, 'par1':par1, 'par2':par2}, **test_kwargs),
        ]

    for lkey,ch in b_ch.items():
        interventions.append(cv.change_beta(days=b_days, changes=b_ch[lkey], layers=lkey))

    sim.update_pars(interventions=interventions)

    sim.initialize()

    # Define age susceptibility
    mapping = {
        0: 0.2,
        20: 0.9,
        40: 1.0,
        70: 2.5,
        80: 5.0,
        90: 10.0,
    }

    for age, val in mapping.items():
        sim.people.rel_sus[sim.people.age > age] = val

    return sim

if __name__ == "__main__":

    time_num_old = []
    time_num_new = []
    time_num_flat = []
    # 2020-05-10 - 2020-01-27
    days = (sc.readdate("2020-05-10")-sc.readdate("2020-01-27")).days + 1
    yield_num_old = np.zeros(days)
    yield_num_new = np.zeros(days)
    yield_num_flat = np.zeros(days)
    # Need a couple of sims to make sure we are not looking at noise
    stage_old = {'mild': 0, 'sev': 0, 'crit': 0}
    stage_new = {'mild': 0, 'sev': 0, 'crit': 0}
    stage_flat = {'mild': 0, 'sev': 0, 'crit': 0}

    start = 1
    end = 2
    n_run = end-start
    for i in range(start,end):
       sim = single_sim_old(rand_seed=i)
       t = sc.tic()
       sim.run()
       idx = sim.people.diagnosed
       time_num_old = np.append(time_num_old, sim.people.date_diagnosed[idx] - sim.people.date_symptomatic[idx])
       yield_num_old = yield_num_old + sim.results['test_yield'].values
       idx_dia = sim.people.diagnosed
       idx = ~np.isnan(sim.people.date_symptomatic)
       stage_old['mild'] += sum(idx[idx_dia])/sum(idx)
       idx = ~np.isnan(sim.people.date_severe)
       stage_old['sev'] += sum(idx[idx_dia])/sum(idx)
       idx = ~np.isnan(sim.people.date_critical)
       stage_old['crit'] += sum(idx[idx_dia])/sum(idx)
       sc.toc(t)

       sim = single_sim_new(rand_seed=i)
       t = sc.tic()
       sim.run()
       idx = sim.people.diagnosed
       time_num_new = np.append(time_num_new, sim.people.date_diagnosed[idx] - sim.people.date_symptomatic[idx])
       yield_num_new  = yield_num_new + sim.results['test_yield'].values
       idx_dia = sim.people.diagnosed
       idx = ~np.isnan(sim.people.date_symptomatic)
       stage_new['mild'] += sum(idx[idx_dia])/sum(idx)
       idx = ~np.isnan(sim.people.date_severe)
       stage_new['sev'] += sum(idx[idx_dia])/sum(idx)
       idx = ~np.isnan(sim.people.date_critical)
       stage_new['crit'] += sum(idx[idx_dia])/sum(idx)
       sc.toc(t)
       sim.plot(to_plot={'test':['new_tests']})
       pl.show()
       if do_save:
           cv.savefig('testScalingNum.png')
           pl.close()

       sim = single_sim_new(rand_seed=i, dist=None)
       t = sc.tic()
       sim.run()
       idx = sim.people.diagnosed
       time_num_flat = np.append(time_num_flat, sim.people.date_diagnosed[idx] - sim.people.date_symptomatic[idx])
       yield_num_flat  = yield_num_flat + sim.results['test_yield'].values
       idx_dia = sim.people.diagnosed
       idx = ~np.isnan(sim.people.date_symptomatic)
       stage_flat['mild'] += sum(idx[idx_dia])/sum(idx)
       idx = ~np.isnan(sim.people.date_severe)
       stage_flat['sev'] += sum(idx[idx_dia])/sum(idx)
       idx = ~np.isnan(sim.people.date_critical)
       stage_flat['crit'] += sum(idx[idx_dia])/sum(idx)
       sc.toc(t)

    yield_num_old = yield_num_old/n_run
    yield_num_new = yield_num_new/n_run
    yield_num_flat = yield_num_flat/n_run
    stage_old['mild'] /= n_run
    stage_old['sev'] /= n_run
    stage_old['crit'] /= n_run
    stage_new['mild'] /= n_run
    stage_new['sev'] /= n_run
    stage_new['crit'] /= n_run
    stage_flat['mild'] /= n_run
    stage_flat['sev'] /= n_run
    stage_flat['crit'] /= n_run


import pandas as pd

data = pd.read_csv(swabfile)
data = data.loc[data['Test Delay']!='#NAME?',]
pdf = cvu.get_pdf('lognormal',10,170)


# Check that not using a distribution gives the same answer as before
pl.hist(time_num_old, np.arange(-2.5,25.5), density=True)
pl.hist(time_num_flat, np.arange(-2.5,25.5), density=True, alpha=.25)
pl.xlim([-2,20])
pl.xlabel('Symptom onset to swab')
pl.ylabel('Percent of tests')
pl.show()
if do_save:
    cv.savefig('testNumOld.png')
    pl.close()

# See how close the default distribution is the the WA data and what the model
# produces
pl.hist(time_num_new, np.arange(-2.5,25.5), density=True)
pl.plot(data['Test Delay'], data['Percent']/100)
pl.plot(np.arange(100), pdf.pdf(np.arange(100)))
pl.xlim([-2,20])
pl.xlabel('Symptom onset to swab')
pl.ylabel('Percent of tests')
pl.legend(['Data','Distribution','Sim Histogram'])
pl.show()
if do_save:
    cv.savefig('testNumEmperical.png')
    pl.close()

# Make sure we get the same test_yields regruadless of distribution
pl.plot(yield_num_old)
pl.plot(yield_num_new)
pl.plot(yield_num_flat, alpha=.25)
pl.legend(['test_num_old','test_num_new','test_num_flat'])
pl.show()
if do_save:
    cv.savefig('testYieldNum.png')
    pl.close()

# Check who is being tested
pl.bar(stage_old.keys(), stage_old.values())
pl.bar(stage_new.keys(), stage_new.values(), alpha=.25)
pl.legend(['old','new'])
pl.show()
if do_save:
    pl.savefig('stageTestedNum.png')
    pl.close()


# Test that it works with no symptomatics.
# Does fail in the finalize stage with no infections

pop_type  = 'hybrid'
pop_size  = 225000
pop_scale = 2.25e6/pop_size
start_day = '2020-01-27'
end_day   = '2020-05-01'   # for calibration plots


pars = {
 'verbose': 0,
 'pop_size': pop_size,
 'pop_infected': 0,   # 300
 'pop_type': pop_type,
 'start_day': start_day,
 'n_days': (sc.readdate(end_day)-sc.readdate(start_day)).days,
 'pop_scale': pop_scale,
 'beta': 0.015,
 'rand_seed': 1,
}

sim = cv.Sim(pars, datafile=datafile)

# Define beta interventions
b_days = ['2020-03-04', '2020-03-12', '2020-03-23']
b_ch = sc.objdict()
b_ch.h = [1.00, 1.10, 1.20]
b_ch.s = [1.00, 0.00, 0.00]
b_ch.w = [0.60, 0.40, 0.25]
b_ch.c = [0.60, 0.40, 0.25]

# Define testing interventions
daily_tests = sim.data['new_tests']
test_kwargs = {'quar_test': 0, 'sensitivity': 1.0, 'test_delay': 0,
               'loss_prob': 0}
interventions = [
  cv.test_num(daily_tests=daily_tests, symp_test=70,  start_day='2020-01-27', end_day=None,
             swab_delay_dist={'dist':'lognormal', 'par1':10, 'par2':170}, **test_kwargs),
   ]

for lkey,ch in b_ch.items():
    interventions.append(cv.change_beta(days=b_days, changes=b_ch[lkey], layers=lkey))

sim.update_pars(interventions=interventions)

sim.initialize()
sim = sim.run(until='2020-04-30')
