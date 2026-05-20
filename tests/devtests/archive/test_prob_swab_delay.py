'''
Testing the new test_prob intervention that using a delay from symptom onset
to swab. This starts by defining the original test_prob from covasim 1.3.2


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



class test_prob_old(cv.Intervention):
    '''
    Test as many people as required based on test probability.
    Probabilities are OR together, so choose wisely.

    Args:
        symp_prob (float): Probability of testing a symptomatic (unquarantined) person
        asymp_prob (float): Probability of testing an asymptomatic (unquarantined) person
        symp_quar_prob (float): Probability of testing a symptomatic quarantined person
        asymp_quar_prob (float): Probability of testing an asymptomatic quarantined person
        subtarget (dict): subtarget intervention to people with particular indices (see test_prob() for details)
        test_sensitivity (float): Probability of a true positive
        ili_prev (float or array): Prevalence of influenza-like-illness symptoms in the population
        loss_prob (float): Probability of loss to follow-up
        test_delay (int): How long testing takes
        start_day (int): When to start the intervention
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = cv.test_prob(symp_prob=0.1, asymp_prob=0.01) # Test 10% of symptomatics and 1% of asymptomatics
        interv = cv.test_prob(symp_quar_prob=0.4) # Test 40% of those in quarantine with symptoms
    '''
    def __init__(self, symp_prob, asymp_prob=0.0, symp_quar_prob=None, asymp_quar_prob=None, subtarget=None, ili_prev=None,
                 test_sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0, end_day=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.symp_prob        = symp_prob
        self.asymp_prob       = asymp_prob
        self.symp_quar_prob   = symp_quar_prob  if  symp_quar_prob is not None else  symp_prob
        self.asymp_quar_prob  = asymp_quar_prob if asymp_quar_prob is not None else asymp_prob
        self.subtarget        = subtarget
        self.ili_prev         = ili_prev
        self.test_sensitivity = test_sensitivity
        self.loss_prob        = loss_prob
        self.test_delay       = test_delay
        self.start_day        = start_day
        self.end_day          = end_day
        return


    def initialize(self, sim):
        ''' Fix the dates '''
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        self.days      = [self.start_day, self.end_day]
        self.ili_prev  = process_daily_data(self.ili_prev, sim, self.start_day)

        self.initialized = True

        return


    def apply(self, sim):
        ''' Perform testing '''
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # Define symptomatics, accounting for ILI prevalence
        symp_inds  = cvu.true(sim.people.symptomatic)
        if self.ili_prev is not None:
            rel_t = t - self.start_day
            if rel_t < len(self.ili_prev):
                n_ili = int(self.ili_prev[rel_t] * sim['pop_size'])  # Number with ILI symptoms on this day
                ili_inds = cvu.choose(sim['pop_size'], n_ili) # Give some people some symptoms, assuming that this is independent of COVID symptomaticity...
                symp_inds = np.unique(np.concatenate((symp_inds, ili_inds)),0)

        # Define asymptomatics: those who neither have COVID symptoms nor ILI symptoms
        asymp_inds = np.setdiff1d(np.arange(sim['pop_size']), symp_inds)

        # Handle quarantine and other testing criteria
        quar_inds       = cvu.true(sim.people.quarantined)
        symp_quar_inds  = np.intersect1d(quar_inds, symp_inds)
        asymp_quar_inds = np.intersect1d(quar_inds, asymp_inds)
        if self.subtarget is not None:
            subtarget_inds  = self.subtarget['inds']
        diag_inds       = cvu.true(sim.people.diagnosed)

        # Construct the testing probabilities piece by piece -- complicated, since need to do it in the right order
        test_probs = np.zeros(sim.n) # Begin by assigning equal testing probability to everyone
        test_probs[symp_inds]       = self.symp_prob       # People with symptoms
        test_probs[asymp_inds]      = self.asymp_prob      # People without symptoms
        test_probs[symp_quar_inds]  = self.symp_quar_prob  # People with symptoms in quarantine
        test_probs[asymp_quar_inds] = self.asymp_quar_prob # People without symptoms in quarantine
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            test_probs[subtarget_inds] = subtarget_vals # People being explicitly subtargeted
        test_probs[diag_inds] = 0.0 # People who are diagnosed don't test
        test_inds = cvu.binomial_arr(test_probs).nonzero()[0] # Finally, calculate who actually tests

        sim.people.test(test_inds, test_sensitivity=self.test_sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)
        sim.results['new_tests'][t] += int(len(test_inds)*sim['pop_scale']/sim.rescale_vec[t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec

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
    test_kwargs = {'symp_prob':0.05, 'asymp_prob':0.0008, 'symp_quar_prob':0,
                   'asymp_quar_prob':0, 'test_delay':0, 'test_sensitivity':1.0, 'loss_prob':0.0}
    interventions = [
      test_prob_old(start_day='2020-01-27', **test_kwargs),
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
    test_kwargs = {'symp_prob':0.05, 'asymp_prob':0.0008, 'symp_quar_prob':0,
                   'asymp_quar_prob':0, 'test_delay':0, 'test_sensitivity':1.0, 'loss_prob':0.0,
                   'swab_delay':{'dist':dist, 'par1':par1, 'par2':par2}}
    interventions = [
      cv.test_prob(start_day='2020-01-27', **test_kwargs),
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

    time_prob_old = []
    time_prob_new = []
    time_prob_flat = []
    # 2020-05-10 - 2020-01-27
    days = (sc.readdate("2020-05-10")-sc.readdate("2020-01-27")).days + 1
    yield_prob_old = np.zeros(days)
    yield_prob_new = np.zeros(days)
    yield_prob_flat = np.zeros(days)
    # Need a couple of sims to make sure we are not looking at noise
    stage_old = {'mild': 0, 'sev': 0, 'crit': 0}
    stage_new = {'mild': 0, 'sev': 0, 'crit': 0}
    stage_flat = {'mild': 0, 'sev': 0, 'crit': 0}

    start = 1
    end = 11
    n_run = end-start
    for i in range(start,end):
       sim = single_sim_old(rand_seed=i)
       t = sc.tic()
       sim.run()
       idx = sim.people.diagnosed
       time_prob_old = np.append(time_prob_old, sim.people.date_diagnosed[idx] - sim.people.date_symptomatic[idx])
       yield_prob_old = yield_prob_old + sim.results['test_yield'].values
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
       time_prob_new = np.append(time_prob_new, sim.people.date_diagnosed[idx] - sim.people.date_symptomatic[idx])
       yield_prob_new  = yield_prob_new + sim.results['test_yield'].values
       idx_dia = sim.people.diagnosed
       idx = ~np.isnan(sim.people.date_symptomatic)
       stage_new['mild'] += sum(idx[idx_dia])/sum(idx)
       idx = ~np.isnan(sim.people.date_severe)
       stage_new['sev'] += sum(idx[idx_dia])/sum(idx)
       idx = ~np.isnan(sim.people.date_critical)
       stage_new['crit'] += sum(idx[idx_dia])/sum(idx)
       sc.toc(t)

       sim = single_sim_new(rand_seed=i, dist=None)
       t = sc.tic()
       sim.run()
       idx = sim.people.diagnosed
       time_prob_flat = np.append(time_prob_flat, sim.people.date_diagnosed[idx] - sim.people.date_symptomatic[idx])
       yield_prob_flat  = yield_prob_flat + sim.results['test_yield'].values
       idx_dia = sim.people.diagnosed
       idx = ~np.isnan(sim.people.date_symptomatic)
       stage_flat['mild'] += sum(idx[idx_dia])/sum(idx)
       idx = ~np.isnan(sim.people.date_severe)
       stage_flat['sev'] += sum(idx[idx_dia])/sum(idx)
       idx = ~np.isnan(sim.people.date_critical)
       stage_flat['crit'] += sum(idx[idx_dia])/sum(idx)
       sc.toc(t)
    yield_prob_old = yield_prob_old/n_run
    yield_prob_new = yield_prob_new/n_run
    yield_prob_flat = yield_prob_flat/n_run
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
pl.hist(time_prob_old, np.arange(-2.5,25.5), density=True)
pl.hist(time_prob_flat, np.arange(-2.5,25.5), density=True, alpha=.25)
pl.xlim([-2,20])
pl.xlabel('Symptom onset to swab')
pl.ylabel('Percent of tests')
pl.show()
if do_save:
    cv.savefig('testprobOld.png')
    pl.close()

# See how close the default distribution is the the WA data and what the model
# produces
pl.hist(time_prob_new, np.arange(-2.5,25.5), density=True)
pl.plot(data['Test Delay'], data['Percent']/100)
pl.plot(np.arange(100), pdf.pdf(np.arange(100)))
pl.xlim([-2,20])
pl.xlabel('Symptom onset to swab')
pl.ylabel('Percent of tests')
pl.legend(['Data','Distribution','Sim Histogram'])
pl.show()
if do_save:
    cv.savefig('testprobEmperical.png')
    pl.close()

# Make sure we get the same test_yields regruadless of distribution
pl.plot(yield_prob_old)
pl.plot(yield_prob_new)
pl.plot(yield_prob_flat, alpha=.25)
pl.legend(['test_prob_old','test_prob_new','test_prob_flat'])
pl.show()
if do_save:
    cv.savefig('testYieldProb.png')
    pl.close()

# Check who is being tested
pl.bar(stage_old.keys(), stage_old.values())
pl.bar(stage_new.keys(), stage_new.values(), alpha=.25)
pl.legend(['old','new'])
pl.show()
if do_save:
    cv.savefig('stageTestedProb.png')
    pl.close()
