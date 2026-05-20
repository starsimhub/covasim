'''
Illustrate sinusoidally varying transmission via change beta
'''

import numpy as np
import sciris as sc
import covasim as cv

pars = sc.objdict(
    beta = 0.008,
    n_agents = 50e3,
    n_days = 180,
    verbose = 0,
)

beta_days = np.arange(pars.n_days)
beta_vals = np.cos(2*np.pi*beta_days/90)**2+0.5
beta = cv.change_beta(beta_days, beta_vals, do_plot=False)

s1 = cv.Sim(pars, label='Normal')
s2 = cv.Sim(pars, interventions=beta, label='Waves')

if __name__ == '__main__':
    msim = cv.parallel(s1, s2)
    msim.plot(['r_eff'])