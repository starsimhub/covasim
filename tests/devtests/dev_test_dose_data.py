import numpy as np
import covasim as cv
import pylab as pl
import sciris as sc
 
# Create some base parameters
pars = {
    'beta': 0.015,
    'n_days': 120,
}

# Create some vaccination data
# First doses: start vaccinating 100 people/day on day 30 then scale up by 100 doses/day
def num_first_doses(t):
    if t < 30: # No doses at first
        return 0
    elif t < 76: # Scaling up supply
        return (t - 29) * 100
    else: # No more doses as demand stops
        return 0

# Store the dosage data
first_dose_data = {t:num_first_doses(t) for t in range(pars['n_days'])}
second_dose_data = {t:num_first_doses(t-41) for t in range(pars['n_days'])}

# The standard Pfizer vaccine has 2 doses -- initialize this to get the default efficacy parameters
pfizer_pars = cv.BaseVaccination(vaccine='pfizer').p

# Create the first dose
pfizer_pars_1 = sc.dcp(pfizer_pars)
pfizer_pars_1['doses'] = 1
pfizer_pars_1['interval'] = None

# Create the second dose
pfizer_pars_2 = sc.dcp(pfizer_pars)
pfizer_pars_2['nab_init'] = None
pfizer_pars_2['doses'] = 1
pfizer_pars_2['interval'] = None
second_dose_target  = {'inds': lambda sim: cv.true(sim.people.doses != 1), 'vals': 0} # Only give a second dose to people who have had a first dose

n_doses = []

pfizer_dose1 = cv.vaccinate_num(vaccine=pfizer_pars_1, num_doses=first_dose_data)
pfizer_dose2 = cv.vaccinate_num(vaccine=pfizer_pars_2, num_doses=second_dose_data, booster=True, subtarget=second_dose_target)

sim = cv.Sim(
    use_waning=True,
    pars=pars,
    interventions=[pfizer_dose1, pfizer_dose2],
    analyzers=lambda sim: n_doses.append(sim.people.doses.copy())
)
sim.run()

pl.figure()
n_doses = np.array(n_doses)
fully_vaccinated = (n_doses == 2).sum(axis=1)
first_dose = (n_doses == 1).sum(axis=1)
pl.stackplot(sim.tvec, first_dose, fully_vaccinated)
pl.legend(['First dose','Fully vaccinated'])
pl.show()
