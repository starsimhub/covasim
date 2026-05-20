'''
Calculate vaccine efficiency for protection against symptomatic covid after first dose
'''

import numpy as np
import sciris as sc
import covasim as cv

cv.check_version('>=3.0.0')

vaccines = ['pfizer', 'moderna', 'az', 'j&j']

# construct analyzer to select placebo arm
class placebo_arm(cv.Analyzer):
    def __init__(self, day, trial_size, **kwargs):
        super().__init__(**kwargs)
        self.day = day
        self.trial_size = trial_size
        return

    def initialize(self, sim=None):
        self.placebo_inds = []
        self.initialized = True
        return

    def apply(self, sim):
        if sim.t == self.day:
            eligible = cv.true(~np.isfinite(sim.people.date_exposed) & ~sim.people.vaccinated)
            self.placebo_inds = eligible[cv.choose(len(eligible), min(self.trial_size, len(eligible)))]
        return

pars = {
    'pop_size': 20000,
    'beta': 0.015,
    'n_days': 120,
    'verbose': -1,
}

# Define vaccine arm
trial_size = 4000
start_trial = 20

def subtarget(sim):
    ''' Select people who are susceptible '''
    if sim.t == start_trial:
        eligible = cv.true(~np.isfinite(sim.people.date_exposed))
        inds = eligible[cv.choose(len(eligible), min(trial_size//2, len(eligible)))]
    else:
        inds = []
    return {'vals': [1.0 for ind in inds], 'inds': inds}

# Initialize
sims = []
for vaccine in vaccines:
    vx = cv.vaccinate_prob(vaccine=vaccine, days=[start_trial], prob=0.0, subtarget=subtarget)
    sim = cv.Sim(
        label=vaccine,
        use_waning=True,
        pars=pars,
        interventions=vx,
        analyzers=placebo_arm(day=start_trial, trial_size=trial_size//2)
    )
    sims.append(sim)

# Run
# Run
msim = cv.MultiSim(sims)
msim.run(keep_people=True)

results = sc.objdict()
print('Vaccine efficiency:')
for sim in msim.sims:
    vaccine = sim.label
    results[vaccine] = sc.objdict()
    vacc_inds = cv.true(sim.people.vaccinated)  # Find trial arm indices, those who were vaccinated
    placebo_inds = sim['analyzers'][0].placebo_inds
    assert (len(set(vacc_inds).intersection(set(placebo_inds))) == 0)  # Check that there is no overlap
    # Calculate vaccine efficacy against infection
    VE_inf = 1 - (np.isfinite(sim.people.date_exposed[vacc_inds]).sum() /
                  np.isfinite(sim.people.date_exposed[placebo_inds]).sum())
    # Calculate vaccine efficacy against symptoms
    VE_symp = 1 - (np.isfinite(sim.people.date_symptomatic[vacc_inds]).sum() /
                   np.isfinite(sim.people.date_symptomatic[placebo_inds]).sum())
    # Calculate vaccine efficacy against severe disease
    VE_sev = 1 - (np.isfinite(sim.people.date_severe[vacc_inds]).sum() /
                  np.isfinite(sim.people.date_severe[placebo_inds]).sum())
    results[vaccine]['inf'] = VE_inf
    results[vaccine]['symp'] = VE_symp
    results[vaccine]['sev'] = VE_sev
    print(f'  {vaccine:8s}: infection: {VE_inf * 100:0.2f}%, symptoms: {VE_symp * 100:0.2f}%, severity: {VE_sev * 100:0.2f}%')

# Plot
to_plot = cv.get_default_plots('default', 'scen')
to_plot['Vaccinations'] = ['cum_vaccinated']
# msim.plot(to_plot=to_plot)

print('Done')
