'''
Example script to vary relative transmissibility and susceptibility for the elderly
'''

#%% Imports and settings
import covasim as cv
import sciris as sc

pars = dict(
    n_days=120,
    )

sim = cv.Sim(pars)
sim.initialize() # This is necessary so sim.people.age is populated for the intervention...
sim.initialized = False # ...but this is necessary so the prognosis parameters are able to take effect

new_prognoses = sc.dcp(sim.pars['prognoses'])
trans_ORs     = new_prognoses['trans_ORs']
sus_ORs       = new_prognoses['sus_ORs']
age_cutoffs   = new_prognoses['age_cutoffs']

trans_ORs[age_cutoffs>=70]  *= 0.8 # Reduce relative transmissibility for people over 70 by 20%
sus_ORs[age_cutoffs>=70]    *= 0.8 # Reduce relative susceptibility for people over 70 by 20%

n_people  = sim['pop_size']
n_days    = sim['n_days']
start_day = 40

# Define the scenarios
scenarios = {
    'baseline': {
        'name': 'Baseline',
        'pars': {
        }
    },
    'protectelderly': {
        'name': 'Protect the elderly',
        'pars': {
            'prognoses': new_prognoses,
            'interventions': cv.test_num(start_day=start_day, daily_tests=[0.005*n_people]*n_days, subtarget={'inds': sim.people.age>50, 'val': 2.0}),
            }
    },
    'testelderly': {
    'name': 'Test the elderly',
    'pars': {
        'interventions': cv.test_prob(start_day=start_day, symp_prob=0.1, asymp_prob=0.005, subtarget={'inds': sim.people.age>50, 'val': 2.0}),
        }
    },
}

to_plot = [
    'cum_infections',
    'new_infections',
    'cum_diagnoses',
    'n_severe',
    'n_critical',
    'cum_deaths',
]
fig_args = dict(figsize=(24, 16))

scens = cv.Scenarios(sim=sim, scenarios=scenarios, metapars={'n_runs': 1})
scens.run(keep_people=True)
scens.plot(to_plot=to_plot, n_cols=2, fig_args=fig_args)
