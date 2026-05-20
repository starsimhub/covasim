import numpy as np
import sciris as sc
assert sc.__version__ > '1.2.0'
import covasim as cv

if __name__ == '__main__':

    pars = sc.objdict(
        pop_infected=0,
        n_agents=1e5,
        pop_scale=10,
        start_day='2021-06-01',
        end_day='2021-10-01',
        use_waning=True,
        variants=cv.variant('delta', days='2021-06-01', n_imports=10, rescale=False),
    )

    sims = []

    for scen in ['baseline', 'vx_extra_pfizer', 'vx_extra_az']:

        vxdict = {}


        def age_sequence(people):
            return np.argsort(people.age)


        if 'extra' in scen:
            total_doses = 2.0e6
        else:
            total_doses = 0.0e5
        extra_days = sc.daterange('2021-07-11', '2021-07-20')
        doses = int(total_doses / len(extra_days))
        for day in extra_days:
            vxdict[day] = doses

        vax = 'az' if 'az' in scen else 'pfizer'
        vx = cv.vaccinate_num(vaccine=vax, num_doses=vxdict,
                              sequence=age_sequence, label=f'{scen}')  # Most vaccinations have been AstraZeneca to date

        sim = cv.Sim(pars=pars,
                     interventions=[vx],
                     label=f'{scen}',
                     )
        sims.append(sim)

    msim = cv.MultiSim(sims)
    msim.run()
    to_plot = cv.get_default_plots('default', 'scen')
    to_plot['Doses'] = ['cum_doses']
    to_plot['Infections'] = ['cum_infections']
    msim.plot(to_plot=to_plot)
