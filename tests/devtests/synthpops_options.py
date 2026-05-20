'''
Script for testing integration with new SynthPops features -- namely, long-term
care facilities.
'''

import covasim as cv

which = ['simple', 'complex'][1]

if which == 'simple':

    sim = cv.Sim(pop_size=5000, pop_type='synthpops')
    sim.popdict = cv.make_synthpop(sim, with_facilities=True, layer_mapping={'LTCF':'f'})
    sim.reset_layer_pars()
    sim.run()


else:

    pars = dict(
        pop_size=20000,
        pop_type='synthpops',
        n_days=120,
        )

    sims = []
    for ltcf in [False, True]:
        print(f'Running LTCF {ltcf}')
        sim = cv.Sim(pars)
        sim.popdict = cv.make_synthpop(sim, with_facilities=ltcf, layer_mapping={'LTCF':'f'})
        sim.reset_layer_pars()
        sims.append(sim)

    to_plot = ['cum_infections', 'new_infections', 'cum_severe', 'cum_deaths']
    msim = cv.MultiSim(sims)
    msim.run()
    msim.plot(to_plot=to_plot)
