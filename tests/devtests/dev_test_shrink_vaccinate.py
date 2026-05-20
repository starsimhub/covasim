'''
shrink vaccination intervention
'''
import covasim as cv
import sciris as sc

cv.check_version('>3.0.0', die=True)


pars = {'pop_size':50000, 'use_waning':True,}

for vaccine in ['jj', 'pfizer']:
    print('\nFor ' + vaccine)

    sim = cv.Sim(pars)
    sim['interventions'] = [cv.vaccinate(vaccine='jj', days=10, prob=0.4)]
    sim.run(verbose=False);

    print('Sim size is:')
    sc.checkmem(sim, descend=False)

    sim.shrink()

    print('Shrunk size is:')
    sc.checkmem(sim, descend=False)

    print('Shrunk vaccination')
    sim['interventions'][0].shrink()
    sc.checkmem(sim, descend=False)

    print('No vaccination')
    sim['interventions'][0] = None
    sc.checkmem(sim, descend=False)