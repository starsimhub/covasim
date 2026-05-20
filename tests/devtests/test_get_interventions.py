import covasim as cv

tp = cv.test_prob(symp_prob=0.1)
cb = cv.change_beta(days=0.5, changes=0.3, label='NPI')
ct = cv.contact_tracing()
sim = cv.Sim(interventions=[ct, tp, cb]) # Wrong order to test that it raises a warning
cb = sim.get_interventions('NPI')
cb = sim.get_interventions('NP', partial=True)
cb = sim.get_interventions(cv.change_beta)
cb = sim.get_interventions(1)
cb = sim.get_interventions()
tp, cb = sim.get_interventions([0,1])
ind = sim.get_interventions(cv.change_beta, as_inds=True) # Returns [1]
sim.get_interventions('summary')
sim.initialize() # Should print warning