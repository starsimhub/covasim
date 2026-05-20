import sciris as sc
import covasim as cv

sc.tic()

people = cv.People(pop_size=2000)

sc.toc(label='default')

plist = people.to_people()

sc.toc(label='to people')

ppl2 = cv.People()
ppl2.from_people(plist)

sc.toc(label='from people')

ppl3 = people + ppl2

sim = cv.Sim(pop_type='random', pop_size=20000)
cv.make_people(sim)
ppl4 = sim.people

sc.toc(label='as sim')

df = ppl4.to_df()
arr = ppl4.to_arr()

sc.toc(label='to df/arr')

sc.toc()

sim.people.initialize()

df = sim.people.contacts

sc.toc(label='prognoses')


#%% Test contacts creation
contacts_list, contact_keys = cv.make_random_contacts(100, {'a':10, 'b':20})


#%% Test layers

sim2 = cv.Sim(pop_type='random', use_layers=True, pop_size=500)
popdict = cv.make_randpop(sim2, microstructure=sim['pop_type'])
cv.make_people(sim2)
ppl5 = sim2.people
