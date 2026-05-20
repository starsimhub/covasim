'''
Include an extra contact network -- in this example, assisted living facilities.
'''

import covasim as cv
import sciris as sc
import numpy as np
import pandas as pd
from covasim import base as cvb

'''
Define ALF parameters
According to LTC 2015-2016 data for all of Washington:
83% of residents in nursing homes are 65> and 95% of residents in residential care communities are 65>.
Therefore we set the minimum age to 70.
The average size of a nursing home is 76 and average size of a residential care community is 18.
There are 200 nursing homes in all of Washington and 2,000 residential care communities, so we
set the average size of an ALF to 25.
There are approximately 51,200 people residing in ALFs in Washington. We compared this to the number of
people living above 60 in Washington State (132,021) to conclude that 38% of these people are living in ALFs.
In the 2,200 combined facilities, there are 23,569 nursing and social work staff employees.
Therefore, we assume each facility has on average 10 health care staff.
'''

min_alf_age = 70
n_alf_contacts = 25
alf_prob = 0.38
n_alf_staff = 10


def remove_contacts_from_layer(layer, inds, sim):
    ''' Removes a set of contacts, specified by index, from a certain layer '''
    layer_to_change = sim.people.contacts[layer]  # extract home layer contacts to adjust
    layer_to_change_df = pd.DataFrame(layer_to_change)  # turn it into a DF
    for person in inds:  # look through each alf, remove them from any pairwise contacts
        layer_to_change_df.drop(layer_to_change_df[layer_to_change_df["p1"] == person].index, inplace=True)
        layer_to_change_df.drop(layer_to_change_df[layer_to_change_df["p2"] == person].index, inplace=True)

    new_layer = cvb.Layer().from_df(layer_to_change_df)  # Convert DF back with Layers (from base class)
    new_layer.validate()
    sim.people.contacts[layer] = new_layer  # reassign home layer contacts as new home layer

    return


sim = cv.Sim(pop_type='hybrid', pop_size=2e4)
sim.initialize()

# Set alf parameters to look like household
for key in ['contacts', 'dynam_layer', 'beta_layer', 'quar_eff']:
    sim[key]['alf'] = sim[key]['h']

pop_size = sim.pars['pop_size']

# Create the contacts for alf residents
layer_keys = ['alf']
alf_inds = cv.binomial_filter(alf_prob, sc.findinds(sim.people.age >= min_alf_age))
assert max(alf_inds) < pop_size
contacts_list = [{key: [] for key in layer_keys} for i in range(pop_size)]
alf_contacts, _, clusters = cv.make_microstructured_contacts(len(alf_inds), {'alf': n_alf_contacts})
alf_dict = clusters['alf']

for i, ind in enumerate(alf_inds):
    contacts_list[ind]['alf'] = alf_inds[alf_contacts[i]['alf']].tolist()  # Copy over alf contacts
    if len(contacts_list[ind]['alf']):
        assert max(contacts_list[ind]['alf']) < pop_size

# find the cluster that individual is in and replace with correct index
for id, alf_members in alf_dict.items():
    alf_dict[id] = sc.dcp(alf_inds[alf_members]).tolist()
    assert max(alf_dict[id]) < pop_size

# Clip edges of ALF residents in home and community
alf_inds = alf_inds.tolist()
remove_contacts_from_layer('h', alf_inds, sim)

# Create list of health care workers
health_care_worker_age = [22, 55]
health_care_worker_inds = sc.findinds(
    (sim.people.age >= health_care_worker_age[0]) * (sim.people.age < health_care_worker_age[1]))
assert max(health_care_worker_inds) < pop_size

# Loop through all ALFs, select a number of health care workers from health_care_worker_inds, add them to ALF and
# to list of contacts, then remove them from health_care_worker_inds; remove them from workplace and workplace contacts.

health_care_aides = [] # List of health care aides. will use this later to clip their edges in the workplace

for id, cluster_members in alf_dict.items():
    num_health_care_workers = cv.poisson(n_alf_staff)
    health_care_workers_to_add = np.random.choice(health_care_worker_inds, num_health_care_workers, replace=False).tolist()
    assert max(health_care_workers_to_add) < pop_size
    for person in health_care_workers_to_add:
        health_care_aides.append(person)
        contacts_list[person]['alf'] += cluster_members
        assert max(contacts_list[person]['alf']) < pop_size
        cluster_members.append(person)
        for alf_member in cluster_members:
            contacts_list[alf_member]['alf'] += [person]
            assert max(contacts_list[alf_member]['alf']) < pop_size
    for person in range(len(health_care_workers_to_add)):
        health_care_worker_inds = health_care_worker_inds[health_care_worker_inds != health_care_workers_to_add[person]]
        assert max(health_care_worker_inds) < pop_size

# Clip health care aides edges in the workplace
remove_contacts_from_layer('w', health_care_aides, sim)

sim.people.add_contacts(contacts_list, lkey='alf')
print('Total alf contacts: ', len(sim.people.contacts['alf']))

# Run
sim.run()
sim.plot()


