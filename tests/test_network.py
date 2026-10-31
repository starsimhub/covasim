"""
Tests for the contact networks (layers)
"""

#%% Imports and settings
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import starsim.library as ssl
import covasim as cv

pop_size = 5000
cv.options.set(interactive=False) # Assume not running interactively


#%% Define the tests

def layer_stats(sim, key):
    """ Get the ages of the people in a layer, and how many contacts they have on average """
    layer = sim.people.contacts[key]
    members = np.unique(np.concatenate([layer['p1'], layer['p2']]))
    ages = sim.people.age[members]
    n_contacts = 2*len(layer)/len(members) # Since each contact has two people
    return ages, n_contacts


def test_random():
    sc.heading('Testing random network')

    sim = cv.Sim(pop_size=pop_size, pop_type='random', verbose=0)
    sim.initialize()
    assert sim.layer_keys() == ['a'] # Just one layer, for all contacts
    assert isinstance(sim.people.contacts['a'], ss.Network) # Layers are Starsim networks

    ages, n_contacts = layer_stats(sim, 'a')
    assert len(ages) == pop_size # Everyone is in the layer
    assert np.isclose(n_contacts, sim['contacts']['a'], rtol=0.05)

    return sim


def test_hybrid():
    sc.heading('Testing hybrid network')

    sim = cv.Sim(pop_size=pop_size, pop_type='hybrid', verbose=0)
    sim.initialize()
    assert sim.layer_keys() == ['h', 's', 'w', 'c'] # Households, schools, workplaces, and community

    # People have the expected number of contacts in the school, work, and community layers
    for key in ['s', 'w', 'c']:
        ages, n_contacts = layer_stats(sim, key)
        assert np.isclose(n_contacts, sim['contacts'][key], rtol=0.05)

    # Households vary in size, so the number of contacts is less exact
    ages, n_contacts = layer_stats(sim, 'h')
    assert 1 < n_contacts < 4

    # Only children and young adults go to school, and only adults go to work
    school_ages, _ = layer_stats(sim, 's')
    work_ages, _ = layer_stats(sim, 'w')
    assert  6 <= school_ages.min() and school_ages.max() < 22
    assert 22 <= work_ages.min()   and work_ages.max()   < 65

    # By default, schools and workplaces are random networks, as in v3
    assert isinstance(sim.networks.s, ss.RandomNet) and isinstance(sim.networks.w, ss.RandomNet)

    # With clustered=True, they are clusters, with the same number of contacts
    clustered = cv.Sim(pop_size=pop_size, pop_type='hybrid', clustered=True, verbose=0).initialize()
    assert isinstance(clustered.networks.s, ssl.ClusterNet) and isinstance(clustered.networks.w, ssl.ClusterNet)
    for key in ['s', 'w']:
        ages, n_contacts = layer_stats(clustered, key)
        assert np.isclose(n_contacts, sim['contacts'][key], rtol=0.05)

    # It can also be changed before the sim is initialized, and is kept when the contacts change
    sim2 = cv.Sim(pop_size=pop_size, pop_type='hybrid', verbose=0)
    sim2['clustered'] = True
    sim2['contacts'] = dict(s=5)
    assert isinstance(sim2.initialize().networks.s, ssl.ClusterNet)

    # Schools can still be turned off
    assert len(cv.Sim(pop_size=pop_size, pop_type='hybrid', clustered=True, contacts=dict(s=0), verbose=0).initialize().networks.s) == 0

    return sim


def test_reproducibility():
    sc.heading('Testing network reproducibility')

    # The same seed gives the same network; a different seed gives a different one
    pars = dict(pop_size=pop_size, pop_type='hybrid', verbose=0)
    s1 = cv.Sim(pars, rand_seed=1).initialize()
    s2 = cv.Sim(pars, rand_seed=1).initialize()
    s3 = cv.Sim(pars, rand_seed=2).initialize()
    for key in s1.layer_keys():
        assert     np.array_equal(s1.people.contacts[key]['p2'], s2.people.contacts[key]['p2'])
        assert not np.array_equal(s1.people.contacts[key]['p2'], s3.people.contacts[key]['p2'])

    # SynthPops populations are not supported in v4
    with pytest.raises(ValueError):
        cv.Sim(pop_type='synthpops')

    return s1



#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim1 = test_random()
    sim2 = test_hybrid()
    sim3 = test_reproducibility()

    sc.toc(T)
    print('Done.')
