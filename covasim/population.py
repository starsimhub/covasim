'''
Defines functions for making the population.
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as cvu
from . import defaults as cvd


# Specify all externally visible functions this file defines
__all__ = ['make_random_contacts', 'make_microstructured_contacts', 'make_hybrid_contacts']


def _tidy_edgelist(p1, p2, mapping):
    ''' Helper function to convert lists to arrays and optionally map arrays '''
    p1 = np.array(p1, dtype=cvd.default_int)
    p2 = np.array(p2, dtype=cvd.default_int)
    if mapping is not None:
        mapping = np.array(mapping, dtype=cvd.default_int)
        p1 = mapping[p1]
        p2 = mapping[p2]
    output = dict(p1=p1, p2=p2)
    return output


def make_random_contacts(pop_size, n, overshoot=1.2, dispersion=None, mapping=None, rng=None):
    '''
    Make random static contacts for a single layer as an edgelist.

    Args:
        pop_size   (int)   : number of agents to create contacts between (N)
        n          (int) : the average number of contacts per person for this layer
        overshoot  (float) : to avoid needing to take multiple Poisson draws
        dispersion (float) : if not None, use a negative binomial distribution with this dispersion parameter instead of Poisson to make the contacts
        mapping    (array) : optionally map the generated indices onto new indices
        rng        (Generator) : if supplied, a numpy random Generator used for deterministic draws (v4 Starsim port); if None, use Covasim's global-RNG helpers (v3 behavior)

    Returns:
        Dictionary of two arrays defining UIDs of the edgelist (sources and targets)

    New in 3.1.1: optimized and updated arguments.
    New in 4.0: optional ``rng`` for deterministic, global-state-free draws (Starsim port).
    '''

    # Preprocessing
    pop_size = int(pop_size) # Number of people
    p1 = [] # Initialize the "sources"
    p2 = [] # Initialize the "targets"

    # Precalculate contacts. The overshoot is used so we won't run out of contacts if the Poisson draws happen to be higher than the expected value.
    n_all_contacts = int(pop_size*n*overshoot)
    if rng is None: # v3 behavior: Covasim's global-RNG numba helpers
        all_contacts = cvu.choose_r(max_n=pop_size, n=n_all_contacts) # Choose people at random
        if dispersion is None:
            p_count = cvu.n_poisson(n, pop_size) # Draw the number of Poisson contacts for this person
        else:
            p_count = cvu.n_neg_binomial(rate=n, dispersion=dispersion, n=pop_size) # Or, from a negative binomial
    else: # v4 behavior: deterministic draws from a supplied numpy Generator
        all_contacts = rng.integers(0, pop_size, size=n_all_contacts) if pop_size > 0 else np.array([], dtype=cvd.default_int)
        if dispersion is None:
            p_count = rng.poisson(n, pop_size)
        else:
            # The negative-binomial path is not yet ported to the deterministic rng; using the global
            # helper here would silently break the determinism the rng is meant to provide, so fail loudly.
            errormsg = 'make_random_contacts: dispersion is not supported with rng (Starsim port); omit dispersion or call without rng.'
            raise NotImplementedError(errormsg)
    p_count = np.array((p_count/2.0).round(), dtype=cvd.default_int)

    # Make contacts
    count = 0
    for p in range(pop_size):
        n_contacts = p_count[p]
        these_contacts = all_contacts[count:count+n_contacts] # Assign people
        count += n_contacts
        p1.extend([p]*n_contacts)
        p2.extend(these_contacts)

    # Tidy up
    output = _tidy_edgelist(p1, p2, mapping)

    return output


def make_microstructured_contacts(pop_size, cluster_size, mapping=None, rng=None):
    '''
    Create microstructured contacts -- i.e. for households.

    Args:
        pop_size (int): total number of people
        cluster_size (int): the average size of each cluster (Poisson-sampled)
        mapping (array): optionally map the generated indices onto new indices
        rng (Generator): if supplied, a numpy random Generator for deterministic draws (v4 Starsim port); if None, use Covasim's global RNG (v3 behavior)

    New in version 3.1.1: optimized updated arguments.
    New in 4.0: optional ``rng`` for deterministic, global-state-free draws (Starsim port).
    '''

    # Preprocessing -- same as above
    pop_size = int(pop_size) # Number of people
    p1 = [] # Initialize the "sources"
    p2 = [] # Initialize the "targets"

    # Initialize
    n_remaining = pop_size # Make clusters - each person belongs to one cluster

    # Loop over the clusters
    cluster_id = -1
    while n_remaining > 0:
        cluster_id += 1 # Assign cluster id
        this_cluster = int(cvu.poisson(cluster_size) if rng is None else rng.poisson(cluster_size))  # Sample the cluster size
        if this_cluster > n_remaining:
            this_cluster = n_remaining

        # Indices of people in this cluster
        cluster_indices = (pop_size-n_remaining) + np.arange(this_cluster)
        for source in cluster_indices: # Add symmetric pairwise contacts in each cluster
            targets = set()
            for target in cluster_indices:
                if target > source:
                    targets.add(target)
            p1.extend([source]*len(targets))
            p2.extend(list(targets))

        n_remaining -= this_cluster

    # Tidy up
    output = _tidy_edgelist(p1, p2, mapping)

    return output


def make_hybrid_contacts(pop_size, ages, contacts, school_ages=None, work_ages=None, rng=None):
    '''
    Create "hybrid" contacts -- microstructured contacts for households and
    random contacts for schools and workplaces, both of which have extremely
    basic age structure. A combination of both make_random_contacts() and
    make_microstructured_contacts().

    The optional ``rng`` (a numpy random Generator) is threaded through to the
    sub-builders for deterministic, global-state-free draws (v4 Starsim port).
    '''

    # Handle inputs and defaults
    contacts = sc.mergedicts({'h':4, 's':20, 'w':20, 'c':20}, contacts) # Ensure essential keys are populated
    if school_ages is None:
        school_ages = [6, 22]
    if work_ages is None:
        work_ages   = [22, 65]

    contacts_dict = {}

    # Start with the household contacts for each person
    contacts_dict['h'] = make_microstructured_contacts(pop_size, contacts['h'], rng=rng)

    # Make community contacts
    contacts_dict['c'] = make_random_contacts(pop_size, contacts['c'], rng=rng)

    # Get the indices of people in each age bin
    ages = np.array(ages)
    s_inds = sc.findinds((ages >= school_ages[0]) * (ages < school_ages[1]))
    w_inds = sc.findinds((ages >= work_ages[0])   * (ages < work_ages[1]))

    # Create the school and work contacts for each person
    contacts_dict['s'] = make_random_contacts(len(s_inds), contacts['s'], mapping=s_inds, rng=rng)
    contacts_dict['w'] = make_random_contacts(len(w_inds), contacts['w'], mapping=w_inds, rng=rng)

    return contacts_dict

