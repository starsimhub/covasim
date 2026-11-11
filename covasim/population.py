"""
Defines functions for making contact edge lists, e.g. for custom layers.

These are not used by cv.Sim, which creates its networks with Starsim (see network.py), except
for the rank-based network, whose edges ``cv.RankNet`` finds with ``find_rank_contacts()``.
They return a dict of ``p1`` and ``p2`` arrays that can be used to make a layer, e.g.
``cv.Layer(**cv.make_random_contacts(pop_size, 10), label='random')``. They use Covasim's
global random number generator, so call ``cv.set_seed()`` first for reproducible edges.
"""

#%% Imports
import numpy as np
import sciris as sc
from . import utils as cvu
from . import defaults as cvd
from . import misc as cvm


# Specify all externally visible functions this file defines
__all__ = ['make_random_contacts', 'make_microstructured_contacts', 'make_hybrid_contacts', 'make_rank_based_contacts', 'find_rank_contacts']


def _tidy_edgelist(p1, p2, mapping):
    """ Helper function to convert lists to arrays and optionally map arrays """
    p1 = np.array(p1, dtype=cvd.default_int)
    p2 = np.array(p2, dtype=cvd.default_int)
    if mapping is not None:
        mapping = np.array(mapping, dtype=cvd.default_int)
        p1 = mapping[p1]
        p2 = mapping[p2]
    output = dict(p1=p1, p2=p2)
    return output


def make_random_contacts(pop_size, n, overshoot=1.2, dispersion=None, mapping=None):
    """
    Make random static contacts for a single layer as an edgelist.

    Args:
        pop_size   (int)   : number of agents to create contacts between (N)
        n          (int) : the average number of contacts per person for this layer
        overshoot  (float) : to avoid needing to take multiple Poisson draws
        dispersion (float) : if not None, use a negative binomial distribution with this dispersion parameter instead of Poisson to make the contacts
        mapping    (array) : optionally map the generated indices onto new indices

    Returns:
        Dictionary of two arrays defining UIDs of the edgelist (sources and targets)

    New in 3.1.1: optimized and updated arguments.
    """

    # Preprocessing
    pop_size = int(pop_size) # Number of people
    p1 = [] # Initialize the "sources"
    p2 = [] # Initialize the "targets"

    # Precalculate contacts. The overshoot is used so we won't run out of contacts if the Poisson draws happen to be higher than the expected value.
    n_all_contacts = int(pop_size*n*overshoot)
    all_contacts = cvu.choose_r(max_n=pop_size, n=n_all_contacts) # Choose people at random
    if dispersion is None:
        p_count = cvu.n_poisson(n, pop_size) # Draw the number of Poisson contacts for this person
    else:
        p_count = cvu.n_neg_binomial(rate=n, dispersion=dispersion, n=pop_size) # Or, from a negative binomial
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


def make_microstructured_contacts(pop_size, cluster_size, mapping=None):
    """
    Create microstructured contacts -- i.e. for households.

    Args:
        pop_size (int): total number of people
        cluster_size (int): the average size of each cluster (Poisson-sampled)
        mapping (array): optionally map the generated indices onto new indices

    New in version 3.1.1: optimized updated arguments.
    """

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
        this_cluster = cvu.poisson(cluster_size)  # Sample the cluster size
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


def make_hybrid_contacts(pop_size, ages, contacts, school_ages=None, work_ages=None):
    """
    Create "hybrid" contacts -- microstructured contacts for households and
    random contacts for schools and workplaces, both of which have extremely
    basic age structure. A combination of both make_random_contacts() and
    make_microstructured_contacts().
    """

    # Handle inputs and defaults
    contacts = sc.mergedicts({'h':4, 's':20, 'w':20, 'c':20}, contacts) # Ensure essential keys are populated
    if school_ages is None:
        school_ages = [6, 22]
    if work_ages is None:
        work_ages   = [22, 65]

    contacts_dict = {}

    # Start with the household contacts for each person
    contacts_dict['h'] = make_microstructured_contacts(pop_size, contacts['h'])

    # Make community contacts
    contacts_dict['c'] = make_random_contacts(pop_size, contacts['c'])

    # Get the indices of people in each age bin
    ages = np.array(ages)
    s_inds = sc.findinds((ages >= school_ages[0]) * (ages < school_ages[1]))
    w_inds = sc.findinds((ages >= work_ages[0])   * (ages < work_ages[1]))

    # Create the school and work contacts for each person
    contacts_dict['s'] = make_random_contacts(len(s_inds), contacts['s'], mapping=s_inds)
    contacts_dict['w'] = make_random_contacts(len(w_inds), contacts['w'], mapping=w_inds)

    return contacts_dict


# The rank-based network -----

shared = {} # The coordinates, shared with worker processes that are started by fork (so they are not sent again with every job)
min_agents_per_worker = 2000 # A rough guide: fewer agents than this per worker and starting the workers costs more than it saves
auto_max_workers = 16 # use_workers=True never picks more than this: on the test server (100,000 agents) efficiency stayed above about 80% up to 16 workers


def choose_workers(pop_size, use_workers=False, n_workers=None):
    """
    Helper for find_rank_contacts(): how many worker processes to use. One means no parallelism.

        - n_workers given: use that many (-1 = every core this job may use), but never more than the cores available or the size justifies (about 2,000 agents per worker)
        - use_workers=True: choose automatically: leave one core free, never more than 16 (more cores help less and less), and only count cores that are not already busy with other work
        - neither: one core
    """
    import os
    usable = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1) # Cores this job may really use
    if n_workers is not None:
        wanted = usable if n_workers < 0 else int(n_workers)
    elif use_workers:
        wanted = usable - 1 if usable > 2 else usable
        wanted = min(wanted, auto_max_workers)
        try:
            wanted = min(wanted, int(usable - os.getloadavg()[0])) # Cores already busy with other work are not free
        except (AttributeError, OSError): # getloadavg does not exist on every platform
            pass
    else:
        wanted = 1
    return max(1, min(wanted, usable, pop_size//min_agents_per_worker))


def pick_ranks(d, ranks):
    """
    Helper for find_rank_contacts(): given the squared distances d from one person to everyone, return who sits
    at each wanted rank (1 = nearest). The largest rank is found first with one partial sort; the smaller ranks are then
    searched only among the people already known to be closer, so each later step works on a smaller list.
    """
    uniq, inverse = np.unique(ranks, return_inverse=True) # Wanted ranks, smallest first (a rank asked for twice is found once)
    idx   = np.arange(len(d)) # Who each entry of d belongs to (d gets reordered below)
    found = np.empty(len(uniq), dtype=np.int64)
    for j in range(len(uniq)-1, -1, -1): # Largest rank first
        r     = uniq[j]
        part  = np.argpartition(d, r-1) # Puts the r-th nearest at position r-1, the r nearest before it
        idx, d = idx[part], d[part]
        found[j] = idx[r-1]
        idx, d = idx[:r], d[:r] # The smaller ranks can only be among these r people
    return found[inverse]


def find_friends(x, y, people, counts, ranks):
    """
    Helper for find_rank_contacts(): for each person in `people`, find the person at each of their wanted ranks.
    ranks holds the wanted ranks of all these people one after another (counts[i] of them for people[i]).
    """
    friends = np.empty(len(ranks), dtype=np.int64)
    dx = np.empty(len(x)) # Reused for every person, so no new arrays are made inside the loop
    dy = np.empty(len(x))
    start = 0
    for p, k in zip(people, counts):
        np.subtract(x, x[p], out=dx)
        np.subtract(y, y[p], out=dy)
        np.multiply(dx, dx, out=dx)
        np.multiply(dy, dy, out=dy)
        np.add(dx, dy, out=dx) # Squared distance from person p to everyone (squared is fine for ordering)
        dx[p] = np.inf # Put self last so it can never be chosen
        friends[start:start+k] = pick_ranks(dx, ranks[start:start+k])
        start += k
    return friends


def friends_job(people, counts, ranks):
    """
    Helper for find_rank_contacts(): one job for a worker process. The coordinates come from `shared`, which the
    worker inherited when it was forked, so only the small per-job arrays have to be sent to it.
    """
    return find_friends(shared['x'], shared['y'], people, counts, ranks)


def find_rank_contacts(coords, counts, draws, alpha=1.0, use_workers=False, n_workers=None):
    """
    Find the edges of a rank-based friendship network, given how many friendships each person starts and a
    uniform random number for each (see make_rank_based_contacts() for the model). Used by ``cv.RankNet``
    and make_rank_based_contacts().

    Args:
        coords (array): (N,2) array of x,y locations in km, one row per person
        counts (array): the number of friendships each person starts
        draws (array): a uniform random number for each friendship, ``counts.sum()`` of them, in order of person
        alpha (float): how strongly friendships favour nearby people
        use_workers (bool): whether to choose a number of cores automatically (see choose_workers())
        n_workers (int): use this many cores instead (-1 = all the cores this job may use); the result is identical whatever the number of cores

    Returns:
        Two arrays, p1 and p2, of the positions (rows of ``coords``) of the two people in each friendship, with duplicates (A-B and B-A) merged
    """
    coords   = np.asarray(coords, dtype=float)
    pop_size = len(coords)
    counts   = np.asarray(counts, dtype=np.int64)
    empty    = np.array([], dtype=np.int64)
    if pop_size < 2 or counts.sum() == 0: # Nobody to befriend
        return empty, empty

    # Turn the random numbers into ranks: rank 1 = nearest person, rank pop_size-1 = farthest, and the weight of rank r is 1/r^alpha
    rank_weights = np.arange(1, pop_size, dtype=float)**(-alpha)
    rank_cdf     = np.cumsum(rank_weights) # Running total, used to turn random numbers into ranks
    rank_cdf    /= rank_cdf[-1]
    people = np.flatnonzero(counts > 0) # Who starts any friendships
    counts = counts[people]
    ranks  = np.searchsorted(rank_cdf, draws) + 1
    ranks  = np.minimum(ranks, pop_size-1) # Guard against rounding at the far end

    # Find who sits at each chosen rank, on one core or several
    x, y = np.ascontiguousarray(coords[:, 0]), np.ascontiguousarray(coords[:, 1])
    n_workers = choose_workers(pop_size, use_workers, n_workers) # One core unless asked otherwise
    friends = None
    if n_workers > 1:
        import multiprocessing as mp
        if 'fork' in mp.get_all_start_methods(): # Forked workers inherit the coordinates; other platforms use one core
            from concurrent.futures import ProcessPoolExecutor
            ctx    = mp.get_context('fork')
            ends   = np.cumsum(counts)
            starts = ends - counts
            groups = np.array_split(np.arange(len(people)), n_workers*4) # Several small jobs per core keeps all cores busy
            jobs   = [g for g in groups if len(g)]
            shared['x'], shared['y'] = x, y
            try:
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                    futures = [pool.submit(friends_job, people[g], counts[g], ranks[starts[g[0]]:ends[g[-1]]]) for g in jobs]
                    friends = np.concatenate([f.result() for f in futures])
            except (AssertionError, OSError, RuntimeError) as E: # e.g. already inside a worker process (MultiSim), or not allowed to start processes
                cvm.warn(f'Could not start worker processes for the rank network ({E}); using one core instead')
                friends = None
            finally:
                shared.clear()
    if friends is None: # One core: the default, and the fallback
        friends = find_friends(x, y, people, counts, ranks)

    # Merge duplicate pairs (A-B and B-A are the same friendship)
    p1 = np.repeat(people, counts)
    p2 = friends
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    pairs  = np.unique(lo*pop_size + hi) # One number per pair, so unique() is much faster than comparing rows
    return pairs//pop_size, pairs % pop_size


def make_rank_based_contacts(coords, n, alpha=1.0, dispersion=None, mapping=None, use_workers=False, n_workers=None):
    """
    Make a rank-based geographic friendship layer as an edgelist.

    Model: Liben-Nowell et al. (2005), "Geographic routing in social networks", https://doi.org/10.1073/pnas.0503018102.
    Person u links to person v with probability proportional to

        P(u -> v)  ~  1 / rank_u(v)^alpha,    alpha = 1 in the paper

    where rank_u(v) is the number of people w (w != u) who are at least as close to u
    as v is. Rank (not distance) means links are short in dense areas and long in
    sparse areas.

    Args:
        coords     (array) : (N,2) array of x,y locations in km, one row per person
        n          (float) : the average number of friends per person for this layer
        alpha      (float) : how strongly friendships favour nearby people (0 = ignore
                             distance, 1.0 = the value from Liben-Nowell et al., larger = more local)
        dispersion (float) : if not None, use a negative binomial distribution with this
                             dispersion parameter instead of Poisson for friend counts
        mapping    (array) : optionally map the generated indices onto new indices
        use_workers (bool) : False (the default) uses one core, which works everywhere. True lets the code choose how many
                             cores to use for this computer and this size (see choose_workers()).
        n_workers  (int)   : use this many cores instead (-1 = all the cores this job may use). Overrides use_workers.
                             The result is identical whatever the number of cores.

    Returns:
        Dictionary of two arrays, p1 and p2, defining the edgelist (friend pairs), in the
        same format as make_random_contacts() so it can be used as a contact layer.

    Method, for each person p:
        1. Draw how many friends p picks: a random (Poisson) number averaging n/2. Every
           friendship is picked by one of its two people, so a person's total comes from the
           n/2 they pick plus about n/2 picked by others, about n friends in all.
        2. For each friendship, pick a rank r from 1 to N-1. Small ranks are much more
           likely than large ones (probability proportional to 1/r^alpha).
        3. Connect p to the person who is r-th nearest to them (found by partially sorting the
           distances from p to everyone, not a full sort).
        4. Finally, pairs found twice (A-B and B-A, or A-B repeated) are merged into one friendship.
           This is why the average number of friends comes out slightly below n (about 0.9n to 0.95n).

    The cost grows with N^2 (about 2 s for 10,000 people on one core). Like the other functions in this
    file, it uses Covasim's global random number generator.
    """
    coords   = np.asarray(coords, dtype=float)
    pop_size = len(coords)
    if dispersion is None:
        p_count = cvu.n_poisson(n, pop_size) # Draw the number of Poisson contacts for this person
    else:
        p_count = cvu.n_neg_binomial(rate=n, dispersion=dispersion, n=pop_size) # Or, from a negative binomial
    p_count = np.array((p_count/2.0).round(), dtype=cvd.default_int) # Halve: each friendship has two ends
    draws = np.random.random(int(p_count.sum())) # One for the rank of each friendship
    p1, p2 = find_rank_contacts(coords, p_count, draws, alpha=alpha, use_workers=use_workers, n_workers=n_workers)
    output = _tidy_edgelist(p1, p2, mapping)
    return output
