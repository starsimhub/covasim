"""
Contact networks for Covasim, using Starsim's networks.

The "random" population type has a single random network, ``a``; the "hybrid" population type uses
``ssl.HybridNet``, which becomes four networks when added to the sim: households (``h``), schools
(``s``), workplaces (``w``), and community (``c``). Schools and workplaces are clusters if ``clustered=True``
and random networks, as in v3, if ``clustered=False``. As in v3, the networks are static, and the number
of contacts (or the household size) is Poisson distributed, unless a Starsim distribution is supplied
instead of a number (e.g. ``ss.nbinom()`` for overdispersed contacts). Per-layer transmissibility (v3's
``beta_layer``) is applied by the disease (``cv.COVID.pars.beta_layer``), so the edge betas are 1.

The "rank" population type has a single static network, ``a``, of type ``cv.RankNet``, whose contacts are
mostly with people who live nearby (see ``cv.make_rank_based_contacts()``).
"""
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl

from . import parameters as cvpar
from . import population as cvpop
from . import geography as cvgeo

__all__ = ['RankNet', 'make_networks', 'get_contacts']


class RankNet(ss.RandomNet):
    """
    A static rank-based geographic network: each person's contacts are mostly with people who live nearby,
    with the chance of a contact falling with how many people live closer (the nearest person has rank 1),
    as in Liben-Nowell et al. (2005), https://doi.org/10.1073/pnas.0503018102 (see ``cv.make_rank_based_contacts()``).
    Rank (not distance) means links are short in dense areas and long in sparse areas.

    The locations are the people's home locations, ``people.x`` and ``people.y`` (in degrees of longitude and latitude;
    see geography.py), which are converted to km for the distances. The edges are made once, when the sim is
    initialized; duplicate pairs (A-B and B-A) are merged, so the mean number of contacts is slightly below
    ``n_contacts`` (about 0.9 to 0.95 of it).

    Args:
        n_contacts (float/ss.Dist): the number of contacts of each person, as for ``ss.RandomNet``; a number is the mean of a Poisson distribution
        alpha (float): how strongly contacts favour nearby people (0 = ignore distance, 1.0 = the value from Liben-Nowell et al., larger = more local)
        use_workers (bool): whether to use several cores to find the contacts, choosing how many automatically (see ``cv.population.choose_workers()``)
        n_workers (int): use this many cores instead (-1 = all the cores this job may use); the result is identical whatever the number of cores
        kwargs (dict): passed to ``ss.RandomNet``

    **Example**::

        sim = cv.Sim(pop_type='rank', pop_size=5000, networks=cv.RankNet(name='a', n_contacts=10, use_workers=True))
    """
    def __init__(self, pars=None, n_contacts=20, alpha=1.0, use_workers=False, n_workers=None, **kwargs):
        if sc.isnumber(n_contacts):
            n_contacts = ss.poisson(n_contacts)
        super().__init__(n_contacts=n_contacts, dynamic=False)
        self.define_pars(alpha=alpha, use_workers=use_workers, n_workers=n_workers)
        self.update_pars(pars, **kwargs)
        return

    def get_edges(self, inds, n_contacts):
        """ Find the edges: each person in ``inds`` starts ``n_contacts`` friendships, each with the person at a random rank of distance from them """
        people = self.sim.people
        auids = people.auids
        x, y = people.x[auids], people.y[auids]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            errormsg = 'cv.RankNet needs a location for each person (people.x and people.y); see geography.py'
            raise ValueError(errormsg)
        coords = cvgeo.lonlat_to_km(np.column_stack((x, y)))
        counts = np.zeros(len(auids), dtype=np.int64) # Per agent, in the order of auids
        counts[np.searchsorted(auids, inds)] = np.broadcast_to(n_contacts, np.shape(inds))
        draws = self.dist.rng.random(int(counts.sum())) # One for the rank of each friendship
        self.dist.jump() # Reset the RNG manually, as ss.RandomNet does
        p = self.pars
        p1, p2 = cvpop.find_rank_contacts(coords, counts, draws, alpha=p.alpha, use_workers=p.use_workers, n_workers=p.n_workers)
        return auids[p1], auids[p2]


def make_networks(pop_type='random', contacts=None, clustered=None):
    """
    Make the networks for a Covasim population type.

    Args:
        pop_type (str): 'random' (a single network, 'a'), 'hybrid' (networks 'h', 's', 'w', and 'c'), or 'rank' (a single rank-based geographic network, 'a')
        contacts (dict): the number of contacts in each network (for 'h', the household size): a number is the mean of a Poisson distribution, or an ``ss.Dist`` is used as is; defaults to Covasim's values (see parameters.reset_layer_pars())
        clustered (bool): for 'hybrid', whether schools and workplaces are clusters, whose members all know each other (True), or random networks, as in v3 (False); defaults to the value in cv.make_pars()

    Returns:
        A list of Starsim networks

    **Example**::

        # Overdispersed contacts: a negative binomial with a mean of 20 and a dispersion of 0.5, instead of a Poisson
        networks = cv.make_networks(contacts=dict(a=ss.nbinom(mean=20, n=0.5)))
    """
    if clustered is None:
        clustered = cvpar.make_pars()['clustered']
    layer_pars = dict(pop_type=pop_type, contacts=contacts)
    cvpar.reset_layer_pars(layer_pars) # Fills in the default contacts for this population type, for any layers not supplied
    c = {} # A number is the mean of a Poisson distribution; a distribution is copied, so that each network has its own
    for key,val in layer_pars['contacts'].items():
        c[key] = sc.dcp(val) if isinstance(val, ss.Dist) else ss.poisson(val)
    if pop_type == 'random':
        networks = [ss.RandomNet(name='a', n_contacts=c['a'], dynamic=False, method='uniform')] # As in v3, targets are chosen uniformly
    elif pop_type == 'hybrid':
        hybrid = ssl.HybridNet(
            household_size = c['h'],
            n_contacts = dict(s=c['s'], w=c['w'], c=c['c']),
            method = 'uniform', # As in v3, targets are chosen uniformly
            clustered = clustered,
        )
        networks = [hybrid]
    elif pop_type == 'rank':
        networks = [RankNet(name='a', n_contacts=c['a'])]
    else:
        errormsg = f'pop_type "{pop_type}" not supported (choices: "random", "hybrid", "rank")'
        raise ValueError(errormsg)
    return networks


def get_contacts(networks):
    """
    Get the mean number of contacts in each network (v3's ``sim['contacts']``), before or after they
    are added to a sim. For households, this is the mean household size. Every network is included,
    so the keys are the layer keys. For a number of contacts drawn from a distribution other than Poisson
    or constant (e.g. ``ss.nbinom()``), the value is the distribution; for a network without a number of
    contacts (e.g. a ``cv.Layer``), it is None.

    Args:
        networks (list): the networks, e.g. from make_networks()

    Returns:
        A dict of the mean number of contacts, keyed by network name
    """
    def mean(val):
        if isinstance(val, ss.poisson):
            return val.pars.mean
        elif isinstance(val, ss.constant):
            return val.pars.value
        elif isinstance(val, ss.Dist):
            return val # The mean isn't available for other distributions, so return the distribution itself
        return val

    contacts = {}
    for net in networks:
        if isinstance(net, ssl.HybridNet): # Not yet expanded into the four networks
            contacts['h'] = mean(net.pars.household_size)
            contacts.update({key:mean(val) for key,val in net.pars.n_contacts.items()})
        elif isinstance(net, ssl.ClusterNet):
            contacts[net.name] = mean(net.pars.cluster_size)
        elif 'n_contacts' in net.pars:
            contacts[net.name] = mean(net.pars.n_contacts)
        else:
            contacts[net.name] = None
    return contacts
