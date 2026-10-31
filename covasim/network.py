"""
Contact networks for Covasim, using Starsim's networks.

The "random" population type has a single random network, ``a``; the "hybrid" population type uses
``ssl.HybridNet``, which becomes four networks when added to the sim: households (``h``), schools
(``s``), workplaces (``w``), and community (``c``). Schools and workplaces are clusters if ``clustered=True``
and random networks, as in v3, if ``clustered=False``. As in v3, the networks are static, and the number
of contacts (or the household size) is Poisson distributed. Per-layer transmissibility (v3's
``beta_layer``) is applied by the disease (``cv.COVID.pars.beta_layer``), so the edge betas are 1.
"""
import starsim as ss
import starsim.library as ssl

from . import parameters as cvpar

__all__ = ['make_networks', 'get_contacts']


def make_networks(pop_type='random', contacts=None, clustered=None):
    """
    Make the networks for a Covasim population type.

    Args:
        pop_type (str): 'random' (a single network, 'a') or 'hybrid' (networks 'h', 's', 'w', and 'c')
        contacts (dict): the mean number of contacts in each network (for 'h', the household size); defaults to Covasim's values (see parameters.reset_layer_pars())
        clustered (bool): for 'hybrid', whether schools and workplaces are clusters, whose members all know each other (True), or random networks, as in v3 (False); defaults to the value in cv.make_pars()

    Returns:
        A list of Starsim networks
    """
    if clustered is None:
        clustered = cvpar.make_pars()['clustered']
    layer_pars = dict(pop_type=pop_type, contacts=contacts)
    cvpar.reset_layer_pars(layer_pars) # Fills in the default contacts for this population type, for any layers not supplied
    c = layer_pars['contacts']
    if pop_type == 'random':
        networks = [ss.RandomNet(name='a', n_contacts=ss.poisson(c['a']), dynamic=False, method='uniform')] # As in v3, targets are chosen uniformly
    elif pop_type == 'hybrid':
        hybrid = ssl.HybridNet(
            household_size = ss.poisson(c['h']),
            contacts = dict(s=c['s'], w=c['w'], c=c['c']),
            beta = dict(h=1.0, s=1.0, w=1.0, c=1.0), # Per-layer transmissibility is applied by the disease
            method = 'uniform', # As in v3, targets are chosen uniformly
            clustered = clustered,
        )
        networks = [hybrid]
    else:
        errormsg = f'pop_type "{pop_type}" not supported (choices: "random", "hybrid")'
        raise ValueError(errormsg)
    return networks


def get_contacts(networks):
    """
    Get the mean number of contacts in each network (v3's ``sim['contacts']``), before or after they
    are added to a sim. For households, this is the mean household size. Every network is included,
    so the keys are the layer keys; the value is None for a network without a mean number of contacts
    (e.g. a ``cv.Layer``, or a number of contacts drawn from a distribution other than Poisson).

    Args:
        networks (list): the networks, e.g. from make_networks()

    Returns:
        A dict of the mean number of contacts, keyed by network name
    """
    def mean(val):
        if isinstance(val, ss.poisson):
            return val.pars.lam
        elif isinstance(val, ss.constant):
            return val.pars.v
        elif isinstance(val, ss.Dist):
            return None # The mean isn't available for other distributions
        return val

    contacts = {}
    for net in networks:
        if isinstance(net, ssl.HybridNet): # Not yet expanded into the four networks
            contacts['h'] = mean(net.pars.household_size)
            contacts.update({key:mean(val) for key,val in net.pars.contacts.items()})
        elif isinstance(net, ssl.ClusterNet):
            contacts[net.name] = mean(net.pars.cluster_size)
        elif 'n_contacts' in net.pars:
            contacts[net.name] = mean(net.pars.n_contacts)
        else:
            contacts[net.name] = None
    return contacts
