"""
Contact networks for Covasim, using Starsim's networks.

The "random" population type has a single random network, ``a``; the "hybrid" population type uses
``ss.HybridNet``, which becomes four networks when added to the sim: households (``h``), schools
(``s``), workplaces (``w``), and community (``c``). As in v3, the networks are static, and the number
of contacts (or the household size) is Poisson distributed. Per-layer transmissibility (v3's
``beta_layer``) is applied by the disease (``cv.COVID.pars.beta_layer``), so the edge betas are 1.
"""
import starsim as ss

from . import parameters as cvpar

__all__ = ['make_networks', 'get_contacts']


def make_networks(pop_type='random', contacts=None):
    """
    Make the networks for a Covasim population type.

    Args:
        pop_type (str): 'random' (a single network, 'a') or 'hybrid' (networks 'h', 's', 'w', and 'c')
        contacts (dict): the mean number of contacts in each network (for 'h', the household size); defaults to Covasim's values (see parameters.reset_layer_pars())

    Returns:
        A list of Starsim networks
    """
    layer_pars = dict(pop_type=pop_type, contacts=contacts)
    cvpar.reset_layer_pars(layer_pars)
    c = layer_pars['contacts']
    if pop_type == 'random':
        networks = [ss.RandomNet(name='a', n_contacts=ss.poisson(c['a']), dynamic=False, uniform_targets=True)] # As in v3, targets are chosen uniformly
    else:
        hybrid = ss.HybridNet(
            household_size = ss.poisson(c['h']),
            contacts = dict(s=c['s'], w=c['w'], c=c['c']),
            beta = dict(h=1.0, s=1.0, w=1.0, c=1.0), # Per-layer transmissibility is applied by the disease
            uniform_targets = True, # As in v3, targets are chosen uniformly
        )
        networks = [hybrid]
    return networks


def get_contacts(networks):
    """
    Get the mean number of contacts in each network (v3's ``sim['contacts']``), before or after they
    are added to a sim. For households, this is the mean household size.

    Args:
        networks (list): the networks, e.g. from make_networks()

    Returns:
        A dict of the mean number of contacts, keyed by network name
    """
    def mean(val):
        return val.pars.lam if isinstance(val, ss.poisson) else val

    contacts = {}
    for net in networks:
        if isinstance(net, ss.HybridNet): # Not yet expanded into the four networks
            contacts['h'] = mean(net.pars.household_size)
            contacts.update({key:mean(val) for key,val in net.pars.contacts.items()})
        elif isinstance(net, ss.ClusterNet):
            contacts[net.name] = mean(net.pars.cluster_size)
        elif 'n_contacts' in net.pars:
            contacts[net.name] = mean(net.pars.n_contacts)
    return contacts
