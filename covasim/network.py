"""
Contact networks for Covasim on the Starsim base.

``cv.Network(ss.Network)`` reproduces Covasim's contact layers: the random
single-layer ('a') backend and the hybrid four-layer (household 'h', school 's',
work 'w', community 'c') backend. Each contact layer is one ``cv.Network``
instance (one-instance-per-layer, the hpvsim pattern; ``sim.networks`` is the
analog of Covasim's ``People.contacts``). Both backends are **static** (Covasim's
``dynam_layer`` is 0 for every default layer), so edges are built once in
``add_pairs()`` from the ported ``population.py`` builders and never change.

Per-layer transmissibility (Covasim's ``beta_layer``) is carried on the disease
(``cv.COVID``'s per-layer ``beta`` dict), not here, so the per-edge ``beta`` stays
1.0 and ``net_beta`` is inherited unchanged from ``ss.Network``.
"""
import numpy as np
import sciris as sc
import starsim as ss

from . import population as cvpop

__all__ = ['Network', 'make_networks']


# Covasim's default per-layer mean contacts (parameters.reset_layer_pars), keyed by pop_type.
_CONTACTS = {
    'random': dict(a=20),
    'hybrid': dict(h=2.0, s=20, w=16, c=20),
}

# Age windows for the hybrid school/work layers (population.make_hybrid_contacts defaults).
_SCHOOL_AGES = (6, 22)
_WORK_AGES   = (22, 65)

# Per-layer RNG offset so layers draw independent (but deterministic) streams from one seed.
_LAYER_SEED = dict(a=0, h=1, s=2, w=3, c=4)


class Network(ss.Network):
    """A single Covasim contact layer on the Starsim base.

    Args:
        layer (str): layer key -- 'a' (random/all), 'h' (household), 's' (school),
            'w' (work), or 'c' (community).
        n_contacts (float): mean contacts per person for this layer; for the household
            layer this is the Poisson cluster-size parameter (not the realized degree).
        microstructure (str): 'random' (random pairing) or 'household' (Poisson-sized,
            fully-connected disjoint clusters).
        age_range (tuple): optional (lo, hi); restrict the layer to agents with
            lo <= age < hi (the school and work layers).
    """

    def __init__(self, layer='a', n_contacts=20, microstructure='random', age_range=None, **kwargs):
        kwargs.setdefault('name', layer)   # the per-layer key in sim.networks (and the betamap key)
        kwargs.setdefault('label', layer)
        super().__init__(**kwargs)
        self.layer = layer
        self.n_contacts = n_contacts
        self.microstructure = microstructure
        self.age_range = age_range
        return

    def _rng(self):
        """Deterministic per-(seed, layer) numpy Generator.

        Network draws are distributional (degree + age-mixing), not per-agent CRN, so a
        plain seeded Generator suffices at M1; per-agent network CRN stability is revisited
        in M8 if scenario differencing needs it.
        """
        try:
            base = int(self.sim.pars.rand_seed)
        except Exception:
            base = 0
        return np.random.default_rng(base*100 + _LAYER_SEED.get(self.layer, 0))

    def add_pairs(self):
        """Build this layer's static edgelist from the ported population.py builders."""
        people = self.sim.people
        pop_size = len(people)
        rng = self._rng()

        if self.microstructure == 'household':
            edges = cvpop.make_microstructured_contacts(pop_size, self.n_contacts, rng=rng)
        elif self.age_range is not None:
            ages = np.asarray(people.age)
            lo, hi = self.age_range
            inds = sc.findinds((ages >= lo) & (ages < hi))
            edges = cvpop.make_random_contacts(len(inds), self.n_contacts, mapping=inds, rng=rng)
        else:
            edges = cvpop.make_random_contacts(pop_size, self.n_contacts, rng=rng)

        p1 = ss.uids(edges['p1'])
        p2 = ss.uids(edges['p2'])
        beta = np.ones(len(p1))
        self.append(p1=p1, p2=p2, beta=beta)
        return


def make_networks(pop_type='random', contacts=None):
    """Return the list of cv.Network instances for a Covasim population type.

    Args:
        pop_type (str): 'random' (single layer 'a') or 'hybrid' (h/s/w/c).
        contacts (dict): optional per-layer mean contacts; defaults to Covasim's values
            (parameters.reset_layer_pars).

    Returns:
        list of cv.Network, one per contact layer.
    """
    if pop_type not in _CONTACTS:
        raise ValueError(f"pop_type {pop_type!r} not supported in M1 (choices: 'random', 'hybrid').")
    cmap = sc.mergedicts(_CONTACTS[pop_type], contacts)
    if pop_type == 'random':
        return [Network('a', n_contacts=cmap['a'], microstructure='random')]
    return [
        Network('h', n_contacts=cmap['h'], microstructure='household'),
        Network('s', n_contacts=cmap['s'], microstructure='random', age_range=_SCHOOL_AGES),
        Network('w', n_contacts=cmap['w'], microstructure='random', age_range=_WORK_AGES),
        Network('c', n_contacts=cmap['c'], microstructure='random'),
    ]
