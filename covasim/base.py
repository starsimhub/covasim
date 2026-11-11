"""
The v3 contact layers (``cv.Layer``, now a Starsim network, and ``cv.Contacts``), and the v3
``cv.Result`` (now Starsim's).
"""

import sciris as sc
import starsim as ss

# Specify all externally visible classes this file defines
__all__ = ['Result', 'FlexDict', 'Contacts', 'Layer']

# Results are now Starsim results
Result = ss.Result


class FlexDict(dict):
    """
    A dict that allows more flexible element access: in addition to obj['a'],
    also allow obj[0]. Lightweight implementation of the Sciris odict class.
    """

    def __getitem__(self, key):
        """ Lightweight odict -- allow indexing by number, with low performance """
        try:
            return super().__getitem__(key)
        except KeyError as KE:
            try: # Assume it's an integer
                dictkey = self.keys()[key]
                return self[dictkey]
            except Exception:
                raise sc.KeyNotFoundError(KE) # Raise the original error

    def keys(self):
        return list(super().keys())

    def values(self):
        return list(super().values())

    def items(self):
        return list(super().items())


class Layer(ss.Network):
    """
    A v3 contact layer: a static Starsim network, i.e. one whose edges don't change unless modified,
    e.g. by an intervention. See ``ss.Network`` for details.

    **Example**::

        n = 10_000
        n_people = 1000
        p1 = np.random.randint(n_people, size=n)
        p2 = np.random.randint(n_people, size=n)
        weight = np.ones(n)
        layer = cv.Layer(p1=p1, p2=p2, weight=weight, label='rand')

    As in v3, the weight of each edge can also be given as ``beta``.
    """
    def __init__(self, pars=None, edges=None, **kwargs):
        edges = dict(edges) if edges is not None else {} # dict() also handles dataframes
        if 'beta' in kwargs: # v3's name for the weight of each edge
            kwargs['weight'] = kwargs.pop('beta')
        if 'beta' in edges:
            edges['weight'] = edges.pop('beta')
        super().__init__(pars=pars, edges=edges, **kwargs)
        return

    def step(self):
        """ The edges are static """
        pass

    def keys(self):
        """ v3: a layer was a dict of its edge arrays (p1, p2, weight, and any others), so e.g. cv.Layer(**layer) copies it """
        return list(self.edges.keys())


def to_layer(lkey, layer):
    """ Return the layer as a network named lkey: a network is used as it is, and a dict of edges (p1, p2, and weight, or v3's beta) becomes a cv.Layer """
    if isinstance(layer, ss.Network):
        if layer.name != lkey:
            layer.name = lkey
        return layer
    return Layer(edges=layer, name=lkey, label=lkey)


class Contacts(FlexDict):
    """
    The contact layers (i.e. networks), keyed by name, e.g. ``sim.people.contacts['h']``.

    ``sim.people.contacts`` is made from the sim's networks each time it is read. Adding a layer
    adds a network to the sim (after it has been initialized, but before it has been run); layers
    can't be removed from or replaced in a sim this way, so pass ``networks`` to ``cv.Sim()`` instead.

    Args:
        data (dict): a dictionary of layers (or of dicts of p1, p2, and weight arrays)
        layer_keys (list): if provided, create an empty Contacts object with these layers
        sim (Sim): if provided, the sim whose networks these are
        kwargs (dict): additional layer(s), merged with data

    New in version 4.0.0: the layers are Starsim networks; ``sim`` argument.
    """
    def __init__(self, data=None, layer_keys=None, sim=None, **kwargs):
        self.sim = None # Set at the end, so that adding the layers here doesn't raise an error in __setitem__()
        data = sc.mergedicts(data, kwargs)
        for lkey in sc.tolist(layer_keys):
            self[lkey] = Layer(name=lkey, label=lkey)
        for lkey,layer in data.items():
            self[lkey] = to_layer(lkey, layer)
        self.sim = sim
        return

    def __setitem__(self, key, layer):
        """ A sim's layers can't be replaced, since the sim has already been set up with them """
        if self.sim is not None:
            if key in self.sim.networks:
                errormsg = f'Cannot replace layer "{key}" in the sim; please create the sim with the layers you want, e.g. cv.Sim(networks=...)'
                raise ValueError(errormsg)
            else:
                errormsg = f'To add layer "{key}" to the sim, please use sim.people.contacts.add_layer({key}=layer)'
                raise ValueError(errormsg)
        return super().__setitem__(key, layer)

    @classmethod
    def from_sim(cls, sim):
        """ Create the contacts from the sim's networks """
        return cls(data=dict(sim.networks.items()), sim=sim)

    def __repr__(self):
        """ Use slightly customized repr"""
        keys_str = ', '.join([str(k) for k in self.keys()])
        output = f'Contacts({keys_str})\n'
        for key in self.keys():
            output += f'\n"{key}": '
            output += self[key].__repr__() + '\n'
        return output

    def __len__(self):
        """ The length of the contacts is the length of all the layers """
        return sum([len(layer) for layer in self.values()])

    def add_layer(self, **kwargs):
        """
        Small method to add one or more layers to the contacts. Layers should
        be provided as keyword arguments.

        **Example**::

            hospitals_layer = cv.Layer(label='hosp')
            sim.people.contacts.add_layer(hospitals=hospitals_layer)
        """
        for lkey,layer in kwargs.items():
            layer = to_layer(lkey, layer)
            layer.validate()
            if self.sim is not None:
                self.sim._add_network(layer)
            dict.__setitem__(self, lkey, layer) # Not self[lkey] = layer, which is only for contacts that aren't from a sim
        return

    def pop_layer(self, *args):
        """
        Remove the layer(s) from the contacts. Layers can't be removed from a sim, so this
        is only for contacts that aren't from a sim.

        **Example**::

            contacts = cv.Contacts(a=dict(p1=[0,1], p2=[1,2], weight=[1,1]), b=dict(p1=[2], p2=[3], weight=[1]))
            contacts.pop_layer('b')
        """
        if self.sim is not None:
            errormsg = 'Cannot remove layers from the sim; please create the sim with the layers you want, e.g. cv.Sim(networks=...)'
            raise ValueError(errormsg)
        for lkey in args:
            self.pop(lkey)
        return

    def to_graph(self): # pragma: no cover
        """
        Convert all layers to a networkx MultiDiGraph

        **Example**::

            import networkx as nx
            sim = cv.Sim(pop_size=50, pop_type='hybrid').run()
            G = sim.people.contacts.to_graph()
            nx.draw(G)
        """
        import networkx as nx
        H = nx.MultiDiGraph()
        for lkey,layer in self.items():
            G = layer.to_graph()
            H = nx.compose(H, nx.MultiDiGraph(G))
        return H
