'''
The v3 contact layers (``cv.Layer``, now a Starsim network, and ``cv.Contacts``), and the v3
``cv.Result`` (now Starsim's).
'''

import sciris as sc
import starsim as ss

# Specify all externally visible classes this file defines
__all__ = ['Result', 'FlexDict', 'Contacts', 'Layer']

# Results are now Starsim results
Result = ss.Result


class FlexDict(dict):
    '''
    A dict that allows more flexible element access: in addition to obj['a'],
    also allow obj[0]. Lightweight implementation of the Sciris odict class.
    '''

    def __getitem__(self, key):
        ''' Lightweight odict -- allow indexing by number, with low performance '''
        try:
            return super().__getitem__(key)
        except KeyError as KE:
            try: # Assume it's an integer
                dictkey = self.keys()[key]
                return self[dictkey]
            except:
                raise sc.KeyNotFoundError(KE) # Raise the original error

    def keys(self):
        return list(super().keys())

    def values(self):
        return list(super().values())

    def items(self):
        return list(super().items())


class Layer(ss.Network):
    '''
    A v3 contact layer: a static Starsim network, i.e. one whose edges don't change unless modified,
    e.g. by an intervention. See ``ss.Network`` for details.

    **Example**::

        n = 10_000
        n_people = 1000
        p1 = np.random.randint(n_people, size=n)
        p2 = np.random.randint(n_people, size=n)
        beta = np.ones(n)
        layer = cv.Layer(p1=p1, p2=p2, beta=beta, label='rand')
    '''
    def step(self):
        ''' The edges are static '''
        pass


class Contacts(FlexDict):
    '''
    The contact layers (i.e. networks), keyed by name, e.g. ``sim.people.contacts['h']``.

    ``sim.people.contacts`` is a view of the sim's networks, so adding a layer adds a network to the
    sim (after it has been initialized, but before it has been run).

    Args:
        data (dict): a dictionary of layers (or of dicts of p1, p2, and beta arrays)
        layer_keys (list): if provided, create an empty Contacts object with these layers
        sim (Sim): if provided, the sim whose networks these are
        kwargs (dict): additional layer(s), merged with data

    New in version 3.1.2: swapped order of arguments
    '''
    def __init__(self, data=None, layer_keys=None, sim=None, **kwargs):
        self.sim = None # Set at the end, so the layers aren't added to the sim again
        data = sc.mergedicts(data, kwargs)
        for lkey in sc.tolist(layer_keys):
            self[lkey] = Layer(name=lkey, label=lkey)
        for lkey,layer in data.items():
            self[lkey] = layer if isinstance(layer, ss.Network) else Layer(edges=layer, name=lkey, label=lkey)
        self.sim = sim
        return

    @classmethod
    def from_sim(cls, sim):
        ''' Create the contacts from the sim's networks '''
        return cls(data=dict(sim.networks.items()), sim=sim)

    def __repr__(self):
        ''' Use slightly customized repr'''
        keys_str = ', '.join([str(k) for k in self.keys()])
        output = f'Contacts({keys_str})\n'
        for key in self.keys():
            output += f'\n"{key}": '
            output += self[key].__repr__() + '\n'
        return output

    def __len__(self):
        ''' The length of the contacts is the length of all the layers '''
        return sum([len(layer) for layer in self.values()])

    def add_layer(self, **kwargs):
        '''
        Small method to add one or more layers to the contacts. Layers should
        be provided as keyword arguments.

        **Example**::

            hospitals_layer = cv.Layer(label='hosp')
            sim.people.contacts.add_layer(hospitals=hospitals_layer)
        '''
        for lkey,layer in kwargs.items():
            if not isinstance(layer, ss.Network):
                layer = Layer(edges=layer, label=lkey)
            layer.name = lkey
            layer.validate()
            self[lkey] = layer
            if self.sim is not None:
                self.sim._add_network(layer)
        return

    def pop_layer(self, *args):
        '''
        Remove the layer(s) from the contacts (but not from the sim).

        **Example**::

            sim.people.contacts.pop_layer('hospitals')
        '''
        for lkey in args:
            self.pop(lkey)
        return

    def to_graph(self): # pragma: no cover
        '''
        Convert all layers to a networkx MultiDiGraph

        **Example**::

            import networkx as nx
            sim = cv.Sim(pop_size=50, pop_type='hybrid').run()
            G = sim.people.contacts.to_graph()
            nx.draw(G)
        '''
        import networkx as nx
        H = nx.MultiDiGraph()
        for lkey,layer in self.items():
            G = layer.to_graph()
            H = nx.compose(H, nx.MultiDiGraph(G))
        return H
