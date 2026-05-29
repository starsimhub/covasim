"""Contact-structure metrics for the M1 acceptance gate.

These summarize a built contact network so a v4 run can be compared to v3.1.8:
  - degree_by_layer(networks, pop_size) -> {layer: per-agent degree array}
  - age_mixing_matrix(network, ages, bin_edges) -> source-age x target-age count matrix
  - cosine_similarity(a, b) -> float in [-1, 1] (used to compare age-mixing matrices)
"""
import numpy as np


def _edges(net):
    """Return (p1, p2) integer arrays for a cv.Network."""
    return np.asarray(net.edges['p1']), np.asarray(net.edges['p2'])


def degree_by_layer(networks, pop_size):
    """Per-agent contact degree for each layer.

    Args:
        networks: an iterable (or ndict) of cv.Network instances.
        pop_size (int): number of agents (degree array length).

    Returns:
        dict of {layer_label: np.ndarray of per-agent degree (length pop_size)}.
    """
    nets = networks.values() if hasattr(networks, 'values') else networks
    out = {}
    for net in nets:
        p1, p2 = _edges(net)
        deg = np.zeros(pop_size, dtype=int)
        np.add.at(deg, p1, 1)  # each edge endpoint contributes one contact to each member
        np.add.at(deg, p2, 1)
        out[net.label] = deg
    return out


def age_mixing_matrix(network, ages, bin_edges):
    """Source-age x target-age contact-count matrix for one layer.

    Args:
        network: a cv.Network instance.
        ages (array): per-agent ages (indexed by UID).
        bin_edges (array): age-bin edges (e.g. np.arange(0, 105, 5)).

    Returns:
        np.ndarray of shape (n_bins, n_bins); both edge directions are counted.
    """
    ages = np.asarray(ages)
    p1, p2 = _edges(network)
    nb = len(bin_edges) - 1
    b1 = np.digitize(ages[p1], bin_edges) - 1
    b2 = np.digitize(ages[p2], bin_edges) - 1
    matrix = np.zeros((nb, nb))
    for a, b in ((b1, b2), (b2, b1)):  # undirected: count both directions
        valid = (a >= 0) & (a < nb) & (b >= 0) & (b < nb)
        np.add.at(matrix, (a[valid], b[valid]), 1)
    return matrix


def cosine_similarity(a, b):
    """Cosine similarity of two flattened arrays (1.0 = identical shape)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / denom) if denom else 0.0
