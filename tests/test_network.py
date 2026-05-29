"""Contact-structure tests for cv.Network (the M1 acceptance gate, structural half).

Builds cv.Network instances (random single-layer 'a' and the four hybrid layers
h/s/w/c) on a properly-aged ss.People and asserts per-layer degree and age-window
structure. The v3.1.8 EQUIVALENCE half (vs a gitignored baseline) lives in
test_contact_structure_equivalence and skips cleanly when the baseline is absent.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import starsim as ss
import covasim as cv
import covasim.defaults as cvd

sys.path.insert(0, str(Path(__file__).parent))
from regression.contact_stats import degree_by_layer, age_mixing_matrix, cosine_similarity  # noqa: E402

POP = 5000  # small enough to be fast; large enough for stable degree stats


def _age_data():
    """Covasim's default age pyramid as the Nx2 [age_lower_edge, value] array ss.People wants."""
    d = np.asarray(cvd.default_age_data, dtype=float)
    return np.c_[d[:, 0], d[:, 2]]


def _inited_sim(pop_type='random', n=POP, seed=0):
    """Build and initialize (not run) an ss.Sim with cv.Networks over a Covasim-aged population."""
    people = ss.People(n, age_data=_age_data())
    nets = cv.make_networks(pop_type)
    sim = ss.Sim(people=people, networks=nets, diseases=[],
                 start=ss.date('2020-03-01'), dur=ss.days(1), dt=ss.days(1),
                 rand_seed=seed, verbose=0)
    sim.init()
    return sim


def _mean_degree(deg, participants_only=False):
    """Mean degree; over participants (degree>0) if requested, else over all agents."""
    deg = np.asarray(deg)
    if participants_only:
        deg = deg[deg > 0]
    return float(deg.mean()) if len(deg) else 0.0


# --- structural tests (no v3 baseline needed) --------------------------------

def test_random_single_layer_mean_degree():
    sim = _inited_sim('random')
    deg = degree_by_layer(sim.networks, POP)
    assert set(deg) == {'a'}, f'expected one layer "a", got {set(deg)}'
    md = _mean_degree(deg['a'])
    assert 18.0 <= md <= 22.0, f'random layer mean degree should be ~20, got {md:.2f}'


def test_hybrid_layer_keys_and_degrees():
    sim = _inited_sim('hybrid')
    deg = degree_by_layer(sim.networks, POP)
    assert set(deg) == {'h', 's', 'w', 'c'}, f'expected h/s/w/c, got {set(deg)}'
    # Community: random over all -> mean ~20 over everyone.
    assert 18.0 <= _mean_degree(deg['c']) <= 22.0, f"community mean degree ~20, got {_mean_degree(deg['c']):.2f}"
    # School/work: random over the age subset -> mean ~20 / ~16 among participants.
    assert 17.0 <= _mean_degree(deg['s'], participants_only=True) <= 23.0, \
        f"school mean degree ~20 among participants, got {_mean_degree(deg['s'], True):.2f}"
    assert 13.5 <= _mean_degree(deg['w'], participants_only=True) <= 18.5, \
        f"work mean degree ~16 among participants, got {_mean_degree(deg['w'], True):.2f}"
    # Household: Poisson(2.0) fully-connected clusters -> realized mean degree ~2 (NOT == contacts['h']).
    assert 1.0 <= _mean_degree(deg['h'], participants_only=True) <= 4.0, \
        f"household realized mean degree ~2, got {_mean_degree(deg['h'], True):.2f}"


def test_school_work_age_windows():
    sim = _inited_sim('hybrid')
    ages = np.asarray(sim.people.age)
    for layer, (lo, hi) in [('s', (6, 22)), ('w', (22, 65))]:
        net = sim.networks[layer]
        members = np.unique(np.concatenate([np.asarray(net.edges['p1']), np.asarray(net.edges['p2'])]))
        member_ages = ages[members]
        assert (member_ages >= lo).all() and (member_ages < hi).all(), \
            f'layer {layer!r} has members outside age window [{lo},{hi}): ' \
            f'ages [{member_ages.min():.1f}, {member_ages.max():.1f}]'


def test_deterministic_same_seed():
    """Same rand_seed -> identical network edges (v4 reproducibility)."""
    a = _inited_sim('hybrid', seed=3)
    b = _inited_sim('hybrid', seed=3)
    for layer in ['h', 's', 'w', 'c']:
        assert np.array_equal(a.networks[layer].edges['p1'], b.networks[layer].edges['p1']), f'{layer} p1 not reproducible'
        assert np.array_equal(a.networks[layer].edges['p2'], b.networks[layer].edges['p2']), f'{layer} p2 not reproducible'


def test_age_mixing_matrix_symmetry():
    sim = _inited_sim('hybrid')
    ages = np.asarray(sim.people.age)
    edges = np.arange(0, 105, 5)
    M = age_mixing_matrix(sim.networks['c'], ages, edges)
    assert M.shape == (len(edges) - 1, len(edges) - 1)
    assert M.sum() > 0
    # Both edge directions are counted, so the contact matrix must be symmetric.
    assert np.allclose(M, M.T), 'age-mixing matrix should be symmetric'


def test_cosine_similarity_helper():
    # cosine_similarity distinguishes identical / orthogonal inputs and is scale-invariant.
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


# --- v3.1.8 equivalence (skips when the gitignored baseline is absent) -------

_BASELINE = Path(__file__).parent / 'regression' / 'v3_m1_contacts.json'


@pytest.mark.skipif(not _BASELINE.exists(),
                    reason='Missing v3.1.8 contact baseline (regression/v3_m1_contacts.json); '
                           'generate from a frozen v3.1.8 env (M1 Task 4).')
def test_contact_structure_equivalence():
    import json
    base = json.loads(_BASELINE.read_text())
    # Compares per-layer mean degree (within tolerance) + age-mixing cosine similarity (> threshold).
    # Final tolerances are pinned here once the v3.1.8 baseline exists (M1 Task 4).
    raise NotImplementedError('Wired up in M1 Task 4 once the v3.1.8 contact baseline is generated.')
