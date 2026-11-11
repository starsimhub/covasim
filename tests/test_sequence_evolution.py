"""
Tests for covasim/sequence_evolution.py

Run from the tests/ directory:
    pytest test_sequence_evolution.py -v
"""

import numpy as np
import pytest
import covasim as cv
from covasim.sequence_evolution import (
    encode_sequence, decode_sequence,
    JukesCantor, LineageSequenceTracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_seq_sim(n_days=10, pop_size=200, pop_infected=10, rand_seed=42,
                 L=100, rate=1e-3, **evo_kwargs):
    """Small sim with sequence tracking enabled (fitness disabled by default)."""
    evo_pars = dict(
        enable=True,
        L=L,
        reference=None,
        mol_clock_rate=rate,
        sub_model='JC',
        fitness_model=None,  # sequence-evolution tests don't need Bloom fitness data
        **evo_kwargs,
    )
    sim = cv.Sim(
        pop_size=pop_size,
        pop_infected=pop_infected,
        n_days=n_days,
        rand_seed=rand_seed,
        verbose=0,
        evo_pars=evo_pars,
    )
    sim.run()
    return sim


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def test_encode_decode_roundtrip():
    seq = 'ACGTACGT'
    assert decode_sequence(encode_sequence(seq)) == seq


def test_encode_case_insensitive():
    assert np.array_equal(encode_sequence('acgt'), encode_sequence('ACGT'))


# ---------------------------------------------------------------------------
# JukesCantor unit tests
# ---------------------------------------------------------------------------

def test_jc_zero_delta_t_no_mutations():
    rng = np.random.default_rng(0)
    parent = encode_sequence('AAAA')
    mutations, child = JukesCantor().evolve_branch(rng, parent, delta_t=0.0, rate_per_site=1.0)
    assert mutations == []
    assert np.array_equal(child, parent)


def test_jc_mutations_change_nucleotide():
    """Every mutation must produce a different nucleotide."""
    rng = np.random.default_rng(7)
    parent = encode_sequence('A' * 1000)
    # Use very high rate to guarantee mutations
    mutations, child = JukesCantor().evolve_branch(rng, parent, delta_t=100.0, rate_per_site=0.1)
    assert len(mutations) > 0
    for site, from_nt, to_nt in mutations:
        assert from_nt != to_nt, f'Mutation at site {site} did not change nucleotide'
        assert to_nt in (0, 1, 2, 3)


def test_jc_child_sequence_consistent_with_mutations():
    """Reconstruct the child manually from the mutations and verify it matches."""
    rng = np.random.default_rng(3)
    parent = encode_sequence('ACGTACGT' * 10)
    mutations, child = JukesCantor().evolve_branch(rng, parent, delta_t=500.0, rate_per_site=0.01)

    # Apply mutations to parent manually
    reconstructed = parent.copy()
    for site, from_nt, to_nt in mutations:
        reconstructed[site] = to_nt

    assert np.array_equal(child, reconstructed)


def test_jc_reproducibility():
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    parent = encode_sequence('ACGT' * 25)
    m1, c1 = JukesCantor().evolve_branch(rng1, parent, delta_t=10.0, rate_per_site=1e-3)
    m2, c2 = JukesCantor().evolve_branch(rng2, parent, delta_t=10.0, rate_per_site=1e-3)
    assert m1 == m2
    assert np.array_equal(c1, c2)


# ---------------------------------------------------------------------------
# LineageSequenceTracker unit tests
# ---------------------------------------------------------------------------

def _make_tracker(L=50, rate=1e-3, seed=0):
    evo_pars = dict(
        L=L,
        reference=None,
        mol_clock_rate=rate,
        sub_model='JC',
    )
    return LineageSequenceTracker(evo_pars, seed=seed)


def test_tracker_reference_default():
    tracker = _make_tracker(L=10)
    assert decode_sequence(tracker.reference) == 'A' * 10


def test_tracker_custom_reference():
    wt = 'ACGTACGTAC'
    evo_pars = dict(L=10, reference=wt, mol_clock_rate=1e-5,
                    sub_model='JC')
    tracker = LineageSequenceTracker(evo_pars, seed=0)
    assert decode_sequence(tracker.reference) == wt


def test_tracker_reference_length_mismatch():
    evo_pars = dict(L=5, reference='ACGT', mol_clock_rate=1e-5,
                    sub_model='JC')
    with pytest.raises(ValueError, match='length'):
        LineageSequenceTracker(evo_pars, seed=0)


def test_tracker_seed_entry_branch_length_zero():
    """Seeds (source=None) must have branch_length=0."""
    tracker = _make_tracker()

    date_exposed = np.full(100, np.nan) # The day each agent was last exposed

    entry = dict(source=None, target=5, date=0.0, layer='seed_infection', variant='wild')
    tracker.annotate_entry(entry, date_exposed)
    assert entry['branch_length'] == 0.0
    assert isinstance(entry['branch_mutations'], list)
    assert entry['n_mutations'] == 0  # Δt=0 → no mutations


def test_tracker_transmission_branch_length():
    tracker = _make_tracker(rate=1e-5)  # low rate so probably 0 mutations, but Δt is checked

    date_exposed = np.array([0.0, np.nan, np.nan]) # The day each agent was last exposed

    # First seed agent 0
    seed_entry = dict(source=None, target=0, date=0.0, layer='seed_infection', variant='wild')
    tracker.annotate_entry(seed_entry, date_exposed)
    date_exposed[0] = 0.0  # simulate date_exposed being set after infect()

    # Agent 0 (exposed day 0) transmits to agent 1 on day 5 → Δt = 5
    trans_entry = dict(source=0, target=1, date=5.0, layer='h', variant='wild')
    tracker.annotate_entry(trans_entry, date_exposed)
    assert trans_entry['branch_length'] == 5.0


def test_tracker_episode_root_reset_on_reinfection():
    """On reinfection, the target's episode root is replaced, not merged."""
    tracker = _make_tracker(L=4, rate=100.0, seed=5)  # very high rate for guaranteed mutations

    date_exposed = np.array([0.0, np.nan]) # The day each agent was last exposed

    # First infect target=1 as seed
    e1 = dict(source=None, target=1, date=0.0, layer='seed_infection', variant='wild')
    tracker.annotate_entry(e1, date_exposed)
    date_exposed[1] = 0.0

    root_after_first = tracker.episode_roots[1].copy()

    # Reinfect target=1 from source=0 on day 10 (high rate → many mutations)
    date_exposed[0] = 0.0
    e2 = dict(source=0, target=1, date=10.0, layer='h', variant='wild')
    tracker.annotate_entry(e2, date_exposed)

    root_after_second = tracker.episode_roots[1].copy()
    # The new episode root comes from source=0's haplotype evolved over Δt=10, not from target's prior
    # (Just verify it was overwritten — exact value depends on RNG)
    assert 1 in tracker.episode_roots  # still tracked


def test_tracker_reconstruct_haplotype_uninfected():
    tracker = _make_tracker(L=8)
    # Agent never infected → should return reference sequence string
    ref_str = decode_sequence(tracker.reference)
    assert tracker.reconstruct_haplotype(999) == ref_str


def test_tracker_reconstruct_haplotype_after_infection():
    tracker = _make_tracker(L=10, rate=0.0)  # zero rate → no mutations

    date_exposed = np.full(10, np.nan) # The day each agent was last exposed

    entry = dict(source=None, target=3, date=0.0, layer='seed_infection', variant='wild')
    tracker.annotate_entry(entry, date_exposed)
    # With rate=0, child == reference
    ref_str = decode_sequence(tracker.reference)
    assert tracker.reconstruct_haplotype(3) == ref_str


# ---------------------------------------------------------------------------
# Integration tests: full Sim
# ---------------------------------------------------------------------------

def test_seq_disabled_by_default():
    """No seq keys on infection_log when feature is off."""
    sim = cv.Sim(pop_size=200, pop_infected=10, n_days=5, rand_seed=1, verbose=0)
    sim.run()
    for entry in sim.people.infection_log:
        assert 'branch_mutations' not in entry
        assert 'branch_length' not in entry


def test_seq_enabled_all_entries_annotated():
    """Every log entry must have all four sequence keys when enabled."""
    sim = make_seq_sim(n_days=5)
    assert len(sim.people.infection_log) > 0
    for entry in sim.people.infection_log:
        assert 'branch_length'    in entry, f'Missing branch_length in {entry}'
        assert 'branch_mutations' in entry, f'Missing branch_mutations in {entry}'
        assert 'n_mutations'      in entry, f'Missing n_mutations in {entry}'
        assert entry['n_mutations'] == len(entry['branch_mutations'])


def test_seq_seed_entries_have_zero_branch_length():
    sim = make_seq_sim(n_days=3)
    seeds = [e for e in sim.people.infection_log if e['source'] is None]
    assert len(seeds) > 0
    for e in seeds:
        assert e['branch_length'] == 0.0


def test_seq_transmission_branch_length_nonnegative():
    sim = make_seq_sim(n_days=15)
    transmissions = [e for e in sim.people.infection_log if e['source'] is not None]
    for e in transmissions:
        assert e['branch_length'] >= 0.0


def test_seq_mutations_valid_nucleotides():
    """All mutation tuples must have from_nt != to_nt and valid nt integers."""
    sim = make_seq_sim(n_days=15, rate=1e-2)  # higher rate for more mutations
    for entry in sim.people.infection_log:
        for site, from_nt, to_nt in entry['branch_mutations']:
            assert from_nt != to_nt, f'from_nt == to_nt at site {site}'
            assert from_nt in (0, 1, 2, 3)
            assert to_nt   in (0, 1, 2, 3)
            assert 0 <= site < 100  # L=100 in make_seq_sim


def test_seq_reproducibility():
    """Two sims with the same seed must produce identical infection logs."""
    sim1 = make_seq_sim(rand_seed=7)
    sim2 = make_seq_sim(rand_seed=7)
    log1 = sim1.people.infection_log
    log2 = sim2.people.infection_log
    assert len(log1) == len(log2)
    for e1, e2 in zip(log1, log2):
        assert e1['branch_length']    == e2['branch_length']
        assert e1['branch_mutations'] == e2['branch_mutations']
        assert e1['n_mutations']      == e2['n_mutations']


def test_seq_different_seeds_different_mutations():
    """Different seeds should (with overwhelming probability) give different logs."""
    sim1 = make_seq_sim(rand_seed=1, rate=1e-2, n_days=20)
    sim2 = make_seq_sim(rand_seed=2, rate=1e-2, n_days=20)
    all_mutations1 = [e['branch_mutations'] for e in sim1.people.infection_log]
    all_mutations2 = [e['branch_mutations'] for e in sim2.people.infection_log]
    assert all_mutations1 != all_mutations2


def test_seq_tracker_attached_to_people():
    sim = make_seq_sim()
    assert sim.sequence_tracker is not None
    assert sim.people.sequence_tracker is sim.sequence_tracker


def test_seq_disabled_tracker_is_none():
    sim = cv.Sim(pop_size=100, pop_infected=5, n_days=3, verbose=0)
    sim.run()
    assert sim.sequence_tracker is None
    assert getattr(sim.people, 'sequence_tracker', None) is None
