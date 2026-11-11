"""
Tests for covasim/clinical.py

Run from the tests/ directory:
    pytest test_clinical.py -v
"""

import pytest
import covasim as cv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_clinical_sim(cs, n_days=30, pop_size=500, pop_infected=50, rand_seed=42):
    """Small sim with sequence tracking and a ClinicalSequencer analyzer."""
    sim = cv.Sim(
        pop_size=pop_size,
        pop_infected=pop_infected,
        n_days=n_days,
        rand_seed=rand_seed,
        verbose=0,
        evo_pars=dict(
            enable=True,
            L=100,
            reference=None,
            mol_clock_rate=1e-6,
            fitness_model=None,
        ),
        analyzers=[cs],
    )
    sim.run()
    return sim


def all_released_agent_ids(cs):
    """Set of agent IDs appearing in any released batch."""
    ids = set()
    for batch in cs.samples.values():
        for s in batch:
            ids.add(s.agent_id)
    return ids


# ---------------------------------------------------------------------------
# Immediate release (batch_size=1)
# ---------------------------------------------------------------------------

def test_batch_size_one_releases_on_collection_days():
    """batch_size=1 should release specimens on the same day they are collected."""
    sample_days = [5, 10, 15, 20]
    cs = cv.ClinicalSequencer(
        days=sample_days,
        n_samples=10,
        pool='infectious',
        sequencing_batch_size=1,
    )
    sim = make_clinical_sim(cs)
    cs = sim.get_analyzer('ClinicalSequencer')

    for day in sample_days:
        assert day in cs.samples
        assert len(cs.samples[day]) > 0
        assert cs.n_pending == 0


def test_batch_size_one_matches_collected_count():
    """Each release day should contain exactly the specimens collected that day."""
    cs = cv.ClinicalSequencer(
        days=[10, 20],
        n_samples=5,
        pool='infectious',
        sequencing_batch_size=1,
    )
    sim = make_clinical_sim(cs, n_days=25)
    cs = sim.get_analyzer('ClinicalSequencer')

    for day, batch in cs.samples.items():
        assert len(batch) <= 5
        for s in batch:
            assert s.day == day


# ---------------------------------------------------------------------------
# Batched release
# ---------------------------------------------------------------------------

def test_batch_threshold_delays_release():
    """No sequences released until pending queue reaches batch_size."""
    cs = cv.ClinicalSequencer(
        days=[5, 10, 15],
        n_samples=3,
        pool='infectious',
        sequencing_batch_size=10,
    )
    sim = make_clinical_sim(cs, n_days=20)
    cs = sim.get_analyzer('ClinicalSequencer')

    # 3 specimens/day × 3 days = 9 — below threshold
    assert cs.samples == {}
    assert cs.n_pending == 9


def test_batch_releases_exactly_batch_size():
    """First release should contain exactly sequencing_batch_size specimens."""
    cs = cv.ClinicalSequencer(
        days=[5, 10, 15, 20],
        n_samples=3,
        pool='infectious',
        sequencing_batch_size=10,
    )
    sim = make_clinical_sim(cs, n_days=25)
    cs = sim.get_analyzer('ClinicalSequencer')

    assert len(cs.samples) >= 1
    first_release_day = min(cs.samples)
    assert len(cs.samples[first_release_day]) == 10


def test_batch_fifo_order():
    """Released specimens should be the oldest collected (FIFO)."""
    cs = cv.ClinicalSequencer(
        days=[5, 10, 15, 20],
        n_samples=3,
        pool='infectious',
        sequencing_batch_size=10,
    )
    sim = make_clinical_sim(cs, n_days=25)
    cs = sim.get_analyzer('ClinicalSequencer')

    first_release_day = min(cs.samples)
    batch = cs.samples[first_release_day]
    collection_days = [s.day for s in batch]
    # First 10 specimens: 3 from day 5, 3 from day 10, 3 from day 15, 1 from day 20
    assert collection_days.count(5) == 3
    assert collection_days.count(10) == 3
    assert collection_days.count(15) == 3
    assert collection_days.count(20) == 1


def test_pending_remainder_discarded():
    """Specimens below batch_size at end of sim stay pending and are never released."""
    cs = cv.ClinicalSequencer(
        days=[10],
        n_samples=7,
        pool='infectious',
        sequencing_batch_size=10,
    )
    sim = make_clinical_sim(cs, n_days=15)
    cs = sim.get_analyzer('ClinicalSequencer')

    assert cs.samples == {}
    assert cs.n_pending == 7


def test_pending_agents_not_in_samples():
    """Agent IDs in the pending queue must not appear in released samples."""
    cs = cv.ClinicalSequencer(
        days=[10],
        n_samples=7,
        pool='infectious',
        sequencing_batch_size=10,
    )
    sim = make_clinical_sim(cs, n_days=15)
    cs = sim.get_analyzer('ClinicalSequencer')

    pending_ids = {s.agent_id for s in cs.pending}
    released_ids = all_released_agent_ids(cs)
    assert pending_ids.isdisjoint(released_ids)


# ---------------------------------------------------------------------------
# to_fasta
# ---------------------------------------------------------------------------

def test_to_fasta_empty_before_release():
    """to_fasta returns empty string for days with no released batch."""
    cs = cv.ClinicalSequencer(
        days=[5, 10],
        n_samples=3,
        pool='infectious',
        sequencing_batch_size=10,
    )
    sim = make_clinical_sim(cs, n_days=15)
    cs = sim.get_analyzer('ClinicalSequencer')

    assert cs.to_fasta(5) == ''
    assert cs.to_fasta(10) == ''


def test_to_fasta_record_count_matches_batch():
    """to_fasta should emit one record per released specimen."""
    cs = cv.ClinicalSequencer(
        days=[5, 10, 15, 20],
        n_samples=3,
        pool='infectious',
        sequencing_batch_size=10,
    )
    sim = make_clinical_sim(cs, n_days=25)
    cs = sim.get_analyzer('ClinicalSequencer')

    release_day = min(cs.samples)
    fasta = cs.to_fasta(release_day)
    n_records = sum(1 for line in fasta.split('\n') if line.startswith('>'))
    assert n_records == 10


def test_to_fasta_header_uses_collection_day():
    """FASTA headers should reflect each specimen's collection day, not release day."""
    cs = cv.ClinicalSequencer(
        days=[5, 10, 15, 20],
        n_samples=3,
        pool='infectious',
        sequencing_batch_size=10,
    )
    sim = make_clinical_sim(cs, n_days=25)
    cs = sim.get_analyzer('ClinicalSequencer')

    release_day = min(cs.samples)
    batch = cs.samples[release_day]
    fasta = cs.to_fasta(release_day)
    headers = [line for line in fasta.split('\n') if line.startswith('>')]
    for header, sample in zip(headers, batch):
        assert f'day={sample.day}' in header
        if sample.day != release_day:
            assert f'day={release_day}' not in header


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_batch_size_raises():
    with pytest.raises(ValueError, match='sequencing_batch_size'):
        cv.ClinicalSequencer(days=[10], n_samples=5, sequencing_batch_size=0)
