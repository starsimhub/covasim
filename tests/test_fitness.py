"""
Tests for covasim/fitness.py and extract_founding_mutations alignment.

Run from the tests/ directory:
    pytest test_fitness.py -v
"""

import csv
import io
import math
import os
import tempfile
import warnings

import numpy as np
import pytest

from covasim.fitness import BloomNtFitnessModel
from covasim.sequence_evolution import (
    encode_sequence,
    extract_founding_mutations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(rows):
    """Write a list of dicts to a temp CSV file; return the path."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    f.close()
    return f.name


def _write_fasta(seq, header='seq'):
    """Write a sequence to a temp FASTA file; return the path."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
    f.write(f'>{header}\n{seq}\n')
    f.close()
    return f.name


REF = 'ACGTACGTACGT'  # 12 bp toy reference
REF_ARR = encode_sequence(REF)


# ---------------------------------------------------------------------------
# BloomNtFitnessModel unit tests
# ---------------------------------------------------------------------------

class TestBloomNtFitnessModel:

    def _make_model(self, rows, scale=1.0):
        path = _write_csv(rows)
        model = BloomNtFitnessModel(path, scale=scale)
        os.unlink(path)
        return model

    def test_reference_haplotype_returns_one(self):
        rows = [{'nt_site': 1, 'nt': 'A', 'fitness': '0.0'},
                {'nt_site': 1, 'nt': 'C', 'fitness': '-1.5'}]
        model = self._make_model(rows)
        assert model.compute_fitness(frozenset()) == pytest.approx(1.0)

    def test_beneficial_mutation_above_one(self):
        rows = [{'nt_site': 1, 'nt': 'C', 'fitness': '2.0'}]
        model = self._make_model(rows, scale=1.0)
        # site 0-indexed → 1-indexed = 1; A(0)→C(1)
        muts = frozenset([(0, 0, 1)])
        assert model.compute_fitness(muts) == pytest.approx(math.exp(2.0))

    def test_deleterious_mutation_below_one(self):
        rows = [{'nt_site': 1, 'nt': 'C', 'fitness': '-1.0'}]
        model = self._make_model(rows, scale=1.0)
        muts = frozenset([(0, 0, 1)])
        assert model.compute_fitness(muts) == pytest.approx(math.exp(-1.0))

    def test_unknown_mutation_neutral(self):
        rows = [{'nt_site': 1, 'nt': 'C', 'fitness': '-1.0'}]
        model = self._make_model(rows, scale=1.0)
        # Site 5 not in data → treated as 0.0
        muts = frozenset([(4, 0, 2)])
        assert model.compute_fitness(muts) == pytest.approx(1.0)

    def test_additive_log_fitness(self):
        rows = [{'nt_site': 1, 'nt': 'C', 'fitness': '1.0'},
                {'nt_site': 2, 'nt': 'G', 'fitness': '0.5'}]
        model = self._make_model(rows, scale=1.0)
        muts = frozenset([(0, 0, 1), (1, 1, 2)])  # A→C @ 0, C→G @ 1
        assert model.compute_fitness(muts) == pytest.approx(math.exp(1.5))

    def test_scale_parameter(self):
        rows = [{'nt_site': 1, 'nt': 'C', 'fitness': '1.0'}]
        model = self._make_model(rows, scale=0.1)
        muts = frozenset([(0, 0, 1)])
        assert model.compute_fitness(muts) == pytest.approx(math.exp(0.1))

    def test_caching(self):
        rows = [{'nt_site': 1, 'nt': 'C', 'fitness': '1.0'}]
        model = self._make_model(rows)
        muts = frozenset([(0, 0, 1)])
        result1 = model.compute_fitness(muts)
        result2 = model.compute_fitness(muts)
        assert result1 is result2  # same object from cache


# ---------------------------------------------------------------------------
# extract_founding_mutations — same-length (fast path)
# ---------------------------------------------------------------------------

class TestExtractFoundingMutations:

    def test_identical_sequence_no_mutations(self):
        path = _write_fasta(REF)
        muts = extract_founding_mutations(path, REF_ARR)
        os.unlink(path)
        assert muts == frozenset()

    def test_single_snp_detected(self):
        seq = 'TCGTACGTACGT'  # A→T at position 0
        path = _write_fasta(seq)
        muts = extract_founding_mutations(path, REF_ARR)
        os.unlink(path)
        assert (0, 0, 3) in muts  # site=0, ref=A(0), alt=T(3)
        assert len(muts) == 1

    def test_n_treated_as_reference(self):
        seq = 'NCGTACGTACGT'  # N at position 0 → treated as ref A
        path = _write_fasta(seq)
        muts = extract_founding_mutations(path, REF_ARR)
        os.unlink(path)
        assert all(site != 0 for site, _, _ in muts)

    def test_multiline_fasta(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        f.write('>seq\n')
        f.write(REF[:6] + '\n')
        f.write(REF[6:] + '\n')
        f.close()
        muts = extract_founding_mutations(f.name, REF_ARR)
        os.unlink(f.name)
        assert muts == frozenset()


# ---------------------------------------------------------------------------
# extract_founding_mutations — length mismatch (alignment path)
# ---------------------------------------------------------------------------

class TestExtractFoundingMutationsAlignment:

    def test_no_snps_on_identical_prefix(self):
        # Sequence is reference minus last two bases — all positions that align
        # cleanly should show no SNPs
        path = _write_fasta(REF[:-2])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            muts = extract_founding_mutations(path, REF_ARR)
        os.unlink(path)
        # No substitutions — only deletions at the end, which are not SNPs
        assert len(muts) == 0

    def test_snp_in_shorter_sequence(self):
        # Reference: ACGTACGTACGT (12 bp)
        # Query:     TCGTACGTACT  (11 bp — first base mutated, last 'G' deleted)
        seq = 'TCGTACGTACT'
        path = _write_fasta(seq)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            muts = extract_founding_mutations(path, REF_ARR)
        os.unlink(path)
        # Should detect A→T substitution at position 0
        snp_sites = {site for site, _, _ in muts}
        assert 0 in snp_sites

    def test_longer_sequence_no_error(self):
        # Sequence longer than reference (insertion) — should not raise
        seq = REF + 'A'
        path = _write_fasta(seq)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            muts = extract_founding_mutations(path, REF_ARR)
        os.unlink(path)
        assert isinstance(muts, frozenset)

    def test_ambiguous_bases_treated_as_reference(self):
        # R, Y, etc. are IUPAC ambiguous — should not register as mutations
        seq = REF[:-2]  # shorter, triggers alignment path
        seq = 'R' + seq[1:]  # replace first char with ambiguous base
        path = _write_fasta(seq)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            muts = extract_founding_mutations(path, REF_ARR)
        os.unlink(path)
        snp_sites = {site for site, _, _ in muts}
        assert 0 not in snp_sites
