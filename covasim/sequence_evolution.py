"""
Sequence evolution for Covasim.

When enabled (``evo_pars['enable'] = True``), each infection is annotated with the
nucleotide mutations that arose along its transmission edge, using a Jukes-Cantor (or
other) substitution model. The tracker is created by ``cv.COVID`` and called for every
new infection (in ``COVID.set_prognoses()``), and the annotations are stored in the
infection log (``sim.people.infection_log``: ``branch_length``, ``branch_mutations``, and
``n_mutations``).

Sequences are encoded as NumPy uint8 arrays: A=0, C=1, G=2, T=3. Only the mutations
along each branch are stored in the log; the haplotype of each agent's current or last
infection is kept by the tracker.
"""

import os
import numpy as np
from abc import ABC, abstractmethod

__all__ = ['SubstitutionModel', 'JukesCantor', 'LineageSequenceTracker',
           'encode_sequence', 'decode_sequence', 'read_fasta', 'extract_founding_mutations',
           'agent_mutations', 'decode_mutations']

nt_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
             'a': 0, 'c': 1, 'g': 2, 't': 3}
int_to_nt = ['A', 'C', 'G', 'T']


def encode_sequence(seq_str):
    """Convert a nucleotide string (ACGT) to a uint8 array."""
    return np.array([nt_to_int[c] for c in seq_str], dtype=np.uint8)


def decode_sequence(seq_arr):
    """Convert a uint8 array back to a nucleotide string."""
    return ''.join(int_to_nt[int(b)] for b in seq_arr)


def read_fasta(fasta_path):
    """Read the sequence of a single-record FASTA file as a string."""
    seq_lines = []
    with open(fasta_path) as fh:
        for line in fh:
            if not line.startswith('>'):
                seq_lines.append(line.strip())
    return ''.join(seq_lines)


def align_to_reference(seq_str, ref_str):
    """
    Global pairwise alignment of seq_str to ref_str using biopython.

    Returns (aligned_ref, aligned_query) strings with '-' for gaps.
    Only called when lengths differ.
    """
    try:
        from Bio.Align import PairwiseAligner
    except ImportError as E:
        errormsg = 'biopython is required to align sequences of different lengths: pip install biopython'
        raise ImportError(errormsg) from E
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score   = -10
    aligner.extend_gap_score = -0.5
    aligner.match_score      = 2
    aligner.mismatch_score   = -1
    alignment = next(aligner.align(ref_str, seq_str))
    ref_coords, query_coords = alignment.aligned
    aligned_ref, aligned_query = [], []
    r_prev, q_prev = 0, 0
    for (r_start, r_end), (q_start, q_end) in zip(ref_coords, query_coords):
        if r_start > r_prev:       # deletion in query
            aligned_ref.append(ref_str[r_prev:r_start])
            aligned_query.append('-' * (r_start - r_prev))
        if q_start > q_prev:       # insertion in query
            aligned_query.append(seq_str[q_prev:q_start])
            aligned_ref.append('-' * (q_start - q_prev))
        aligned_ref.append(ref_str[r_start:r_end])
        aligned_query.append(seq_str[q_start:q_end])
        r_prev, q_prev = r_end, q_end
    if r_prev < len(ref_str):
        aligned_ref.append(ref_str[r_prev:])
        aligned_query.append('-' * (len(ref_str) - r_prev))
    if q_prev < len(seq_str):
        aligned_query.append(seq_str[q_prev:])
        aligned_ref.append('-' * (len(seq_str) - q_prev))
    return ''.join(aligned_ref), ''.join(aligned_query)


def extract_founding_mutations(fasta_path, reference):
    """
    Diff a single-record FASTA against the reference and return a frozenset of SNPs.

    When the FASTA length matches the reference, a fast character-by-character diff
    is used. When lengths differ (common for consensus genomes with indels), the
    sequence is globally aligned to the reference with biopython; indel columns are
    skipped since the downstream fitness model only handles SNPs.

    Args:
        fasta_path (str): path to FASTA file (single record)
        reference  (ndarray): uint8 reference sequence from LineageSequenceTracker

    Returns:
        frozenset of (site_0indexed, ref_nt_int, alt_nt_int)
    """
    seq_str = read_fasta(fasta_path)

    if len(seq_str) != len(reference):
        ref_str = decode_sequence(reference)
        aligned_ref, aligned_query = align_to_reference(seq_str, ref_str)
        mutations = set()
        ref_site = 0
        for r_char, q_char in zip(aligned_ref, aligned_query):
            if r_char == '-':           # insertion in query → no reference coordinate
                continue
            if q_char == '-':           # deletion in query → not an SNP
                ref_site += 1
                continue
            if q_char not in nt_to_int:    # ambiguous base → treat as reference
                ref_site += 1
                continue
            ref_nt = nt_to_int[r_char]
            alt_nt = nt_to_int[q_char]
            if alt_nt != ref_nt:
                mutations.add((ref_site, ref_nt, alt_nt))
            ref_site += 1
        return frozenset(mutations)

    # Fast path: lengths match — character-by-character diff
    # Replace N/n (ambiguous bases) with the reference nucleotide so they don't
    # register as mutations — consensus FASTAs commonly contain Ns at low-coverage sites.
    seq_str = ''.join(
        int_to_nt[int(reference[i])] if c in ('N', 'n') else c
        for i, c in enumerate(seq_str)
    )
    seq_arr = encode_sequence(seq_str)
    return frozenset(
        (i, int(reference[i]), int(seq_arr[i]))
        for i in range(len(reference))
        if seq_arr[i] != reference[i]
    )


def apply_branch_mutations(parent_muts, branch_mutations, reference):
    """
    Compute a child's mutation set by applying branch mutations to a parent's set.

    parent_muts and the returned frozenset are relative to the reference:
    each element is (site_0indexed, ref_nt_int, child_nt_int).
    Reversions back to the reference allele are removed.

    Args:
        parent_muts      (frozenset): parent's mutations vs reference
        branch_mutations (list):      [(site, from_parent_nt, to_child_nt), ...]
        reference        (ndarray):   uint8 reference sequence

    Returns:
        frozenset of (site_0indexed, ref_nt_int, child_nt_int)
    """
    if not branch_mutations: # Nothing changed along the branch
        return parent_muts
    current = {site: to_nt for site, _, to_nt in parent_muts}
    for site, _from, to_child in branch_mutations:
        ref_nt = int(reference[site])
        if to_child == ref_nt:
            current.pop(site, None)   # reversion to reference
        else:
            current[site] = to_child  # new or updated mutation
    return frozenset((site, int(reference[site]), to_nt) for site, to_nt in current.items())


def agent_mutations(tracker, uid, reference=None):
    """
    Return the mutation frozenset (relative to the reference) of the current or last infection of agent ``uid``;
    empty if the agent has never been infected. Shared by ClinicalSequencer and WastewaterSampler.
    """
    return tracker.agent_mutations.get(int(uid), frozenset())


def decode_mutations(mutations, reference):
    """
    Reconstruct an ACGT string from a mutation frozenset and reference uint8 array.

    Shared by ClinicalSequencer and WastewaterSampler.
    """
    seq = reference.copy()
    for site, _ref_nt, alt_nt in mutations:
        seq[site] = alt_nt
    return decode_sequence(seq)


class SubstitutionModel(ABC):
    """Abstract base for nucleotide substitution models."""

    @property
    @abstractmethod
    def name(self):  # pragma: no cover
        ...

    @abstractmethod
    def evolve_branch(self, rng, parent_seq, delta_t, rate_per_site):
        """
        Apply substitutions along a branch of length delta_t.

        Args:
            rng            (Generator): numpy random generator
            parent_seq     (ndarray):   uint8 array of length L (parent haplotype)
            delta_t        (float):     branch length in days
            rate_per_site  (float):     expected substitutions per site per day (μ)

        Returns:
            mutations (list of tuple): (site, from_nt_int, to_nt_int) for each event
            child_seq (ndarray):       uint8 array, haplotype after mutations
        """
        ...  # pragma: no cover


class JukesCantor(SubstitutionModel):
    """
    Jukes-Cantor (JC69) substitution model.

    Total mutations over L sites and branch length Δt are drawn from
    Poisson(L * μ * Δt), then allocated to uniformly random sites.
    At each site a uniform random one of the three alternative nucleotides
    replaces the current base.
    """

    name = 'JC'

    def evolve_branch(self, rng, parent_seq, delta_t, rate_per_site):
        L = len(parent_seq)
        n_events = int(rng.poisson(L * rate_per_site * delta_t)) if delta_t > 0 else 0
        if n_events == 0:
            return [], parent_seq

        child_seq = parent_seq.copy()
        sites   = rng.integers(0, L, size=n_events)
        offsets = rng.integers(1, 4, size=n_events)  # 1, 2, or 3 → guarantees different nucleotide

        mutations = []
        for site, offset in zip(sites, offsets):
            from_nt = int(child_seq[site])
            to_nt   = int((from_nt + int(offset)) % 4)
            child_seq[site] = to_nt
            mutations.append((int(site), from_nt, to_nt))

        return mutations, child_seq


substitution_models = {'JC': JukesCantor} # The substitution models, by their names in evo_pars['sub_model']


class LineageSequenceTracker:
    """
    Tracks viral sequence evolution along the transmission tree.

    Keeps the haplotype of each agent's current or last infection (replaced on each new
    infection), and works out the mutations along each transmission edge. ``cv.COVID``
    creates it when ``evo_pars['enable']`` is True (``sim.sequence_tracker``), and calls
    ``annotate()`` for each new set of infections.

    Fitness:
        If ``fitness_model`` is set, each new infection gets a transmissibility multiplier
        from its haplotype (``fitness()``), which ``cv.COVID`` stores in ``rel_fitness``.
        Set ``variant_founding_mutations[label] = frozenset(...)`` for any variant
        whose founding genotype differs from the reference (done by ``cv.COVID`` for
        variants with a ``founding_fasta``).

    Args:
        evo_pars (dict): the sequence evolution parameters (see ``cv.make_pars()``)
        seed (int): the random seed for the mutations
    """

    def __init__(self, evo_pars, seed=None):
        self.L    = evo_pars.get('L', 1000)
        self.rate = evo_pars.get('mol_clock_rate', 1e-5)

        ref = evo_pars.get('reference')
        if ref is None:
            ref = 'A' * self.L
        elif isinstance(ref, str):
            looks_like_path = ref.endswith(('.fasta', '.fa', '.fas', '.fna')) or os.sep in ref or '/' in ref
            if looks_like_path and not os.path.isfile(ref):
                errormsg = (f"evo_pars['reference'] path not found: '{ref}'. Current working directory is '{os.getcwd()}'. "
                            "Use an absolute path or ensure the file exists relative to the current folder.")
                raise FileNotFoundError(errormsg)
            if os.path.isfile(ref): # A FASTA file: read the sequence, which sets L
                ref = read_fasta(ref)
                self.L = len(ref)
        if isinstance(ref, str):
            if len(ref) != self.L:
                raise ValueError(f"evo_pars['reference'] has length {len(ref)} but L={self.L}")
            ref = encode_sequence(ref)
        self.reference = ref.astype(np.uint8)

        model_key = evo_pars.get('sub_model', 'JC')
        if model_key not in substitution_models:
            raise ValueError(f"Unknown substitution model '{model_key}'; choices: {list(substitution_models)}")
        self.model = substitution_models[model_key]()

        self.rng = np.random.default_rng(seed)
        self.episode_roots = {}  # uid → uint8 haplotype of the agent's current or last infection

        # Fitness and founding genotypes (set by cv.COVID)
        self.fitness_model = None                # VariantFitnessModel instance or None
        self.variant_founding_mutations = {}     # label (str) → frozenset{(site, ref_nt, alt_nt)}
        self.agent_mutations = {}                # uid → frozenset{(site, ref_nt, alt_nt)}, relative to the reference

    def annotate_entry(self, entry, date_exposed):
        """
        Annotate one infection_log entry with sequence evolution data.

        Mutates entry in place, adding:
          branch_length      (float): Δt in days
          branch_mutations   (list):  [(site, from_nt_int, to_nt_int), ...]
          n_mutations        (int):   len(branch_mutations)

        Also updates the haplotype (``episode_roots``) and the mutations relative to the
        reference (``agent_mutations``) of the target.

        Call this before the date of exposure of the target is set for this infection.

        Args:
            entry (dict): with keys ``source`` (None for a seed or import), ``target``, ``date`` (the day), and ``variant`` (the label)
            date_exposed (array): the day each agent was last exposed, indexed by UID (e.g. ``covid.ti_exposed.raw``)
        """
        source        = entry['source']
        target        = int(entry['target'])
        date          = float(entry['date'])
        variant_label = entry.get('variant', 'wild')

        if source is None:
            # Seed or import: branch length 0. The episode root is the variant's
            # founding haplotype (reference + its defining SNPs), not the bare
            # reference — otherwise every variant would collapse to the reference sequence.
            delta_t     = 0.0
            founding    = self.variant_founding_mutations.get(variant_label, frozenset())
            if founding:
                parent_seq = self.reference.copy()
                for site, _ref_nt, alt_nt in founding:
                    parent_seq[site] = alt_nt
            else:
                parent_seq = self.reference
            parent_muts = founding
        else:
            source      = int(source)
            parent_seq  = self.episode_roots.get(source, self.reference)
            src_exposed = float(date_exposed[source])
            delta_t     = 0.0 if np.isnan(src_exposed) else max(0.0, date - src_exposed)
            parent_muts = self.agent_mutations.get(source, frozenset())

        mutations, child_seq = self.model.evolve_branch(self.rng, parent_seq, delta_t, self.rate)
        self.episode_roots[target] = child_seq
        self.agent_mutations[target] = apply_branch_mutations(parent_muts, mutations, self.reference)

        entry['branch_length']    = delta_t
        entry['branch_mutations'] = mutations
        entry['n_mutations']      = len(mutations)
        return entry

    def annotate(self, uids, sources, variants, day, date_exposed):
        """
        Annotate a set of new infections, all on the same day (see ``annotate_entry()``).

        Args:
            uids (array): the agents infected
            sources (array): who infected each one (NaN or negative for seeds and imports)
            variants (list): the variant label of each infection
            day (int): the day of the infections
            date_exposed (array): the day each agent was last exposed, indexed by UID

        Returns:
            Arrays of the branch length, the number of mutations, and the mutations (an object array of lists) of each infection
        """
        n = len(uids)
        branch_length = np.zeros(n)
        n_mutations = np.zeros(n, dtype=int)
        branch_mutations = np.empty(n, dtype=object)
        for i in range(n):
            source = sources[i]
            source = None if (np.isnan(source) or source < 0) else int(source)
            entry = dict(source=source, target=uids[i], date=day, variant=variants[i])
            self.annotate_entry(entry, date_exposed)
            branch_length[i] = entry['branch_length']
            n_mutations[i] = entry['n_mutations']
            branch_mutations[i] = entry['branch_mutations']
        return branch_length, n_mutations, branch_mutations

    def initialize_founder_haplotype(self, uid, variant_label):
        """Set the mutations of a seeded or imported agent to the founding set of their variant (as annotate_entry() does)."""
        self.agent_mutations[int(uid)] = self.variant_founding_mutations.get(variant_label, frozenset())

    def get_fitness_multiplier(self, uid):
        """Return the rel_beta multiplier of agent uid; 1.0 if no fitness model is set."""
        if self.fitness_model is None:
            return 1.0
        muts = self.agent_mutations.get(int(uid), frozenset())
        return self.fitness_model.compute_fitness(muts)

    def fitness(self, uids):
        """The rel_beta multiplier of each of these agents (all 1.0 if no fitness model is set)."""
        return np.array([self.get_fitness_multiplier(uid) for uid in uids])

    def reconstruct_haplotype(self, uid):
        """
        Return the haplotype string of the current or last infection of agent uid.
        Returns the reference sequence string if the agent has never been infected.
        """
        return decode_sequence(self.episode_roots.get(int(uid), self.reference))
