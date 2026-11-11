"""
Wastewater sampling analyzer for Covasim.

WastewaterSampler is a cv.Analyzer that, at specified time points, snapshots
the set of circulating viral haplotypes and their relative proportions weighted
by individual viral shedding — i.e. what a wastewater sample would look like.
Haplotypes are stored as frozensets of (site_0indexed, ref_nt_int, alt_nt_int)
mutation tuples relative to the simulation reference sequence.
Snapshots can be exported as FASTA or fed directly to Bygul's
simulate_proportions to generate synthetic sequencing reads.

Requires pars['evo_pars']['enable'] = True.
"""

import os
import csv
import tempfile
import dataclasses
from collections import defaultdict

import numpy as np
import sciris as sc

from . import interventions as cvi
from .analysis import Analyzer
from .sequence_evolution import agent_mutations, decode_mutations

__all__ = ['WastewaterSampler', 'WastewaterSample']


@dataclasses.dataclass
class WastewaterSample:
    """Snapshot of circulating haplotypes and their viral-shedding-weighted proportions."""
    day:              int
    date:             str
    mutation_sets:    list  # deduplicated frozensets of (site_0idx, ref_nt_int, alt_nt_int), one per distinct haplotype
    variant_labels:   list  # variant label string for each distinct haplotype
    proportions:      list  # normalized viral-shedding fractions, sums to 1.0
    raw_loads:        list  # un-normalized total viral shedding per haplotype
    n_infectious:     int   # number of currently infectious agents
    n_genotypes:      int   # number of distinct haplotypes present
    variant_shedding: dict = None  # {variant_label: total_shedding} aggregated by named variant


def variant_label(covid, variant):
    """ The label of a variant index (NaN, for an agent with no current infection, is 'wild', as in the fork's v3 version) """
    return 'wild' if np.isnan(variant) else covid.variant_map[int(variant)]


class WastewaterSampler(Analyzer):
    """
    Analyzer that snapshots the viral haplotype mixture present in a wastewater
    sample at one or more time points.

    For each sampled day:
      1. Identifies all currently infectious agents (in the sewershed, if ``sid`` is given).
      2. Reads each agent's viral shedding from ``sim.people.viral_shedding`` (see ``cv.COVID``).
      3. Retrieves each agent's mutation frozenset from the LineageSequenceTracker.
      4. Groups identical haplotypes and sums their viral shedding contributions.
      5. Normalizes to proportions.

    Snapshots are stored in self.samples (dict: int day → WastewaterSample, or None if nobody was shedding).
    Use to_fasta(day) to get a multi-FASTA string, or simulate_sample(day, ...)
    to pass the mixture directly to Bygul.

    Requires pars['evo_pars']['enable'] = True.

    Args:
        days  (list): simulation days (int) or calendar date strings to sample.
        label (str):  optional label for the analyzer.
        sid   (int):  optional sewershed ID; None samples everyone,
                      -1 samples people outside all supplied polygons. Use one analyzer per sid.
        kwargs (dict): passed to ``cv.Analyzer``

    **Example**::

        ww = cv.WastewaterSampler(days=[56, 120])
        sim = cv.Sim(pars, analyzers=[ww])
        sim.run()
        print(sim.get_analyzer('WastewaterSampler').to_fasta(56))
    """

    def __init__(self, days, label=None, sid=None, **kwargs):
        if label is None:
            label = 'WastewaterSampler'
        super().__init__(label=label, **kwargs)
        if sid is not None and (isinstance(sid, (bool, np.bool_)) or not isinstance(sid, (int, np.integer)) or sid < -1):
            raise ValueError('sid must be an integer >= -1, or None for the entire population')
        self.sid = sid
        self.days = days # Converted to day indices in init_post()
        self.samples = {}
        self.reference = None # Set in init_post()
        return

    def init_post(self):
        super().init_post()
        covid = self.sim.diseases.covid
        if covid.sequence_tracker is None:
            errormsg = "WastewaterSampler requires pars['evo_pars']['enable'] = True"
            raise ValueError(errormsg)
        if covid.shedding_lookup is None:
            errormsg = 'WastewaterSampler requires the viral shedding, but the shedding parameter is None'
            raise ValueError(errormsg)
        self.reference = covid.sequence_tracker.reference
        self.days = cvi.process_days(self.sim, sc.tolist(self.days))
        return

    def step(self):
        if self.today in self.days:
            self.take_sample()
        return

    def take_sample(self):
        """ Take the wastewater sample on this day """
        sim = self.sim
        day = self.today
        covid = sim.diseases.covid
        people = sim.people

        # Infectious agents only
        eligible = covid.infectious
        if self.sid is not None:
            eligible = eligible & (people.sewershed == self.sid)
        uids = eligible.uids

        if len(uids) == 0:
            self.samples[day] = None
            return

        loads    = covid.viral_shedding[uids]
        tracker  = covid.sequence_tracker
        variants = covid.infectious_variant[uids]

        load_by_muts  = defaultdict(float)
        label_by_muts = {}   # first-seen variant label per distinct haplotype
        load_by_variant = defaultdict(float)
        for uid, load, var in zip(uids, loads, variants):
            muts = agent_mutations(tracker, uid)
            label = variant_label(covid, var)
            load_by_muts[muts] += float(load)
            load_by_variant[label] += float(load)
            if muts not in label_by_muts:
                label_by_muts[muts] = label

        total = sum(load_by_muts.values())
        if total <= 0:
            self.samples[day] = None
            return
        mutation_sets = list(load_by_muts.keys())
        raw_loads     = [load_by_muts[m] for m in mutation_sets]
        proportions   = [v / total for v in raw_loads]
        proportions[-1] = 1.0 - sum(proportions[:-1])

        self.samples[day] = WastewaterSample(
            day              = day,
            date             = sim.date(day),
            mutation_sets    = mutation_sets,
            variant_labels   = [label_by_muts[m] for m in mutation_sets],
            proportions      = proportions,
            raw_loads        = raw_loads,
            n_infectious     = int(len(uids)),
            n_genotypes      = len(mutation_sets),
            variant_shedding = dict(load_by_variant),
        )
        return

    def to_fasta(self, day):
        """Return a multi-FASTA string for the haplotype mixture on the given day."""
        sample = self.samples[day]
        if sample is None:
            return ''
        ref   = self.reference
        lines = []
        for i, (muts, prop, vlabel) in enumerate(zip(sample.mutation_sets, sample.proportions, sample.variant_labels)):
            lines.append(f'>genotype_{i}  proportion={prop:.8f}  day={day}  date={sample.date}  variant={vlabel}')
            lines.append(decode_mutations(muts, ref))
        return '\n'.join(lines)

    def simulate_sample(self, day, primers, reference, outdir='./reads/', readcnt=500, redo=False, **bygul_kwargs):
        """
        Write a multi-FASTA file and a proportions CSV, then invoke Bygul's simulate_proportions CLI.

        Bygul's simulate_proportions command is a Click CLI that expects genome
        sequences inside a multi-FASTA file and their relative abundances in a CSV.
        This method writes temporary files, builds the required CLI argument list,
        and calls the command via Click's standalone_mode=False interface.

        Args:
            day       (int):  simulation day to sample (must be in self.samples)
            primers   (str):  path to primer BED file (required by Bygul)
            reference (str):  path to reference FASTA file (required by Bygul)
            outdir    (str):  output directory for Bygul reads (default './reads/') (NOTE: If redo=True, Bygul will delete all contents of outdir)
            readcnt   (int):  number of reads per amplicon (default 500)
            redo      (bool): re-run even if output files already exist (default False)
            bygul_kwargs (dict): additional CLI options forwarded as --key value pairs
                            (e.g. wgsim_read_length=150, wgsim_error_rate=0.0001,
                            simulation_mode='amplicon', seed=42)

        Requires the `bygul` package to be installed.
        """
        try:
            from bygul._cli import simulate_proportions
        except ImportError as E:
            errormsg = "The 'bygul' package is required for WastewaterSampler.simulate_sample(). Install it from the Andersen Lab: https://github.com/andersen-lab/Bygul"
            raise ImportError(errormsg) from E

        sample = self.samples[day]
        if sample is None:
            raise ValueError(f'No infectious agents were present on day {day}; cannot simulate.')

        with tempfile.TemporaryDirectory() as tmpdir:
            multifasta_path = os.path.join(tmpdir, f'wastewater_day{day}.fasta')
            csv_path = os.path.join(tmpdir, 'proportions.csv')
            ref = self.reference

            # Build proportions list that sums to exactly 1.0 after float parsing.
            # Adjust the last value to absorb any rounding error.
            props = sample.proportions
            rounded = [round(p, 10) for p in props[:-1]]
            rounded.append(round(1.0 - sum(rounded), 10))

            # Write both the multi-FASTA and the proportions CSV
            with open(multifasta_path, 'w') as f_fasta, open(csv_path, 'w', newline='') as f_csv:
                csv_writer = csv.writer(f_csv)
                csv_writer.writerow(['sample_name', 'proportion'])
                for i, muts in enumerate(sample.mutation_sets):
                    seq_name = f'genotype_{i}'
                    seq = decode_mutations(muts, ref)
                    f_fasta.write(f'>{seq_name}\n{seq}\n') # Append sequence to the multi-FASTA
                    csv_writer.writerow([seq_name, rounded[i]]) # Write row to the CSV mapping name to proportion

            # Build Click CLI args list using the --multifasta and --csv parameters
            args = [
                '--multifasta', multifasta_path,
                '--csv', csv_path,
                '--primers', primers,
                '--reference', reference,
                '--outdir', outdir,
                '--readcnt', str(readcnt),
            ]
            for key, val in bygul_kwargs.items():
                args.extend([f'--{key}', str(val)])
            if redo:
                args.extend(['--redo'])

            # standalone_mode=False returns the result instead of calling sys.exit
            simulate_proportions.main(args=args, standalone_mode=False)
        return
