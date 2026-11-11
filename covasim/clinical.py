"""
Clinical sequencing analyzer for Covasim.

ClinicalSequencer is a cv.Analyzer that, on specified days, randomly draws a
fixed number of agents from the current pool of infectious (or symptomatic /
diagnosed) individuals and records each agent's evolved haplotype as a set of
mutations relative to the simulation reference sequence.

This mirrors the real-world practice of routine clinical surveillance sequencing,
where a sample of positive tests is sent for whole-genome sequencing each day.

Requires pars['evo_pars']['enable'] = True.
"""

import collections
import dataclasses
import enum

import sciris as sc
import starsim as ss

from . import misc as cvm
from . import interventions as cvi
from .analysis import Analyzer
from .sequence_evolution import agent_mutations, decode_mutations
from .wastewater import variant_label

__all__ = ['ClinicalPool', 'ClinicalSequencer', 'ClinicalSample']


class ClinicalPool(enum.Enum):
    """Agent pools eligible for clinical sequencing sampling."""
    infectious  = 'infectious'
    symptomatic = 'symptomatic'
    diagnosed   = 'diagnosed'


def coerce_pool(pool):
    """Accept a ClinicalPool or pool name string."""
    if isinstance(pool, ClinicalPool):
        return pool
    if isinstance(pool, str):
        try:
            return ClinicalPool(pool)
        except ValueError:
            valid = [p.value for p in ClinicalPool]
            raise ValueError(f"pool must be one of {valid}, got '{pool}'")
    raise TypeError(f'pool must be a ClinicalPool or str, got {type(pool)}')


@dataclasses.dataclass
class ClinicalSample:
    """Per-agent clinical sequencing snapshot for a single day."""
    day:           int
    date:          str
    agent_id:      int        # agent UID
    mutations:     frozenset  # frozenset of (site_0idx, ref_nt_int, alt_nt_int) vs reference
    variant_label: str        # variant label for this agent


class ClinicalSequencer(Analyzer):
    """
    Analyzer that randomly draws a specified number of infectious agents on
    given days and records each agent's full evolved haplotype.

    For each sampled day:
      1. Identifies agents in the sampling pool (infectious, symptomatic, or
         diagnosed — controlled by the ``pool`` parameter).
      2. Draws ``n_samples`` agents uniformly at random without replacement.
         If the pool is smaller than ``n_samples``, all pool members are
         returned and a warning is issued.
      3. Records each agent's haplotype as a frozenset of
         ``(site_0indexed, ref_nt_int, alt_nt_int)`` mutation tuples relative
         to the simulation reference.  Call ``to_fasta(day)`` to reconstruct
         full ACGT strings on demand.
      4. Stores one ``ClinicalSample`` per agent in ``self.samples[day]``.

    Specimens collected on a sampling day are not necessarily released the
    same day: they first enter a FIFO pending queue, and are only released
    (moved into ``self.samples``) once the queue holds at least
    ``sequencing_batch_size`` specimens.  This models real-world sequencing
    runs, where samples are batched to fill a flow cell before being
    processed.  With the default ``sequencing_batch_size=1``, every
    specimen is released immediately on the day it is collected (the
    original, unbatched behavior).

    Samples are stored in self.samples (dict: int release_day → list[ClinicalSample]),
    one ``ClinicalSample`` per sequenced agent.  Each ``ClinicalSample.day``
    still reflects the day the specimen was *collected*, which may differ
    from the release day it appears under in ``self.samples`` once batching
    is enabled.  Specimens still awaiting release live in ``self.pending``
    (a FIFO queue); ``self.n_pending`` reports its length. Any specimens
    still pending when the sim ends are never released.
    Use to_fasta(day) to get a multi-FASTA string suitable for downstream
    analysis or comparison with wastewater reads.

    Requires pars['evo_pars']['enable'] = True.

    Args:
        days     (list or dict): Simulation days (int) or calendar date strings
                                 at which to sample.  Pass a plain list together
                                 with a scalar ``n_samples`` for uniform sampling,
                                 or a dict mapping day → n_samples for variable
                                 per-day counts.
        n_samples (int):         Number of agents to sample per day when ``days``
                                 is a list.  Ignored when ``days`` is a dict.
        pool      (str or ClinicalPool): Which agents are eligible for sampling.
                                 ``ClinicalPool.diagnosed`` (default) — agents with a
                                 confirmed diagnosis.
                                 ``ClinicalPool.infectious`` — all currently infectious
                                 agents.
                                 ``ClinicalPool.symptomatic`` — agents who are
                                 currently symptomatic.  Equivalent string
                                 values are also accepted.
        sequencing_batch_size (int): Number of specimens that must accumulate in
                                 the pending queue before a batch is released
                                 (default 1 = release immediately on collection).
                                 Must be a positive integer.
        label     (str):         Optional label for the analyzer.
        kwargs (dict): passed to ``cv.Analyzer``

    **Example**::

        cs = cv.ClinicalSequencer(days=[56, 120], n_samples=50)
        sim = cv.Sim(pars, analyzers=[cs])
        sim.run()
        cs = sim.get_analyzer('ClinicalSequencer')
        print(cs.to_fasta(56))
    """

    def __init__(self, days, n_samples=None, pool='diagnosed', sequencing_batch_size=1, label=None, **kwargs):
        if label is None:
            label = 'ClinicalSequencer'
        super().__init__(label=label, **kwargs)
        if not sc.isnumber(sequencing_batch_size) or int(sequencing_batch_size) != sequencing_batch_size or sequencing_batch_size < 1:
            errormsg = f'sequencing_batch_size must be a positive integer, got {sequencing_batch_size}'
            raise ValueError(errormsg)
        if not isinstance(days, dict) and n_samples is None:
            raise ValueError('n_samples must be provided when days is a list')
        self.days                  = days # Converted to {day index: number of samples} in init_post()
        self.n_samples             = n_samples
        self.pool                  = coerce_pool(pool)
        self.sequencing_batch_size = int(sequencing_batch_size)
        self.samples               = {}
        self.pending               = collections.deque()  # FIFO queue of collected-but-unreleased ClinicalSamples
        self.reference             = None  # set in init_post()
        self.choose = ss.choose_n() # Who to sequence on each day
        return

    @property
    def n_pending(self):
        """Number of collected specimens awaiting release in the batch queue."""
        return len(self.pending)

    def init_post(self):
        super().init_post()
        covid = self.sim.diseases.covid
        if covid.sequence_tracker is None:
            errormsg = "ClinicalSequencer requires pars['evo_pars']['enable'] = True"
            raise ValueError(errormsg)
        self.reference = covid.sequence_tracker.reference

        # Build the {day index: n_samples} lookup
        if isinstance(self.days, dict):
            days = cvi.process_days(self.sim, list(self.days.keys()))
            counts = list(self.days.values())
        else:
            days = cvi.process_days(self.sim, sc.tolist(self.days))
            counts = [self.n_samples]*len(days)
        self.day_to_n = {int(day):int(n) for day,n in zip(days, counts)}
        return

    def step(self):
        if self.today in self.day_to_n:
            self.take_sample(self.day_to_n[self.today])
        return

    def take_sample(self, n):
        """ Sequence n agents from the pool on this day """
        sim     = self.sim
        day     = self.today
        covid   = sim.diseases.covid
        tracker = covid.sequence_tracker

        # The eligible pool
        if self.pool == ClinicalPool.infectious:
            pool_uids = covid.infectious.uids
        elif self.pool == ClinicalPool.symptomatic:
            pool_uids = covid.symptomatic.uids
        else:  # ClinicalPool.diagnosed
            pool_uids = covid.diagnosed.uids

        if len(pool_uids) == 0:
            return

        if len(pool_uids) < n:
            cvm.warn(f"ClinicalSequencer day {day}: requested {n} samples but only "
                     f"{len(pool_uids)} agents are available in pool='{self.pool.value}'. "
                     f"Returning all {len(pool_uids)} agents.")
            sampled = pool_uids
        else:
            self.choose.set(n=n)
            sampled = self.choose.filter(pool_uids)

        date = sim.date(day)
        day_samples = []
        for uid in sampled:
            day_samples.append(ClinicalSample(
                day           = day,
                date          = date,
                agent_id      = int(uid),
                mutations     = agent_mutations(tracker, uid),
                variant_label = variant_label(covid, covid.infectious_variant[uid]),
            ))

        self.pending.extend(day_samples)
        self.release_batches(day)
        return

    def release_batches(self, release_day):
        """Move complete batches from the pending FIFO queue into self.samples."""
        while len(self.pending) >= self.sequencing_batch_size:
            batch = [self.pending.popleft() for _ in range(self.sequencing_batch_size)]
            self.samples.setdefault(release_day, []).extend(batch)
        return

    def to_fasta(self, day):
        """Return a multi-FASTA string, one record per sampled agent, for the given day."""
        day_samples = self.samples.get(day, [])
        if not day_samples:
            return ''
        ref   = self.reference
        lines = []
        for s in day_samples:
            lines.append(f'>agent_{s.agent_id}  day={s.day}  date={s.date}  variant={s.variant_label}')
            lines.append(decode_mutations(s.mutations, ref))
        return '\n'.join(lines)
