"""
Variants and (in M4) immunity for Covasim on the Starsim base.

M3 restores ``cv.variant`` -- the v3 public class for adding a co-circulating variant
to a sim. In **Design B** (single ``cv.COVID`` module with an internal variant axis),
``cv.variant`` is NOT a disease/module; it is a lightweight registration + seeding
descriptor, exactly as in v3 (`_v2_legacy/immunity.py`):

  - ``parse()``      -- resolve a string alias (``'alpha'``) or a pars dict into the 5
                        per-variant keys (``rel_beta``/``rel_symp_prob``/...).
  - ``initialize()`` -- register the variant into the single COVID module's
                        ``variant_map``/``variant_pars``, assign ``self.index``, grow ``nv``.
  - ``apply()``      -- on each matched introduction day, seed ``n_imports`` susceptibles
                        with this variant (via ``covid.import_variant``), bumping ``n_imports``.

``cv.Sim(variants=[...])`` registers each into the one module before state allocation;
with no variants (``nv==1``) the module is byte-identical to M2. The cross-immunity
matrix builder (``build_immunity_matrix``) lives here too and is consumed by
``cv.CrossImmunity`` (covasim/connectors.py). NAb time-kinetics / waning are M4.
"""
import numpy as np
import sciris as sc

from . import parameters as cvpar

__all__ = ['variant', 'build_immunity_matrix']


class variant(sc.prettyobj):
    """
    Add a new variant to the sim.

    Args:
        variant (str/dict): name of a predefined variant (``'alpha'``, ``'delta'``, ...),
            or a dict of the per-variant parameters (``rel_beta``/``rel_symp_prob``/
            ``rel_severe_prob``/``rel_crit_prob``/``rel_death_prob``).
        days (int/list): day index (or indices) on which the variant is introduced.
        label (str): if ``variant`` is a dict, the variant's name (dict key).
        n_imports (int): number of infections to import on each introduction day.
        rescale (bool): whether to scale ``n_imports`` down by ``pop_scale`` (so the
            number of *real* introductions is preserved under population scaling).

    **Example**::

        alpha = cv.variant('alpha', days=10, n_imports=20)
        delta = cv.variant('delta', days=30, n_imports=20)
        sim   = cv.Sim(variants=[alpha, delta]).run()
    """

    def __init__(self, variant, days, label=None, n_imports=1, rescale=True):
        self.days      = days
        self.n_imports = int(n_imports)
        self.rescale   = rescale
        self.index     = None  # set by initialize(): this variant's index in the module's variant axis
        self.label     = None  # variant key (dict label)
        self.p         = None  # the 5 per-variant parameters
        self._days     = None  # int day-index set, resolved in initialize()
        self.parse(variant=variant, label=label)
        self.initialized = False
        return

    def parse(self, variant=None, label=None):
        """Unpack the variant info, given as either a predefined-name string or a pars dict."""
        # Option 1: a predefined variant name (or alias)
        if isinstance(variant, str):
            choices, mapping = cvpar.get_variant_choices()
            known_variant_pars = cvpar.get_variant_pars()
            label = variant.lower()
            for txt in ['.', ' ', 'variant', 'voc']:
                label = label.replace(txt, '')
            if label in mapping:
                label = mapping[label]
                variant_pars = known_variant_pars[label]
            else:
                errormsg = f'The selected variant "{variant}" is not implemented; choices are:\n{sc.pp(choices, doprint=False)}'
                raise NotImplementedError(errormsg)

        # Option 2: a dict of per-variant parameters
        elif isinstance(variant, dict):
            default_variant_pars = cvpar.get_variant_pars(default=True)
            default_keys = list(default_variant_pars.keys())
            variant_pars = dict(variant)
            label = variant_pars.pop('label', label) or 'custom'
            invalid = [k for k in variant_pars if k not in default_keys]
            if invalid:
                errormsg = f'Could not parse variant keys "{sc.strjoin(invalid)}"; valid keys are: "{sc.strjoin(default_keys)}"'
                raise sc.KeyNotFoundError(errormsg)
            for key in default_keys:  # populate any missing keys with the defaults
                variant_pars.setdefault(key, default_variant_pars[key])

        else:
            errormsg = f'Could not understand variant of type {type(variant)}; specify a string name or a pars dict.'
            raise ValueError(errormsg)

        self.label = label
        self.p = dict(variant_pars)
        return

    def initialize(self, covid):
        """Register this variant into the single COVID module's variant axis (grow ``nv``)."""
        covid.variant_pars[self.label] = self.p             # store the 5 per-variant pars
        labels = list(covid.variant_pars.keys())            # wild is always first (index 0)
        covid.variant_map = {i: lab for i, lab in enumerate(labels)}
        covid.nv = len(labels)
        self.index = labels.index(self.label)
        self._days = set(int(round(d)) for d in sc.toarray(self.days))  # day indices for introduction
        self.initialized = True
        return

    def apply(self, covid):
        """On a matched introduction day, seed ``n_imports`` susceptibles with this variant."""
        ti = int(covid.ti)
        if self._days is None or ti not in self._days:
            return
        susc = covid.susceptible.uids
        if not len(susc):
            return
        # Rescale the number of imported *agents* by pop_scale (v3 divides by rescale_vec).
        factor = float(covid.sim.pars.pop_scale) if self.rescale else 1.0
        n = sc.randround(self.n_imports / factor) if factor != 1.0 else self.n_imports
        n = int(min(n, len(susc)))
        if n <= 0:
            return
        # Deterministic per-(seed, variant, day) susceptible draw (CRN-friendly, reproducible).
        try:
            base = int(covid.sim.pars.rand_seed)
        except Exception:
            base = 0
        rng = np.random.default_rng([base, 80, int(self.index), ti])
        import starsim as ss
        chosen = ss.uids(np.sort(rng.choice(np.asarray(susc), size=n, replace=False)))
        covid.import_variant(chosen, variant=self.index)
        return


def build_immunity_matrix(variant_map, override=None):
    """Build the asymmetric ``nv x nv`` cross-immunity matrix (v3 ``init_immunity``).

    ``matrix[target, source]`` is the protection a prior ``source`` infection confers against a
    ``target`` challenge (diagonal 1.0 = full homologous protection). Mirrors
    ``_v2_legacy/immunity.py:284-295``: start from ``np.ones((nv,nv))`` and overwrite known pairs
    from ``get_cross_immunity()`` (a dict-of-dicts keyed by variant label).

    Args:
        variant_map (dict): ``{index: label}`` for the variants in the sim.
        override (array): an explicit ``nv x nv`` matrix to use instead of the defaults.

    Returns:
        an ``nv x nv`` float ndarray.
    """
    nv = len(variant_map)
    if override is not None:
        return np.asarray(override, dtype=float).reshape(nv, nv)
    matrix = np.ones((nv, nv), dtype=float)
    cross = cvpar.get_cross_immunity()
    for ti in range(nv):
        label_t = variant_map[ti]
        for si in range(nv):
            label_s = variant_map[si]
            if label_t in cross and label_s in cross[label_t]:
                matrix[ti, si] = cross[label_t][label_s]
    return matrix
