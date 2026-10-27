"""
Variants and immunity for Covasim on the Starsim base.

``cv.variant`` adds a co-circulating variant to a sim. Since there is a single ``cv.COVID``
module with an internal variant axis, ``cv.variant`` is not a module: as in v3, it registers
the variant's parameters with the COVID module and seeds infections with it on the days it
is introduced. ``cv.Sim(variants=[...])`` registers each into the module before the states
are allocated. The cross-immunity matrix builder (``build_immunity_matrix``) and the
neutralizing antibody (NAb) waning functions live here too, and are used by ``cv.COVID`` and
``cv.CrossImmunity`` (connectors.py). As in v3 unless noted.
"""
import numpy as np
import sciris as sc
import starsim as ss

from . import parameters as cvpar

__all__ = ['variant', 'build_immunity_matrix']


class variant(sc.prettyobj):
    """
    Add a new variant to the sim.

    Args:
        variant (str/dict): name of a predefined variant (``'alpha'``, ``'delta'``, ...),
            or a dict of the per-variant parameters (``rel_beta``/``rel_symp_prob``/
            ``rel_severe_prob``/``rel_crit_prob``/``rel_death_prob``).
        days (int/str/list): day(s) on which the variant is introduced, as day indices or dates (e.g. ``'2020-03-15'``).
        label (str): if ``variant`` is a dict, the variant's name (dict key).
        n_imports (int): number of infections to import on each introduction day.
        rescale (bool): whether to scale ``n_imports`` down by ``pop_scale`` (so the
            number of *real* introductions is preserved under population scaling).

    **Examples**::

        alpha = cv.variant('alpha', days=10, n_imports=20)
        delta = cv.variant('delta', days='2020-04-01', n_imports=20)
        sim   = cv.Sim(variants=[alpha, delta]).run()

        my_var = cv.variant(variant={'rel_beta': 2.5}, label='My variant', days=20) # A custom variant
    """

    def __init__(self, variant, days, label=None, n_imports=1, rescale=True):
        self.days      = days  # converted to day indices by process_days()
        self.n_imports = int(n_imports)
        self.rescale   = rescale
        self.index     = None  # set by initialize(): this variant's index in the module's variant axis
        self.label     = None  # variant key (dict label)
        self.p         = None  # the 5 per-variant parameters
        self.choose_imports = ss.choose_n() # Who to infect on each introduction day
        self.round_imports = ss.constant(v=0, round='stochastic') # Round the number of imports up or down, with rescaling
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
        if self.label not in covid.variant_pars: # Add a new variant; wild is always index 0
            covid.variant_map[len(covid.variant_map)] = self.label
        covid.variant_pars[self.label] = self.p
        covid.nv = len(covid.variant_map)
        self.index = list(covid.variant_map.values()).index(self.label)
        self.initialized = True
        return

    def process_days(self, sim):
        """Convert the days (day indices or dates) to day indices, as for interventions; called by ``cv.COVID.init_pre()``"""
        self.days = sc.toarray(sim.day(self.days)).astype(int)
        return

    def apply(self, covid):
        """On a matched introduction day, seed ``n_imports`` susceptibles with this variant."""
        day = covid.today
        if day not in self.days:
            return
        susc = covid.susceptible.uids
        if not len(susc):
            return
        # Rescale the number of imported agents by the current population scale (as in v3)
        n = self.n_imports
        if self.rescale:
            factor = covid.sim.current_scale
            self.round_imports.set(v=self.n_imports / factor)
            n = int(self.round_imports.rvs(1)[0])
            if (n == 0) and (self.n_imports > 0) and covid.sim.pars.verbose: # As in v3, print a message rather than raising a warning
                print(f'Warning: {self.n_imports} imported infections of {self.label} were specified on day {day}, but given the rescale factor of {factor:n}, no agents were infected. Increase the number of imports or use more agents.')
        if n <= 0:
            return
        self.choose_imports.set(n=n)
        chosen = self.choose_imports.filter(susc)
        covid.import_variant(chosen, variant=self.index)
        return


def build_immunity_matrix(variant_map, override=None):
    """Build the asymmetric ``nv x nv`` cross-immunity matrix (v3 ``init_immunity``).

    ``matrix[target, source]`` is the protection a prior ``source`` infection confers against a
    ``target`` challenge (diagonal 1.0 = full homologous protection). Start from ``np.ones((nv,nv))``
    and overwrite known pairs from ``get_cross_immunity()`` (a dict-of-dicts keyed by variant label).

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


# %% Neutralizing-antibody (NAb) waning functions

def calc_VE(nab, ax, pars):
    """Map effective NAb levels to a per-axis immune-protection factor.

    Inverse-logit form from https://doi.org/10.1101/2021.03.09.21252641:
    ``exp(alpha)·nab**beta / (1 + exp(alpha)·nab**beta)``, with ``(alpha, beta)`` selected per axis
    from ``nab_eff``. ``ax`` is 'sus' (infection), 'symp' (symptoms), or 'sev' (severe disease).
    ``calc_VE(0) == 0`` (since ``0**beta == 0`` for the positive betas used).
    """
    axis_pars = dict( # The names of the (alpha, beta) parameters for each axis
        sus  = ('alpha_inf', 'beta_inf'),
        symp = ('alpha_symp_inf', 'beta_symp_inf'),
        sev  = ('alpha_sev_symp', 'beta_sev_symp'),
    )
    if ax not in axis_pars:
        raise ValueError(f'Axis {ax!r} not in {list(axis_pars.keys())}')
    alpha_key, beta_key = axis_pars[ax]
    nab = np.asarray(nab, dtype=float)
    exp_lo = np.exp(pars[alpha_key]) * nab ** pars[beta_key]
    return exp_lo / (1 + exp_lo)  # inverse logit


def calc_VE_symp(nab, pars):
    """Marginal vaccine efficacy against symptomatic disease.

    ``VE_symp = 1 - (1 - VE_inf)·(1 - VE_symp|inf)`` where each factor is an inverse-logit of NAb. Used
    by the vaccine ``target_eff`` back-calculation to map a target efficacy onto a peak NAb level.
    """
    return 1 - (1 - calc_VE(nab, 'sus', pars)) * (1 - calc_VE(nab, 'symp', pars))


def precompute_waning(length, pars=None):
    """Precompute the per-timestep NAb waning kernel.

    Dispatches on ``pars['form']``: 'nab_growth_decay' (default), 'nab_decay', 'exp_decay', or a
    callable. Returns an array of length ``length`` giving the per-step NAb increment relative to peak.
    If ``pars`` is None, use the default ``nab_decay`` parameters.
    """
    if pars is None:
        pars = cvpar.make_pars()['nab_decay']
    pars = sc.dcp(pars)
    form = pars.pop('form')
    if form is None or form == 'nab_growth_decay':
        return nab_growth_decay(length, **pars)
    elif form == 'nab_decay':
        return nab_decay(length, **pars)
    elif form == 'exp_decay':
        if pars['half_life'] is None:
            pars['half_life'] = np.nan
        return exp_decay(length, **pars)
    elif callable(form):
        return form(length, **pars)
    errormsg = f"Functional form {form!r} not implemented; choices: nab_growth_decay, nab_decay, exp_decay."
    raise NotImplementedError(errormsg)


def nab_growth_decay(length, growth_time, decay_rate1, decay_time1, decay_rate2, decay_time2):
    """Linear NAb growth then two-phase exponential decay (v3 default kernel).

    Based on Khoury et al. (https://www.nature.com/articles/s41591-021-01377-8): linear growth over
    ``growth_time`` days, then exponential decay whose rate decays linearly from ``decay_rate1`` to
    ``decay_rate2`` between ``decay_time1`` and ``decay_time2``.
    """
    if decay_time2 < decay_time1:
        raise ValueError(f'decay_time2 ({decay_time2}) must be >= decay_time1 ({decay_time1}).')

    def f1(t):
        return (1.0 / growth_time) * t  # linear growth

    def f2(t):
        decayRate = np.full(len(t), fill_value=decay_rate1, dtype=float)
        decayRate[np.nonzero(t > decay_time2)[0]] = decay_rate2
        slowing = (1.0 / (decay_time2 - decay_time1)) * (decay_rate1 - decay_rate2)
        mid = np.nonzero((t > decay_time1) * (t <= decay_time2))[0]
        decayRate[mid] = decay_rate1 - slowing * np.arange(len(mid), dtype=int)
        titre = np.zeros(len(t))
        for i in range(1, len(t)):
            titre[i] = titre[i - 1] + decayRate[i]
        return np.exp(-titre)

    length = length + 1
    t1 = np.arange(growth_time, dtype=int)
    t2 = np.arange(length - growth_time, dtype=int)
    y = np.concatenate([f1(t1), f2(t2)])
    return np.diff(y)[0:length]


def nab_decay(length, decay_rate1, decay_time1, decay_rate2):
    """Exponential NAb decay whose rate itself decays after ``decay_time1``."""
    def f1(t):
        return np.exp(-t * decay_rate1)

    def f2(t):
        return np.exp(-t * (decay_rate1 * np.exp(-(t - decay_time1) * decay_rate2)))

    t = np.arange(length, dtype=int)
    y1 = f1(t[np.nonzero(t <= decay_time1)[0]])
    y2 = f2(t[np.nonzero(t > decay_time1)[0]])
    y = np.concatenate([[-np.inf], y1, y2])
    y = np.diff(y)[0:length]
    y[0] = 1
    return y


def exp_decay(length, init_val, half_life, delay=None):
    """Simple exponential NAb decay with an optional linear-growth ``delay``."""
    length = length + 1
    decay_rate = np.log(2) / half_life if not np.isnan(half_life) else 0.0
    if delay is not None:
        t = np.arange(length - delay, dtype=int)
        growth = linear_growth(delay, init_val/delay)
        decay = init_val * np.exp(-decay_rate * t)
        result = np.concatenate([growth, decay], axis=None)
    else:
        t = np.arange(length, dtype=int)
        result = init_val * np.exp(-decay_rate * t)
    return np.diff(result)


def linear_decay(length, init_val, slope):
    """Calculate linear decay; not used by default, but available as a custom waning ``form``"""
    result = -slope*np.ones(length)
    result[0] = init_val
    return result


def linear_growth(length, slope):
    """Calculate linear growth"""
    return slope*np.ones(length)
