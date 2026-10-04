"""
Defines the Sim class for Covasim on the Starsim base.

``cv.Sim(ss.Sim)`` is a thin wrapper that assembles the Covasim modules -- a
``cv.People``, one ``cv.Network`` per contact layer, and a ``cv.COVID`` disease --
and forwards to ``ss.Sim`` with a daily timestep. Per-layer transmissibility
(Covasim's ``beta * beta_layer``) is carried on the disease. ``pop_infected``
agents are seeded exactly at t=0.

Passing ``people=`` / ``networks=`` / ``diseases=`` overrides the corresponding
default.
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

from . import version as cvv
from . import misc as cvm
from . import data as cvdata
from . import compat as cvc
from . import parameters as cvpar
from . import people as cvppl
from . import network as cvnet
from . import covid as cvcov
from . import interventions as cvi
from . import plotting as cvplt
from . import connectors as cvconn
from . import analysis as cva

__all__ = ['Sim', 'AlreadyRunError', 'demo']

# Raised if a sim is run when it has already been run; the same as Starsim's
AlreadyRunError = ss.AlreadyRunError


class Sim(cvc.V3Sim, ss.Sim):
    """
    The Covasim simulation, on the Starsim base.

    Parameters can be supplied as a dict and/or as keyword arguments, using the v3 names (see
    ``cv.make_pars()`` for the full list and their defaults), e.g. ``cv.Sim(pop_size=10e3,
    pop_type='hybrid', beta=0.02)``. The COVID parameters are stored on the disease module
    (``sim.diseases.covid.pars``) and the sim-level ones in ``sim.pars``; ``sim['key']`` gets or
    sets either.

    Args:
        pars (dict): parameters to modify from their default values
        datafile (str/df): filename of (Excel, CSV) data file to load, or a pandas dataframe of the data
        label (str): the name of the simulation (useful to distinguish in batch runs)
        simfile (str): the filename for this simulation, if it's saved
        popfile (str): not supported in v4 (v3 populations cannot be loaded)
        people (People): optionally, the people to use instead of creating them
        version (str): if supplied, use default parameters from this version of Covasim instead of the latest
        kwargs (dict): additional parameters; also passed to ``ss.Sim`` (e.g. ``networks``, ``diseases``, ``connectors``)
    """

    # v3 parameters that are calculated from the others, and so are ignored if supplied (e.g. from cv.make_pars())
    _v3_derived_pars = ['n_variants', 'nab_kin', 'immunity', 'vaccine_pars', 'vaccine_map', 'variant_map', 'variant_pars']

    def __init__(self, pars=None, datafile=None, label=None, simfile=None, popfile=None, people=None, version=None, **kwargs):

        # Parameters can be supplied as a dict (the v3 form) and/or as keyword arguments, which take precedence
        pars = sc.mergedicts(pars, kwargs, _copy=True)
        if 'n_agents' in pars: # The Starsim name for pop_size
            pars['pop_size'] = pars.pop('n_agents')
        if 'dur' in pars and not isinstance(pars['dur'], dict): # The Starsim sim duration, rather than the v3 durations
            dur = pars.pop('dur')
            pars['n_days'] = dur.value if isinstance(dur, ss.dur) else dur
        if popfile is not None:
            errormsg = 'Loading a v3 population (popfile) is not supported in Covasim v4; please create the population instead'
            raise NotImplementedError(errormsg)

        # Split out the v3 parameters, using the defaults (from parameters.py) for any that aren't supplied
        defaults = cvpar.make_pars(version=version)
        v3 = {key:pars.pop(key) for key in list(pars) if key in defaults} # The user-supplied v3 parameters
        if version is not None: # Use all the parameters from this version, rather than the latest defaults
            v3 = sc.mergedicts(defaults, v3)
        for key in self._v3_derived_pars:
            v3.pop(key, None)
        get = lambda key: v3.pop(key, defaults[key]) # pylint: disable=unnecessary-lambda-assignment

        # Sim-level parameters
        pop_size     = int(get('pop_size'))
        pop_infected = int(get('pop_infected'))
        pop_type     = get('pop_type')
        location     = get('location')
        start_day    = get('start_day')
        end_day      = get('end_day')
        n_days       = get('n_days')
        rand_seed    = get('rand_seed')
        verbose      = get('verbose')
        pop_scale    = get('pop_scale')
        total_pop    = get('scaled_pop')
        use_waning   = get('use_waning')
        variants     = get('variants')
        for key in ['interventions', 'analyzers']:
            if key in v3:
                pars[key] = v3.pop(key)
        self.timelimit     = get('timelimit')
        self.stopping_func = get('stopping_func')
        for key in ['rescale', 'rescale_threshold', 'rescale_factor']: # Dynamic rescaling, done by Starsim
            pars[key] = get(key)
        if end_day is not None: # As in v3, an end_day takes precedence over n_days
            n_days = int(sc.daydiff(start_day, end_day))
        if pop_type not in ['random', 'hybrid']:
            errormsg = f'pop_type "{pop_type}" not supported (choices: "random", "hybrid")'
            raise ValueError(errormsg)

        # The per-layer parameters (beta_layer, contacts, etc.): user values, filled in with the defaults for this pop_type
        layer_pars = dict(pop_type=pop_type)
        for key in cvpar.layer_pars:
            if key in v3:
                layer_pars[key] = v3.pop(key)
        cvpar.reset_layer_pars(layer_pars)
        self.dynam_layer = layer_pars['dynam_layer'] # Which networks are recreated on each timestep (applied in init())

        # Location-specific data: the age distribution, and the household size. The People are created when the
        # sim is initialized (so pop_size can still be changed).
        self._age_data = None
        if location is not None:
            sc.printv(f'Loading location-specific data for "{location}"', 1, verbose)
            try:
                self._age_data = cvppl.convert_age_data(cvdata.get_age_distribution(location))
            except ValueError as E:
                cvm.warn(f'Could not load age data for requested location "{location}" ({str(E)}), using default')
            try:
                household_size = cvdata.get_household_size(location)
                if 'h' in layer_pars['contacts']:
                    layer_pars['contacts']['h'] = household_size - 1 # Subtract 1 because e.g. each person in a 3-person household has 2 contacts
                elif verbose:
                    keystr = ', '.join(list(layer_pars['contacts'].keys()))
                    cvm.warn(f'Not loading household size for "{location}" since no "h" key; keys are "{keystr}". Try "hybrid" population type?')
            except ValueError as E:
                if verbose > 1: # These don't exist for many locations, so skip the warning by default
                    cvm.warn(f'Could not load household size data for requested location "{location}" ({str(E)}), using default')

        # One network per contact layer
        networks = pars.pop('networks', None)
        if networks is None:
            networks = cvnet.make_networks(pop_type, contacts=layer_pars['contacts'])

        # The COVID disease: the remaining v3 parameters, plus any others that are COVID parameters (e.g. dur_exp2inf)
        diseases = pars.pop('diseases', None)
        if diseases is None:
            diseases = cvcov.COVID(init_prev=ss.choose_n(pop_infected), variants=variants, use_waning=use_waning,
                                   beta_layer=layer_pars['beta_layer'], iso_factor=layer_pars['iso_factor'], quar_factor=layer_pars['quar_factor'])
            if 'dur' in v3:
                v3.update(cvcov.v3_durs(v3.pop('dur')))
            if 'beta' in v3:
                v3['beta'] = ss.probperday(v3['beta'])
            v3.update({key:pars.pop(key) for key in list(pars) if key in diseases.pars})
            diseases.pars.update(v3)
        elif v3:
            errormsg = f'Cannot set COVID parameters {sc.strjoin(v3.keys())} if also supplying diseases; please set them on the disease instead'
            raise ValueError(errormsg)

        # Add the cross-immunity connector if waning immunity is on, or if more than one variant circulates: it applies
        # cross-immunity each step (NAb-weighted under use_waning, else the static matrix) and enables reinfection
        connectors = pars.pop('connectors', None)
        if connectors is None and (getattr(diseases, 'nv', 1) > 1 or getattr(diseases.pars, 'use_waning', False)):
            connectors = cvconn.CrossImmunity()

        # Absolute population scaling: each agent represents pop_scale real people. Starsim computes one from the
        # other, so only pass one of them.
        if total_pop is not None:
            pars['total_pop'] = total_pop
        elif pop_scale is not None:
            pars['pop_scale'] = pop_scale

        super().__init__(pars=pars, label=label, people=people, networks=networks, diseases=diseases, connectors=connectors,
                         n_agents=pop_size, start=ss.date(start_day), dur=ss.days(n_days), dt=ss.days(1),
                         rand_seed=rand_seed, verbose=verbose)
        self.pop_type = pop_type
        self.simfile = simfile

        # Report Covasim's version and git info (v3 sim.version / sim.git_info), not Starsim's
        self.version = cvv.__version__
        try:
            self.git_info = cvm.git_info(verbose=False)
        except Exception:
            self.git_info = None

        # Optional data to fit against, read into ``self.data`` for ``cv.Fit`` / ``sim.compute_fit``
        self.data = None
        if datafile is not None:
            self.data = cvm.load_data(datafile) # Also for a dataframe, e.g. to add the cumulative columns, as in v3
        return

    def init(self, *args, **kwargs):
        """Create the People (with Covasim's age distribution), then initialize as usual."""
        if not self.initialized:
            self._orig_sim = None
            try: # Keep a copy of the sim before it was initialized, so it can be reset (v3 sim.initialize(reset=True)); stored as bytes so Starsim doesn't treat it as part of this sim
                self._orig_sim = sc.dumpstr(self)
            except Exception: # E.g. if a user-defined object can't be pickled; then the sim can't be reset
                self._orig_sim = None
        if not self.initialized:
            self.set_seed() # As in v3, seed the global random number generators, for user code that uses them
        if self.pars.n_agents < 0:
            errormsg = f'Population size cannot be negative ({self.pars.n_agents})'
            raise ValueError(errormsg)
        if self.pars.get('people') is None:
            self.pars.people = cvppl.People(self.pars.n_agents, age_data=self._age_data)
        ivs = self.pars.interventions # v3: interventions can be defined as dicts, e.g. dict(which='change_beta', pars=dict(days=10, changes=0.5))
        if isinstance(ivs, dict) and 'which' in ivs:
            ivs = [ivs]
        if isinstance(ivs, list):
            self.pars.interventions = [cvi.InterventionDict(**iv) if isinstance(iv, dict) else iv for iv in ivs]
        super().init(*args, **kwargs)
        for key,dynamic in self.dynam_layer.items(): # v3 dynamic layers are networks that are recreated on each timestep
            if dynamic and key in self.networks:
                self.networks[key].pars.dynamic = True
        self.reset_layer_pars() # As in v3, use the default per-layer parameters (e.g. iso_factor) for any networks that don't have them, e.g. user-supplied ones
        return self

    def day(self, day, *args):
        """Convert date(s) to integer day index/indices relative to ``start_day`` (v3 ``Sim.day``).

        Numbers (and numeric arrays) are treated as day offsets and passed through; only genuine dates
        (strings / datetimes) are converted. Accepts a scalar, list, or array.
        """
        start = self['start_day']
        def _one(d):
            if isinstance(d, (int, np.integer)):
                return int(d)
            if isinstance(d, (float, np.floating)):
                return int(round(d))
            return int(sc.daydiff(start, d))  # a date string / datetime
        inputs = list(day) if isinstance(day, (list, tuple, np.ndarray)) else [day]
        inputs += list(args)
        out = [_one(d) for d in inputs]
        return out[0] if len(out) == 1 else out

    def date(self, *args, **kwargs):
        """Convert day index/indices to date string(s) relative to ``start_day`` (v3 ``Sim.date``)."""
        kwargs.setdefault('as_date', False) # v3 returns date strings
        return sc.date(*args, start_date=self['start_day'], **kwargs)

    def _get_ia(self, which, label=None, partial=False, as_list=False, as_inds=False, die=True, first=False):
        """Helper method for get_interventions() and get_analyzers(); see get_interventions()"""
        if (not self.initialized) or self._is_v3_saved(): # v3: available before initialization; also, a sim saved by v3 stores them in its parameters
            ia_list = [ia for ia in sc.tolist(self.pars[which]) if not isinstance(ia, dict)]
        else:
            ia_list = list(getattr(self, which).values())
        n_ia = len(ia_list)
        if label is None: # Get all of them
            label = list(range(n_ia))
        labels = sc.tolist(label.tolist() if isinstance(label, np.ndarray) else label)

        # Calculate the matches
        matches = []
        match_inds = []
        for label in labels:
            if sc.isnumber(label):
                matches.append(ia_list[label]) # This will raise an exception if an invalid index is given
                match_inds.append(n_ia + label if label < 0 else label)
            elif sc.isstring(label) or isinstance(label, type):
                for ind,ia_obj in enumerate(ia_list):
                    if isinstance(label, type):
                        is_match = isinstance(ia_obj, label) or str(ia_obj.__class__) == str(label) # As in v3, also match by class name, since a class defined in a script or notebook can be a different object after a parallel run
                    else:
                        is_match = label in [ia_obj.label, ia_obj.name] or (partial and label in str(ia_obj.label))
                    if is_match:
                        matches.append(ia_obj)
                        match_inds.append(ind)
            else:
                errormsg = f'Could not interpret label type "{type(label)}": should be str, int, list, or {which} class'
                raise TypeError(errormsg)

        # Parse the output options
        if as_inds:
            return match_inds
        elif as_list:
            return matches
        elif len(matches):
            return matches[0 if first else -1]
        elif die:
            errormsg = f'No {which} matching "{label}" were found'
            raise ValueError(errormsg)
        return None

    def get_interventions(self, label=None, partial=False, as_inds=False):
        """
        Find the matching intervention(s) by label, index, or type (the v3 ``Sim.get_interventions``).
        If None, return all interventions.

        Args:
            label (str, int, Intervention, list): the label, index, or type of intervention to get; if a list, iterate over one of those types
            partial (bool): if true, return partial matches (e.g. 'beta' will match all beta interventions)
            as_inds (bool): if true, return matching indices instead of the actual interventions

        **Examples**::

            tp = cv.test_prob(symp_prob=0.1)
            cb1 = cv.change_beta(days=5, changes=0.3, label='NPI')
            cb2 = cv.change_beta(days=10, changes=0.3, label='Masks')
            sim = cv.Sim(interventions=[tp, cb1, cb2])
            cb1, cb2 = sim.get_interventions(cv.change_beta)
            tp, cb2 = sim.get_interventions([0,2])
            ind = sim.get_interventions(cv.change_beta, as_inds=True) # Returns [1,2]
        """
        return self._get_ia('interventions', label=label, partial=partial, as_inds=as_inds, as_list=True)

    def get_intervention(self, label=None, partial=False, first=False, die=True):
        """
        Find the matching intervention by label, index, or type (the v3 ``Sim.get_intervention``). If more than
        one intervention matches, return the last by default. If no label is provided, return the last intervention.

        Args:
            label (str, int, Intervention): the label, index, or type of intervention to get
            partial (bool): if true, return partial matches (e.g. 'beta' will match all beta interventions)
            first (bool): if true, return first matching intervention (otherwise, return last)
            die (bool): whether to raise an exception if no intervention is found
        """
        return self._get_ia('interventions', label=label, partial=partial, first=first, die=die, as_inds=False, as_list=False)

    def get_analyzers(self, label=None, partial=False, as_inds=False):
        """Same as get_interventions(), but for analyzers (the v3 ``Sim.get_analyzers``)."""
        return self._get_ia('analyzers', label=label, partial=partial, as_list=True, as_inds=as_inds)

    def get_analyzer(self, label=None, partial=False, first=False, die=True):
        """Same as get_intervention(), but for analyzers (the v3 ``Sim.get_analyzer``)."""
        return self._get_ia('analyzers', label=label, partial=partial, first=first, die=die, as_inds=False, as_list=False)

    def compute_fit(self, *args, **kwargs):
        """Compute the goodness-of-fit against ``self.data`` (v3 ``Sim.compute_fit``); returns a ``cv.Fit``."""
        self.fit = cva.Fit(self, *args, **kwargs)
        return self.fit

    def compute_r_eff(self, method='daily', smoothing=2, window=7):
        """
        Effective reproduction number based on number of people each person infected (v3 ``Sim.compute_r_eff``).

        Args:
            method (str): 'daily' uses daily infections, 'infectious' counts from the date infectious, 'outcome' counts from the date recovered/dead
            smoothing (int): the number of steps to smooth over for the 'daily' method
            window (int): the size of the window used for 'infectious' and 'outcome' calculations (larger values are more accurate but less precise)

        Returns:
            r_eff (array): the r_eff results array
        """
        covid = self.diseases.covid
        results = covid.results
        window = int(window)
        if method == 'daily':
            covid._compute_r_eff(results, smoothing=smoothing)
            values = results['r_eff'].values
        elif method in ['infectious', 'outcome']:
            # Count the sources on each day, and store a mapping from each source to their date
            sources = np.zeros(self.npts)
            targets = np.zeros(self.npts)
            if method == 'infectious':
                dates = covid.ti_infectious.raw
            else:
                dates = np.fmin(covid.ti_recovered.raw, covid.ti_dead.raw) # Whichever happened
            source_dates = {}
            for t in self.tvec:
                inds = np.flatnonzero(dates == t)
                sources[t] = len(inds)
                source_dates.update({ind:t for ind in inds})

            # Count the targets of each source, skipping seed infections and people with e.g. recovery after the end of the sim
            for entry in cva.make_infection_log(self):
                source = entry['source']
                if source is not None and source in source_dates:
                    targets[source_dates[source]] += 1

            # Calculate the moving average over the window, weighted by the number of sources
            r_eff = np.divide(targets, sources, out=np.full(self.npts, np.nan), where=sources > 0)
            num = np.nancumsum(r_eff * sources)
            num[window:] = num[window:] - num[:-window]
            den = np.cumsum(sources)
            den[window:] = den[window:] - den[:-window]
            values = np.divide(num, den, out=np.full(self.npts, np.nan), where=den > 0)
            results['r_eff'].values[:] = values
        else:
            errormsg = f'Method must be "daily", "infectious", or "outcome", not "{method}"'
            raise ValueError(errormsg)
        return values

    def compute_gen_time(self):
        """
        Calculate the generation time (or serial interval) (v3 ``Sim.compute_gen_time``). There are two
        ways to do this calculation. The 'true' interval (exposure time to exposure time) or 'clinical'
        (symptom onset to symptom onset).

        Returns:
            gen_time (dict): the generation time results
        """
        covid = self.diseases.covid
        date_exposed = covid.ti_exposed.raw
        date_symptomatic = covid.ti_symptomatic.raw
        intervals1 = []
        intervals2 = []
        for entry in cva.make_infection_log(self):
            source, target = entry['source'], entry['target']
            if source is not None:
                intervals1.append(date_exposed[target] - date_exposed[source])
                if np.isfinite(date_symptomatic[source]) and np.isfinite(date_symptomatic[target]):
                    intervals2.append(date_symptomatic[target] - date_symptomatic[source])
        self.gen_time = sc.objdict(
            true         = np.mean(intervals1) if intervals1 else np.nan,
            true_std     = np.std(intervals1) if intervals1 else np.nan,
            clinical     = np.mean(intervals2) if intervals2 else np.nan,
            clinical_std = np.std(intervals2) if intervals2 else np.nan,
        )
        return self.gen_time

    def make_age_histogram(self, *args, output=True, **kwargs):
        """
        Calculate the age histograms of infections, deaths, diagnoses, etc. (v3 ``Sim.make_age_histogram``).
        See cv.age_histogram() for more information. This can be used instead of adding the age histogram
        as an analyzer to the sim, but it can only record the final time point.

        Args:
            output (bool): whether or not to return the age histogram; if not, store in sim.agehist
            args   (list): passed to cv.age_histogram()
            kwargs (dict): passed to cv.age_histogram()
        """
        if not self.results_ready:
            errormsg = 'Cannot make age histogram since results are not ready yet -- did you run the sim?'
            raise RuntimeError(errormsg)
        agehist = cva.age_histogram(*args, sim=self, **kwargs)
        if output:
            return agehist
        else:
            self.agehist = agehist
            return

    def make_transtree(self, *args, output=True, **kwargs):
        """
        Create a TransTree (transmission tree) object (v3 ``Sim.make_transtree``). See cv.TransTree().

        Args:
            output (bool): whether or not to return the TransTree; if not, store in sim.transtree
            args   (list): passed to cv.TransTree()
            kwargs (dict): passed to cv.TransTree()
        """
        if not self.results_ready:
            errormsg = 'Cannot compute transmission tree since results are not ready yet -- did you run the sim?'
            raise RuntimeError(errormsg)
        tt = cva.TransTree(self, *args, **kwargs)
        if output:
            return tt
        else:
            self.transtree = tt
            return

    def brief(self, output=False):
        """Print (or return) a one-line summary of the sim (v3 ``Sim.brief``)."""
        covid = self.diseases.get('covid') if hasattr(self, 'diseases') else None
        if covid is not None and 'cum_infections' in covid.results:
            ci = float(np.asarray(covid.results['cum_infections']).max())
            string = f'Sim({self.label!r}; {self["n_days"]} days; {self["pop_size"]} agents; {ci:n} cumulative infections)'
        else:
            string = f'Sim({self.label!r}; {self["n_days"]} days; {self["pop_size"]} agents; not run)'
        if output:
            return string
        print(string)
        return

    def export_pars(self, filename=None, indent=2, **kwargs):
        """Export the sim's parameters to a JSON-compatible dict (the v3 ``BaseSim.export_pars``).

        Returns the resolved Covasim sim-level config plus the COVID module's scalar parameters; if
        ``filename`` is given, also writes the dict to JSON. Per-distribution / array parameters are
        summarised rather than dumped.

        Args:
            filename (str): if given, write the JSON here.
            indent (int): JSON indent.
        """
        pars = {key:self[key] for key in ['pop_size', 'pop_infected', 'pop_type', 'n_days', 'start_day', 'end_day', 'rand_seed', 'use_waning']}
        covid = self._v3_covid()
        if covid is not None:
            covid_pars = {}
            for key, val in dict(covid.pars).items():
                try:
                    covid_pars[key] = sc.jsonify(val, die=False)  # scalars/lists/dicts; skip the rest
                except Exception:
                    covid_pars[key] = str(type(val).__name__)
            pars['covid'] = covid_pars
        pars = sc.jsonify(pars, die=False)
        if filename is not None:
            sc.savejson(filename, pars, indent=indent)
        return pars

    def init_results(self):
        """
        Initialize the results, then make the COVID results available at the top level, as in v3, e.g.
        ``sim.results['cum_deaths']`` and ``sim.results['variant']['new_infections_by_variant']``. These are
        references to the module's results (not copies).
        """
        super().init_results()
        self._bridged_keys = []
        covid = self.diseases.get('covid')
        if covid is None:
            return
        variant = ss.Results(module=self.label)
        for key, res in covid.results.items():
            if key.endswith('_by_variant'):
                variant[key] = res
            elif isinstance(res, ss.Result) and key not in self.results:
                self.results[key] = res
                self._bridged_keys.append(key)
        self.results['variant'] = variant
        self.results['date'] = self.results['timevec'] # v3 also had the time keys "date" and "t"
        self.results['t'] = np.arange(self.t.npts)
        self._bridged_keys += ['variant', 'date', 't']
        return

    def finalize_results(self):
        """Scale the results, skipping the references to the COVID results, which are scaled by the module"""
        bridged = {key:self.results.pop(key) for key in self._bridged_keys}
        try:
            super().finalize_results()
        finally: # Restore them even if finalizing fails (e.g. if the sim has already been finalized)
            self.results.update(bridged)
        return

    def save(self, filename=None, keep_people=None, shrink=None, **kwargs):
        """Save the sim to disk.

        By default the full sim is saved, including the people, so it can be rerun or continued.

        Args:
            filename (str): path to save to (defaults to the sim's ``simfile``).
            keep_people (bool): whether to keep the people (v3); ``keep_people=False`` is the same as ``shrink=True``
            shrink (bool): drop the people and other large objects before saving (default False).
            kwargs: passed through to ``ss.Sim.save`` / ``sc.makefilepath``.
        """
        if filename is None:
            filename = self.simfile
        if shrink is None:
            shrink = keep_people is False
        return super().save(filename=filename, shrink=shrink, **kwargs)

    @staticmethod
    def load(filename, *args, **kwargs):
        """Load a saved sim from disk; the v3 ``cv.Sim.load`` classmethod, via ``cv.load``."""
        return cvm.load(filename, *args, **kwargs)

    def plot(self, *args, **kwargs):
        '''
        Plot the results of a single simulation.

        Args:
            to_plot      (dict): Dict of results to plot; see get_default_plots() for structure
            do_save      (bool): Whether or not to save the figure
            fig_path     (str):  Path to save the figure
            fig_args     (dict): Dictionary of kwargs to be passed to ``pl.figure()``
            plot_args    (dict): Dictionary of kwargs to be passed to ``pl.plot()``
            scatter_args (dict): Dictionary of kwargs to be passed to ``pl.scatter()``
            axis_args    (dict): Dictionary of kwargs to be passed to ``pl.subplots_adjust()``
            legend_args  (dict): Dictionary of kwargs to be passed to ``pl.legend()``; if show_legend=False, do not show
            date_args    (dict): Control how the x-axis (dates) are shown (see below for explanation)
            show_args    (dict): Control which "extras" get shown: uncertainty bounds, data, interventions, ticks, the legend; additionally, "outer" will show the axes only on the outer plots
            style_args   (dict): Dictionary of kwargs to be passed to Matplotlib; options are dpi, font, fontsize, plus any valid key in ``pl.rcParams``
            n_cols       (int):  Number of columns of subpanels to use for subplot
            font_size    (int):  Size of the font
            font_family  (str):  Font face
            grid         (bool): Whether or not to plot gridlines
            commaticks   (bool): Plot y-axis with commas rather than scientific notation
            setylim      (bool): Reset the y limit to start at 0
            log_scale    (bool): Whether or not to plot the y-axis with a log scale; if a list, panels to show as log
            do_show      (bool): Whether or not to show the figure
            colors       (dict): Custom color for each result, must be a dictionary with one entry per result key in to_plot
            sep_figs     (bool): Whether to show separate figures for different results instead of subplots
            fig          (fig):  Handle of existing figure to plot into
            ax           (axes): Axes instance to plot into
            kwargs       (dict): Parsed among figure, plot, scatter, date, and other settings (will raise an error if not recognized)

        The optional dictionary "date_args" allows several settings for controlling
        how the x-axis of plots are shown, if this axis is dates. These options are:

            - ``as_dates``:   whether to format them as dates (else, format them as days since the start)
            - ``dateformat``: string format for the date (if not provided, choose based on timeframe)
            - ``rotation``:   whether to rotate labels
            - ``start``:      the first day to plot
            - ``end``:        the last day to plot
            - ``interval``:   the interval between x-axis ticks, in days

        The ``show_args`` dictionary allows several other formatting options, such as:

            - ``tight``:    use tight layout for the figure (default false)
            - ``maximize``: try to make the figure full screen (default false)
            - ``outer``:    only show outermost (bottom) date labels (default false)

        Date, show, and other arguments can also be passed directly, e.g. ``sim.plot(tight=True)``.

        For additional style options, see ``cv.options.with_style()``, which is the
        final refuge of arguments that are not picked up by any of the other parsers,
        e.g. ``sim.plot(**{'ytick.direction':'in'})``.

        Returns:
            fig: Figure handle

        **Examples**::

            sim = cv.Sim().run()
            sim.plot() # Default plotting
            sim.plot('overview') # Show overview
            sim.plot('overview', maximize=True, outer=True, rotation=15) # Make some modifications to make plots easier to see
            sim.plot(style='seaborn-whitegrid') # Use a built-in Matplotlib style
            sim.plot(style='simple', font='Rosario', dpi=200) # Use the other house style with several customizations

        | New in version 2.1.0: argument passing, date_args, and mpl_args
        | New in version 3.1.2: updated date arguments; mpl_args renamed style_args
        '''
        fig = cvplt.plot_sim(sim=self, *args, **kwargs)
        return fig

    def plot_result(self, key, *args, **kwargs):
        '''
        Simple method to plot a single result. Useful for results that aren't
        standard outputs. See sim.plot() for explanation of other arguments.

        Args:
            key (str): the key of the result to plot

        Returns:
            fig: Figure handle

        **Example**::

            sim = cv.Sim().run()
            sim.plot_result('r_eff')
        '''
        fig = cvplt.plot_result(sim=self, key=key, *args, **kwargs)
        return fig

    def to_excel(self, filename=None, skip_pars=None):
        """Export results + parameters to an Excel workbook (the v3 ``Sim.to_excel``).

        Writes a 'Results' sheet (the time series via ``to_df``) and a 'Parameters' sheet (the flattened
        Covasim config from ``export_pars``). Returns the ``sc.Spreadsheet``.
        """
        # Build the results sheet from the 1D covid result series only -- the nested 2D by-variant
        # sub-dict breaks a flat DataFrame (the same reason cv.Sim.plot avoids stock ss plotting).
        covid = list(self.diseases.values())[0]
        res = covid.results
        data = {k: np.asarray(res[k]) for k in res.keys()
                if isinstance(res[k], ss.Result) and np.ndim(np.asarray(res[k])) == 1}
        result_df = pd.DataFrame(data)
        result_df.insert(0, 'date', np.asarray(self.t.timevec))
        flat = sc.flattendict(self.export_pars(), sep='_')
        par_df = pd.DataFrame.from_dict({k: [v] for k, v in flat.items()}, orient='index', columns=['Value'])
        spreadsheet = sc.Spreadsheet()
        spreadsheet.freshbytes()
        with pd.ExcelWriter(spreadsheet.bytes, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, sheet_name='Results')
            par_df.to_excel(writer, sheet_name='Parameters')
        spreadsheet.load()
        if filename is not None:
            spreadsheet.save(filename)
        return spreadsheet

    def calibrate(self, calib_pars, **kwargs):
        """
        Automatically calibrate the simulation, returning a Calibration object. See the
        documentation on that class for more information.

        Args:
            calib_pars (dict): a dictionary of the parameters to calibrate of the format dict(key1=[best, low, high])
            kwargs (dict): passed to cv.Calibration()

        Returns:
            A Calibration object

        **Example**::

            sim = cv.Sim(datafile='data.csv')
            calib_pars = dict(beta=[0.015, 0.010, 0.020])
            calib = sim.calibrate(calib_pars, n_trials=50)
            calib.plot_sims()
        """
        calib = cva.Calibration(sim=self, calib_pars=calib_pars, **kwargs)
        calib.calibrate()
        return calib


def demo(preset=None, to_plot=None, scens=None, run_args=None, plot_args=None, **kwargs):
    """
    Shortcut for ``cv.Sim().run().plot()`` (the v3 ``cv.demo``).

    Args:
        preset (str): ignored; kept for backwards compatibility
        to_plot (str): what to plot
        scens (dict): ignored; kept for backwards compatibility
        run_args (dict): passed to sim.run()
        plot_args (dict): passed to sim.plot()
        kwargs (dict): passed to Sim()

    **Example**::

        cv.demo(beta=0.020, run_args={'verbose':0})
    """
    plot_args = sc.mergedicts(plot_args, {'to_plot':to_plot} if to_plot else None)
    sim = Sim(**kwargs)
    sim.run(**sc.mergedicts(run_args))
    sim.plot(**plot_args)
    return sim
