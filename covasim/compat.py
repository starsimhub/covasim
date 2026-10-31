"""
Backwards compatibility with Covasim v3.

Covasim v4 is built on Starsim, which names some things differently (e.g. a module's
``step()`` rather than v3's ``apply(sim)``). The shims here let v3 scripts run unchanged.
They are kept together in this file so the rest of the code can use the Starsim
conventions directly. Each one notes the v3 behavior it preserves.
"""

import functools as ft
import inspect
import numpy as np
import sciris as sc
import starsim as ss
from . import settings as cvset
from . import parameters as cvpar
from . import covid as cvcov
from . import network as cvnet
from . import base as cvb
from . import utils as cvu
from . import misc as cvm

__all__ = ['V3Module', 'V3Sim', 'V3People', 'V3MultiSim', 'V3Scenarios']


def _v3_finalize(func):
    """ Let a v3-style finalize(self, sim) be called without the sim, as Starsim does, and finish the Starsim finalization if it doesn't call super().finalize() """
    @ft.wraps(func)
    def wrapper(self, sim=None, *args, **kwargs):
        out = func(self, self.sim if sim is None else sim, *args, **kwargs)
        if not self.finalized: # v3: custom finalize(sim) methods didn't need to call super().finalize()
            V3Module.finalize(self)
        return out
    return wrapper


def _v3_step(self):
    """ The Starsim step() for a class that defines the v3 apply(self, sim) but not step() """
    return self.apply(self.sim)


def _v3_apply(step_func):
    """ The v3 apply(self, sim) for a class that defines step() but not apply(), e.g. the built-in interventions """
    @ft.wraps(step_func)
    def apply(self, sim=None):
        return step_func(self)
    return apply


class V3Module:
    """
    Mixin for Covasim interventions and analyzers that supports the v3 module methods.

    In v3, custom interventions and analyzers defined ``apply(self, sim)`` (called on
    each timestep), and optionally ``initialize(self, sim)`` and ``finalize(self, sim)``.
    In Starsim these are ``step()``, ``init_post()``, and ``finalize()``. A subclass can
    define either set.
    """

    def __init__(self, *args, do_plot=None, show_label=None, line_args=None, **kwargs):
        """ v3: accept the plotting kwargs, which are used by sim.plot() """
        super().__init__(*args, **kwargs)
        self.do_plot = do_plot
        self.show_label = show_label
        self.line_args = line_args
        return

    def __init_subclass__(cls, **kwargs):
        """
        v3: apply(self, sim) and finalize(self, sim). Starsim calls step() on each timestep, and finalize()
        with no arguments. So that a subclass can override either apply() or step() (e.g. a subclass of
        cv.test_prob that overrides apply() and calls super().apply(sim)), a class that defines only one
        of them gets the other.
        """
        super().__init_subclass__(**kwargs)
        has_apply = 'apply' in cls.__dict__
        has_step = 'step' in cls.__dict__
        if has_apply and not has_step:
            cls.step = _v3_step
        if has_step and not has_apply:
            cls.apply = _v3_apply(cls.__dict__['step'])
        func = cls.__dict__.get('finalize')
        if func is not None:
            if 'sim' in inspect.signature(func).parameters: # With or without a default, e.g. v3's own finalize(self, sim=None)
                cls.finalize = _v3_finalize(func)
        return

    @property
    def today(self):
        """ The day this step starts on (day 0 is the start day), which days given to interventions and analyzers (e.g. ``days=10``) refer to; the step's own index, ``self.ti``, is one more, since time point 0 is the start of the run """
        return self.ti - 1

    def init_post(self):
        """ v3: initialize(self, sim) is called once the sim is initialized """
        super().init_post()
        if type(self).initialize is not V3Module.initialize:
            self.initialize(self.sim)
        return

    def initialize(self, sim=None):
        """ v3: override to initialize the module once the sim is available (use init_post() in v4) """
        pass

    def step(self):
        """ v3: apply(self, sim) is called on each timestep """
        return self.apply(self.sim)

    def apply(self, sim):
        """ v3: override to apply the module on each timestep (use step() in v4) """
        pass

    def finalize(self, sim=None):
        """ v3: finalize(sim) accepts the sim (use finalize() in v4) """
        return super().finalize()



class V3Sim:
    """
    Mixin for cv.Sim that supports the v3 dict-style parameter access, e.g. ``sim['beta'] = 0.02``.

    v3 kept all parameters in one dict. In v4, the sim-level parameters are in ``sim.pars`` (with
    Starsim names, e.g. ``n_agents`` rather than ``pop_size``), and the disease parameters are in
    ``sim.diseases.covid.pars``. The v3 keys below are translated; any other key is looked up in the
    COVID parameters, and then as a sim attribute (the Starsim behavior, e.g. ``sim['diseases']``).
    """

    # Parameters that determine how the sim is built, so can only be changed before it is initialized
    _v3_init_keys = ['pop_size', 'pop_type', 'pop_infected', 'n_days', 'start_day', 'end_day', 'rand_seed', 'pop_scale',
                     'scaled_pop', 'rescale', 'use_waning', 'contacts', 'location', 'dur', 'clustered']

    def _is_v3_saved(self):
        """ Whether this is a sim saved by Covasim v3 and loaded with cv.load(): its results and v3 parameters can be read, but it can't be rerun """
        return 'pop_size' in self.__dict__.get('pars', {}) # In v4, the sim parameters use the Starsim names, e.g. n_agents

    def _v3_covid(self):
        """ The COVID module: the one in the sim after initialization, else the one in the pars """
        if self.initialized:
            return self.diseases.get('covid')
        covid = None
        for disease in sc.tolist(self.pars.diseases): # A single module, or a list
            if isinstance(disease, cvcov.COVID):
                covid = disease
        return covid

    def _v3_key_error(self, key):
        """ As in v3, the error for an unrecognized parameter (e.g. a typo) """
        covid = self._v3_covid()
        keys = list(cvpar.make_pars().keys()) + (list(covid.pars.keys()) if covid is not None else [])
        errormsg = f'Parameter "{key}" not recognized; the parameters are: {sc.strjoin(list(dict.fromkeys(keys)))}'
        return KeyError(errormsg)

    def _v3_networks(self):
        """ The networks, before or after initialization """
        return list(self.networks.values()) if self.initialized else sc.tolist(self.pars.networks)

    def __getitem__(self, key):
        """ v3: sim['key'] returns the parameter """
        if self._is_v3_saved(): # The v3 parameters are all in sim.pars
            return self.pars[key]
        covid = self._v3_covid()
        if key == 'pop_size':
            return self.pars.n_agents
        elif key == 'pop_type':
            return self.pop_type
        elif key == 'pop_infected': # The number of initial infections, from ss.choose_n()
            return covid.pars.init_prev.pars.n
        elif key == 'n_days':
            return int(self.pars.dur.value)
        elif key == 'start_day': # As in v3, a date once the sim is initialized, else a string
            return sc.date(self.pars.start, as_date=self.initialized)
        elif key == 'end_day':
            return sc.datedelta(self['start_day'], days=self['n_days'], as_date=False) # A string, as in v3
        elif key == 'rand_seed':
            return self.pars.rand_seed
        elif key == 'pop_scale': # Before initialization, Starsim may not have calculated this yet
            if self.pars.pop_scale is not None:
                return self.pars.pop_scale
            else:
                return self.pars.total_pop/self.pars.n_agents if self.pars.total_pop else 1.0
        elif key in ['verbose', 'rescale', 'rescale_threshold', 'rescale_factor']:
            return self.pars[key]
        elif key == 'beta': # Stored as a rate; v3 used the daily probability
            beta = covid.pars.beta
            return beta if isinstance(beta, dict) else ss.probperday(beta).value
        elif key == 'contacts': # Before initialization, the dict used to make the default networks, so it can be changed in place
            if (not self.initialized) and (self._v3_contacts is not None):
                return self._v3_contacts
            return cvnet.get_contacts(self._v3_networks())
        elif key in self._v3_pars: # location, scaled_pop, variants, and dur
            return self._v3_pars[key]
        elif key == 'n_variants':
            return covid.nv
        elif key == 'immunity': # v3: the matrix of immunity and cross-immunity factors, calculated when the sim is initialized (None if there is no cross-immunity)
            connector = self.connectors.get('crossimmunity') if self.initialized else None
            return connector.matrix if connector is not None else None
        elif key == 'nab_kin': # v3: calculated from the NAb parameters when the sim was initialized; not used in v4
            return None
        elif key in ['variant_map', 'variant_pars', 'vaccine_map', 'vaccine_pars']:
            return getattr(covid, key)
        elif key == 'prognoses': # Store the defaults when first accessed, so that changing them in place works, as in v3
            if covid.pars.prognoses is None:
                covid.pars.prognoses = cvpar.get_prognoses(by_age=covid.pars.prog_by_age)
            return covid.pars.prognoses
        elif key in ['interventions', 'analyzers']:
            return list(getattr(self, key).values()) if self.initialized else sc.tolist(self.pars[key])
        elif (covid is not None) and (key in covid.pars):
            return covid.pars[key]
        elif hasattr(self, key): # A Starsim attribute, e.g. sim['diseases']
            return getattr(self, key)
        else:
            raise self._v3_key_error(key)

    def __setitem__(self, key, value):
        """ v3: sim['key'] = value sets the parameter """
        covid = self._v3_covid()
        if key in self._v3_init_keys and self.initialized:
            errormsg = f'Cannot set "{key}" after the sim has been initialized; please set it when creating the sim'
            raise RuntimeError(errormsg)

        if key == 'pop_size':
            self.pars.n_agents = value
        elif key == 'pop_infected':
            covid.pars.init_prev = ss.choose_n(int(value))
        elif key == 'n_days':
            self.pars.dur = ss.days(value)
        elif key == 'start_day': # The duration (n_days) is unchanged
            self.pars.start = ss.date(value)
        elif key == 'end_day':
            self.pars.dur = ss.days(sc.daydiff(self['start_day'], value))
        elif key == 'rand_seed':
            self.pars.rand_seed = value
        elif key == 'pop_scale': # Starsim calculates the total population from this, so only one of them can be set
            self.pars.pop_scale = value
            self.pars.total_pop = None
            self._v3_pars['scaled_pop'] = None
        elif key == 'scaled_pop': # Likewise
            self.pars.total_pop = value
            self.pars.pop_scale = None
            self._v3_pars['scaled_pop'] = value
        elif key == 'dur': # Converted to the COVID duration parameters (e.g. dur_exp2inf) when the sim is initialized
            self._v3_pars['dur'] = value
        elif key in ['verbose', 'rescale', 'rescale_threshold', 'rescale_factor']:
            self.pars[key] = value
        elif key in ['interventions', 'analyzers']:
            if not self.initialized:
                self.pars[key] = value
            else: # v3: they can be added to a sim that has been run part-way, e.g. sim['interventions'] += [cv.test_prob(...)]
                value = sc.tolist(value)
                current = [id(mod) for mod in self[key]]
                new = [mod for mod in value if id(mod) not in current]
                if len(value) - len(new) != len(current):
                    errormsg = f'Cannot remove or replace {key} after the sim has been initialized; they can only be added'
                    raise RuntimeError(errormsg)
                for mod in new:
                    self.add_module(mod, key)
        elif key == 'beta':
            covid.pars.update(beta=value) # A number (or an ss.par to fit) stays a daily probability
            if self.initialized:
                self.__dict__.setdefault('_v3_set_pars', {})[key] = value # Set again by initialize(reset=True), since sim.new() doesn't keep changes made to an initialized sim
        elif key in ['pop_type', 'contacts', 'location', 'clustered']: # Recreate the networks (and for pop_type, the per-layer parameters)
            pop_type = self.pop_type
            contacts = self._v3_contacts # For location, keep the current contacts, apart from the household size
            if key == 'pop_type':
                pop_type = value
                contacts = None # The defaults for this pop_type
                layer_pars = dict(pop_type=pop_type)
                cvpar.reset_layer_pars(layer_pars)
                covid.pars.update({k:layer_pars[k] for k in ['beta_layer', 'iso_factor', 'quar_factor']})
            elif key == 'contacts':
                contacts = value
                if sc.isnumber(contacts): # A single value for all layers
                    contacts = {lkey:contacts for lkey in self.layer_keys()}
            elif key == 'clustered':
                self.clustered = value
            else: # Location: reload the age distribution and household size
                self._v3_pars['location'] = value
                self._load_location(value, self.pars.verbose)
            self.pop_type = pop_type
            self._v3_contacts = self._make_contacts(pop_type, contacts, self.pars.verbose)
            self.pars.networks = cvnet.make_networks(pop_type, contacts=self._v3_contacts, clustered=self.clustered)
        elif key in ['use_waning', 'variants', 'n_variants', 'variant_map', 'variant_pars']:
            errormsg = f'Cannot set "{key}" on an existing sim; please set it when creating the sim'
            raise ValueError(errormsg)
        elif (covid is not None) and (key in covid.pars):
            covid.pars[key] = value
            if self.initialized:
                self.__dict__.setdefault('_v3_set_pars', {})[key] = value # As for beta
        elif hasattr(self, key): # A Starsim attribute
            setattr(self, key, value)
        else:
            raise self._v3_key_error(key)
        return

    # v3 properties and methods

    @property
    def rescale_vec(self):
        """ v3: the number of people each agent represents on each timestep, which changes over time with dynamic rescaling """
        if self._is_v3_saved():
            return self.__dict__['rescale_vec']
        if self.pars.rescale:
            return self.results.pop_scale.values
        return np.full(self.t.npts, self.pars.pop_scale)

    @property
    def npts(self):
        """ v3: the number of timepoints """
        if self.initialized and not self._is_v3_saved():
            return self.t.npts
        return self['n_days'] + 1

    @property
    def tvec(self):
        """ v3: the vector of day indices """
        return np.arange(self.npts)

    @property
    def datevec(self):
        """ v3: the vector of dates """
        return np.array(sc.daterange(self['start_day'], self['end_day'], as_date=True))

    @property
    def n(self):
        """ v3: the number of agents (as with len(sim.people), those who are alive) """
        return len(self.people) if self.initialized else self['pop_size']

    @property
    def scaled_pop_size(self):
        """ v3: the number of agents times the population scale factor """
        return self['pop_size']*self['pop_scale']

    def layer_keys(self):
        """ v3: the keys of the contact layers (i.e., networks) """
        if self._is_v3_saved():
            return list(self['beta_layer'].keys())
        if self.initialized:
            return list(self.networks.keys())
        return list(cvnet.get_contacts(self._v3_networks()).keys())

    @property
    def label(self):
        """ The sim label; a sim saved by v3 stores it as an attribute, rather than in the parameters as in Starsim """
        if self._is_v3_saved():
            return self.__dict__.get('label')
        return ss.Sim.label.fget(self)

    @label.setter
    def label(self, label):
        ss.Sim.label.fset(self, label)

    def init_data(self, data=None, check=True):
        """ v3: keep the data as loaded by cv.load_data() (indexed by date, with a "date" column), rather than converting it to the Starsim format """
        if data is not None:
            self.data = cvm.load_data(data)
        return

    def reset_layer_pars(self, layer_keys=None, force=False):
        """ v3: set the per-layer parameters (beta_layer, iso_factor, and quar_factor) to their defaults, for any layers that don't have them """
        covid = self._v3_covid()
        keys = ['beta_layer', 'iso_factor', 'quar_factor']
        pars = dict(pop_type=self.pop_type, **{key:covid.pars[key] for key in keys})
        cvpar.reset_layer_pars(pars, layer_keys=layer_keys or self.layer_keys(), force=force)
        covid.pars.update({key:pars[key] for key in keys})
        return

    def _add_network(self, network):
        """ Add a network to a sim that has been initialized but not yet run, e.g. via sim.people.contacts.add_layer() """
        if self.ti > 0:
            errormsg = f'Cannot add network "{network.name}" since the sim has already started running'
            raise RuntimeError(errormsg)
        self.add_module(network, 'networks')
        self.reset_layer_pars() # Use the default beta_layer etc. for this network, unless already set
        return

    def result_keys(self, which='main'):
        """ v3: the keys of the COVID results; which can be 'main', 'variant', or 'all' """
        choices = ['main', 'variant', 'all']
        if which not in choices:
            errormsg = f'Choice "{which}" not available; choices are: {sc.strjoin(choices)}'
            raise ValueError(errormsg)
        if not self.initialized: # There are no results yet
            return []
        covid = self.diseases.covid
        stocks = {state.name for state in covid.auto_state_list} # v3 named the counts of states n_<state>, e.g. n_symptomatic (see init_results())
        keys = []
        for key,res in covid.results.items():
            is_variant = key.endswith('_by_variant')
            is_v3 = key != 'infection' # Starsim's name for new_infections
            if isinstance(res, ss.Result) and is_v3 and (which == 'all' or is_variant == (which == 'variant')):
                keys.append(f'n_{key}' if key in stocks else key)
        return keys

    def update_pars(self, pars=None, **kwargs):
        """ v3: update the parameters from a dict and/or keyword arguments """
        for key,val in sc.mergedicts(pars, kwargs).items():
            self[key] = val
        return

    def set_seed(self, seed=-1):
        """
        v3: set the seed from the stored or supplied value; this seeds the global random number generators (e.g. used by
        cv.choose()), as in v3, while Starsim's are seeded from rand_seed. Once the sim is initialized, rand_seed can't be
        changed, so a supplied seed only seeds the global ones (e.g. v3 code that reset the seed on each timestep).
        """
        if seed == -1:
            seed = self['rand_seed']
        elif not self.initialized:
            self['rand_seed'] = seed
        cvu.set_seed(seed)
        return

    def step(self):
        """ v3: run a single timestep (use run_one_step() in v4) """
        return self.run_one_step()

    def run(self, do_plot=False, until=None, restore_pars=None, reset_seed=None, verbose=None, **kwargs):
        """
        v3: run the sim, optionally plotting, and stopping early if the timelimit or stopping_func is reached.
        As in v3, ``until`` is the day to run until (not including that day), and if ``restore_pars`` is true
        (default), COVID parameters changed by interventions are restored when the sim is finalized. reset_seed
        is ignored, since v4 uses separate random number streams.
        """
        if not self.initialized:
            self.init()
        if self.ti == 0: # v3: store the parameters before the run, to restore afterwards
            restore_pars = True if restore_pars is None else restore_pars
            self._orig_pars = None
            if restore_pars:
                self._orig_pars = {}
                for key,val in self.diseases.covid.pars.items():
                    if isinstance(val, (dict, list, np.ndarray)): # Containers can be changed in place, e.g. beta_layer by cv.change_beta(layers=...), so copy them
                        val = sc.dcp(val)
                    self._orig_pars[key] = val # Other values, e.g. distributions, are replaced rather than modified by interventions, so keep a reference

        # v3: verbose=1 printed a line per day, and verbose=2 a heading per day; Starsim prints a heading per day for verbose >= 1
        if verbose is None:
            verbose = self.verbose
        if (verbose >= 1) and (verbose < 2):
            verbose = 0.99 # Starsim's progress line for every step
        if verbose == self.verbose: # Only pass it if it's changed, since Starsim restores the sim's own value when the run finishes
            verbose = None

        # Convert v3 "until" (the next timestep to run) to Starsim's (the last date to run)
        if until is not None:
            until = self.day(until)
            if until > self.npts:
                errormsg = f'Requested to run until t={until} but the simulation end is t={self.npts}'
                raise ss.AlreadyRunError(errormsg)
            if self.ti >= until or self.complete:
                errormsg = f'Simulation is currently at t={self.ti}, requested to run until t={until} which has already been reached'
                raise ss.AlreadyRunError(errormsg)

        timelimit = getattr(self, 'timelimit', None)
        stopping_func = getattr(self, 'stopping_func', None)
        if timelimit is None and stopping_func is None:
            super().run(until=None if until is None else self.t.timevec[until-1], verbose=verbose, **kwargs)
        else: # Run one step at a time to check whether to stop; as in v3, the sim isn't finalized if it stops early
            T = sc.timer()
            until = self.npts if until is None else until
            while self.ti < until and not self.complete:
                super().run(until=self.t.timevec[self.ti], verbose=verbose, **kwargs)
                verbose = None # Already set by the first step
                if timelimit is not None and T.toc(output=True) > timelimit:
                    sc.printv(f'Time limit ({timelimit} s) exceeded; call sim.finalize() to compute results if desired', 1, self.verbose)
                    break
                if stopping_func is not None and stopping_func(self):
                    sc.printv('Stopping function terminated the simulation; call sim.finalize() to compute results if desired', 1, self.verbose)
                    break
        if do_plot:
            self.plot()
        return self

    def finalize(self):
        """ v3: restore the COVID parameters changed by interventions, if run(restore_pars=True), and print the summary if verbose """
        verbose = self.verbose # The value for this run, which Starsim's finalize() resets to the sim's own value
        super().finalize()
        orig_pars = getattr(self, '_orig_pars', None)
        if orig_pars is not None:
            for key,val in orig_pars.items():
                self.diseases.covid.pars[key] = val
            self._orig_pars = None
        if verbose > 0: # As in v3, print the summary
            self.summarize(full=False)
        return

    def initialize(self, *args, reset=False, **kwargs):
        """
        v3: initialize the sim (use init() in v4). If already initialized, reset=True recreates it as it was first
        initialized, with any parameters set with sim['key'] since then set again (e.g. to rerun it with a new value).
        """
        if self.initialized:
            if not reset:
                return self
            set_pars = self.__dict__.get('_v3_set_pars') or {} # The COVID parameters set with sim['key'] since it was initialized
            self.new(inplace=True) # Made again from what it was made with (see sim.new())
            self._v3_covid().pars.update(set_pars) # Kept, as in v3
        return self.init(*args, **kwargs)

    def shrink(self, skip_attrs=None, in_place=True, inplace=None, **kwargs):
        """ v3: in_place rather than inplace; skip_attrs is ignored """
        kwargs.setdefault('die', False) # v3 never raised an error on shrinking
        kwargs.setdefault('base_size', 150) # Starsim's default allows 30 KB per module plus 1 KB per timestep, but the COVID module has about 40 KB of parameters and distributions, and its results by variant can be more than 1 KB per timestep
        return super().shrink(inplace=in_place if inplace is None else inplace, **kwargs)

    def summarize(self, full=None, t=None, sep=None, output=None, how=None):
        """
        Compute the summary of the results, or with any of the v3 arguments, print it.

        With no arguments (as Starsim calls it, e.g. at the end of the run), compute, store, and return the
        summary, as compute_summary(). With any of the v3 arguments, print a medium-length summary of the sim
        (as v3 did; this is also printed at the end of a run if verbose > 0), or return it as a string.

        Args:
            full   (bool):    whether to print all results (by default, only cumulative)
            t      (int/str): day or date to summarize (by default, the last day)
            sep    (str):     thousands separator (default ',')
            output (bool):    whether to return the summary as a string instead of printing it
            how    (str):     ignored; the summary is always the value on day t (for Starsim compatibility)

        **Examples**::

            sim = cv.Sim(verbose=0).run()
            sim.summarize() # Return the summary dict
            sim.summarize(full=True) # Print all the results
            sim.summarize(t=24, output=True) # Return the summary for day 24 as a string
        """
        v3_args = [full, t, sep, output]
        if all(arg is None for arg in v3_args):
            return self.compute_summary(output=True)
        summary = self.compute_summary(t=t, update=False, output=True)
        if sep is None:
            sep = cvset.options.sep
        labelstr = f' "{self.label}"' if self.label else ''
        string = f'Simulation{labelstr} summary:\n'
        for key in self.result_keys():
            if full or key.startswith('cum_'):
                val = np.round(summary[key])
                string += f'   {val:10,.0f} {self.results[key].label.lower()}\n'.replace(',', sep)
        if output:
            return string
        print(string)
        return

    def compute_summary(self, full=None, t=None, update=True, output=False, require_run=False):
        """
        v3: compute the summary, the value of each result on day t (by default, the last day), with the COVID
        results named as in v3 (e.g. "cum_infections"); see sim.summarize() to print it.

        Args:
            full        (bool):    ignored, as in v3
            t           (int/str): day or date to summarize (by default, the last day)
            update      (bool):    whether to store it in sim.summary
            output      (bool):    whether to return it
            require_run (bool):    whether to raise an exception if the sim hasn't been run
        """
        if require_run and not self.results_ready:
            errormsg = 'Simulation not yet run'
            raise RuntimeError(errormsg)
        t = -1 if t is None else self.day(t)
        summary = sc.objdict()
        flat = self.results.flatten(sep='_', only_results=False, keep_case=True, columns=True)
        skip = ('covid_', 'variant_') # Skip the duplicates of the COVID results (e.g. "covid_cum_infections", the same as "cum_infections"), and the results by variant, which aren't in the v3 summary
        for key,res in flat.items():
            if not key.startswith(skip) and key not in ['alive', 'female', 'infection', 'died', 'emigrated', 'unrepresented']: # The same as n_alive, n_female, new_infections, new_deaths, and new_emigrants, the v3 names; unrepresented is v4's own
                val = res[t]
                if sc.isnumber(val): # Skip the dates
                    summary[key] = val
        if update:
            self.summary = summary
        if output:
            return summary
        return

    def export_results(self, for_json=True, filename=None, indent=2, **kwargs):
        """
        v3: convert the COVID results to a dict, optionally saving to JSON (kwargs are passed to sc.savejson()).
        With for_json, as in v3, the dict also has the list of result keys ("timeseries_keys"), and the results
        by variant (under "variant").
        """
        results = self.results
        output = {}
        output['t'] = self.tvec
        if for_json:
            output['timeseries_keys'] = self.result_keys()
        for key in self.result_keys():
            output[key] = results[key].values
        if for_json:
            output['variant'] = {key:results['variant'][key].values for key in self.result_keys('variant')}
            output['date'] = [str(d) for d in self.datevec]
            output = sc.jsonify(output) # Convert the arrays to lists
        if filename is not None:
            sc.savejson(filename=filename, obj=output, indent=indent, **kwargs)
        return output

    def to_df(self, date_index=False, **kwargs):
        """ v3: the results as a dataframe, including the columns "t" and "date", and the COVID results named as in v3 (e.g. "cum_infections") """
        df = super().to_df(**kwargs)
        cols = list(df.columns)
        dups = ['timevec', 'infection', 'covid_infection', 'died', 'emigrated'] # The same as "date", "new_infections", "new_deaths", and "new_emigrants"
        for col in cols: # Remove duplicates of the COVID results, e.g. "covid_cum_infections" (same as "cum_infections")
            if col.startswith('variant_'):
                dups.append(col)
            elif col.startswith('covid_') and (col.removeprefix('covid_') in cols):
                dups.append(col)
        df = df.drop(columns=dups, errors='ignore')
        df = df.rename(columns={col:col.removeprefix('covid_') for col in df.columns}) # The results by variant only have the prefixed name, e.g. "covid_cum_infections_by_variant_wild"
        df.insert(0, 't', self.tvec)
        df.insert(1, 'date', self.datevec)
        if date_index:
            df = df.set_index('date')
        return df

    def to_json(self, filename=None, keys=None, verbose=False, indent=2, **kwargs):
        """
        v3: the parameters are stored as "parameters" rather than "pars", and are the v3 parameters from
        sim.export_pars(); the summary is the v3 one (see compute_summary()); the results are as in Starsim
        """
        keys = sc.tolist(keys) if keys is not None else ['pars', 'summary', 'results']
        out = sc.objdict()
        if ('pars' in keys) or ('parameters' in keys):
            out['parameters'] = self.export_pars()
        if 'summary' in keys:
            out['summary'] = dict(self.summary) if self.results_ready else 'Summary not available (Sim has not yet been run)'
        others = [key for key in keys if key not in ['pars', 'parameters', 'summary']]
        if others:
            out.update(super().to_json(keys=others, verbose=verbose))
        out = sc.jsonify(out, **kwargs)
        if filename is not None:
            sc.savejson(filename=filename, obj=out, indent=indent)
        return out


class V3People:
    """
    Mixin for cv.People that supports the v3 per-agent attributes and methods.

    In v3, the disease states (e.g. ``people.exposed``) were stored on People; in v4 they are on
    the COVID module (``sim.diseases.covid.exposed``), so they are looked up there. v3 dates (e.g.
    ``people.date_infectious``) are the v4 time indices (``ti_infectious``), since the timestep
    is one day.

    As in Starsim, these per-agent arrays only include agents who are alive (unlike v3, where
    agents who died stayed in the arrays). The exceptions are ``people.dead`` and ``people.date_dead``,
    which cover every agent ever created, so the index is the UID: e.g. ``people.dead.sum()`` is the
    number of deaths, and ``people.age[cv.true(people.dead)]`` is the ages of those who died (indexing
    by UID works for any agent). For other arrays, ``.raw`` has the values for every agent, e.g.
    ``people.age.raw[:sim.people.n_uids]``.
    """

    # v3: the per-agent durations (e.g. people.dur_exp2inf) are the differences between these dates (NaN if not reached); see also dur_disease
    _v3_durs = dict(
        dur_exp2inf  = ('ti_exposed',     'ti_infectious'),
        dur_inf2sym  = ('ti_infectious',  'ti_symptomatic'),
        dur_sym2sev  = ('ti_symptomatic', 'ti_severe'),
        dur_sev2crit = ('ti_severe',      'ti_critical'),
    )

    def _v3_covid(self):
        """ The COVID module, or None if the sim is not initialized """
        sim = self.__dict__.get('sim') # __dict__ access avoids re-triggering __getattr__
        diseases = getattr(sim, 'diseases', None) if sim is not None else None
        return diseases.get('covid') if diseases is not None else None

    def _v3_all(self, state):
        """ A view of the state over all agents ever created, alive or dead, rather than only those alive (as ss.Filter does for a subset); used for people.date_dead, as Starsim does for people.dead """
        if not self.initialized: # Before initialization, there are no agents yet
            return state
        view = object.__new__(state.__class__)
        view.__dict__ = state.__dict__.copy() # Shares the values, so setting them changes the state
        view.people = sc.objdict(auids=self.uid.raw[:self.n_uids].view(ss.uids)) # Every agent counts as active
        return view

    def __getattr__(self, key):
        """ v3: states such as people.exposed and people.date_exposed; only called if normal attribute lookup fails """
        covid = self._v3_covid()
        if covid is not None and not key.startswith('_'):
            if key in self._v3_durs: # e.g. dur_exp2inf = date_infectious - date_exposed
                start, end = self._v3_durs[key]
                return getattr(covid, end) - getattr(covid, start)
            if key.startswith('date_') and hasattr(covid, 'ti_' + key[5:]): # e.g. date_exposed -> ti_exposed
                state = getattr(covid, 'ti_' + key[5:])
                return self._v3_all(state) if key == 'date_dead' else state # Only agents who have died have a date of death
            state = getattr(covid, key, None)
            if isinstance(state, ss.Arr):
                return state
        errormsg = f"'{self.__class__.__name__}' object has no attribute '{key}'"
        raise AttributeError(errormsg)

    @property
    def sex(self):
        """ v3: sex as an integer array, 0 for female and 1 for male """
        return self.female.asnew((~self.female).values.astype(int))

    @property
    def dur_disease(self):
        """ v3: how long each agent had (or will have) COVID, from exposure to recovery or death """
        covid = self._v3_covid()
        end = covid.ti_recovered.asnew(np.fmin(covid.ti_recovered.values, covid.ti_dead.values)) # Whichever happens
        return end - covid.ti_exposed

    @property
    def flows(self):
        """
        v3: the number of new infections, deaths, tests, etc. on the current timestep, e.g. people.flows['new_infections'];
        between timesteps (e.g. after sim.run(until=...)), those of the timestep just run
        """
        covid = self._v3_covid()
        loop = self.sim.loop
        ti = 0
        if loop.index > 0: # The timestep of the last function run, i.e. the current one, or the one just run if between timesteps
            ti = loop.plan[loop.index-1].ti
        flows = {f'new_{key}':val for key,val in covid.flows.items()}
        flows.update({f'new_{key}':val for key,val in covid.test_flows.items()})
        flows.update({f'new_{key}':val for key,val in covid.vacc_flows.items()})
        flows['new_infections'] = covid.flows_by_variant['new_infections'].sum()
        flows['new_infectious'] = covid.flows_by_variant['new_infectious'].sum()
        flows['new_quarantined'] = np.count_nonzero(covid.date_quarantined == ti)
        return flows

    def infect(self, inds, hosp_max=None, icu_max=None, source=None, layer=None, variant=0):
        """
        v3: infect the specified people (those who are susceptible) and determine their outcomes; see
        cv.COVID.set_prognoses(). hosp_max and icu_max are ignored, since bed capacity is checked by
        the COVID module. Returns the UIDs of the people infected.
        """
        covid = self._v3_covid()
        uids, unique = np.unique(inds, return_index=True) # Remove duplicates
        uids = ss.uids(uids)
        keep = covid.susceptible[uids] # Keep only susceptibles
        uids = uids[keep]
        if source is not None:
            source = ss.uids(np.asarray(source)[unique][keep])
        if len(uids):
            covid.set_prognoses(uids, sources=source, variant=int(variant))
            if layer is not None:
                covid.infection_log.add_data(uids, network=layer)
        return uids

    def true(self, key):
        """ v3: the UIDs of people for whom this state is true """
        return self[key].true()

    def false(self, key):
        """ v3: the UIDs of people for whom this state is false """
        return self[key].false()

    def defined(self, key):
        """ v3: the UIDs of people for whom this state is not NaN """
        return self[key].notnan.true()

    def undefined(self, key):
        """ v3: the UIDs of people for whom this state is NaN """
        return self[key].isnan.true()

    def count(self, key):
        """ v3: the number of people for whom this state is true """
        return np.count_nonzero(self[key])

    def count_not(self, key):
        """ v3: the number of people for whom this state is false """
        return len(self[key]) - self.count(key)

    def layer_keys(self):
        """ v3: the keys of the contact layers (i.e., networks) """
        return list(self.sim.networks.keys())

    @property
    def contacts(self):
        """ v3: the contact layers, i.e. the sim's networks, e.g. people.contacts['h']; see cv.Contacts """
        return cvb.Contacts.from_sim(self.sim)

    @staticmethod
    def remove_duplicates(df):
        """ v3: sort a dataframe of contacts (e.g. from layer.to_df()) and remove duplicates and self-connections """
        p1 = df[['p1', 'p2']].values.min(1) # Reassign p1 to be the lower-valued of the two contacts
        p2 = df[['p1', 'p2']].values.max(1) # Reassign p2 to be the higher-valued of the two contacts
        df['p1'] = p1
        df['p2'] = p2
        df.sort_values(['p1', 'p2'], inplace=True) # Sort by p1, then by p2
        df.drop_duplicates(['p1', 'p2'], inplace=True) # Remove duplicates
        df = df[df['p1'] != df['p2']] # Remove self connections
        df.reset_index(inplace=True, drop=True)
        return df

    def keys(self):
        """ v3: the names of all the per-agent states, including the COVID ones """
        covid = self._v3_covid()
        covid_keys = [state.name for state in covid.state_list] if covid is not None else []
        return list(self.states.keys()) + covid_keys

    def state_keys(self):
        """ v3: the names of the COVID true/false states """
        return [state.name for state in self._v3_covid().state_list if isinstance(state, ss.BoolArr)]

    def date_keys(self):
        """ v3: the names of the COVID dates, e.g. date_exposed """
        return ['date_' + state.name[3:] if state.name.startswith('ti_') else state.name for state in self._v3_covid().state_list if state.name.startswith(('ti_', 'date_'))]

    @property
    def infection_log(self):
        """ v3: the transmission log, a list of dicts with keys source, target, date, layer, and variant """
        from . import analysis as cva # Here to avoid a circular import
        return cva.make_infection_log(self.sim)

    def test(self, inds, test_sensitivity=1.0, loss_prob=0.0, test_delay=0):
        """ v3: test the specified people; see cv.COVID.test() """
        return self._v3_covid().test(inds, test_sensitivity=test_sensitivity, loss_prob=loss_prob, test_delay=test_delay)

    def schedule_quarantine(self, inds, start_date=None, period=None):
        """ v3: schedule a quarantine for the specified people; see cv.COVID.schedule_quarantine() """
        return self._v3_covid().schedule_quarantine(inds, start_date=start_date, period=period)

    def indices(self):
        """ v3: the indices (UIDs) of the agents """
        return self.auids

    def story(self, uid, *args):
        """
        v3: print a short history of events in the life of the specified individual(s).

        **Example**::

            sim = cv.Sim(pop_type='hybrid', verbose=0).run()
            sim.people.story(12)
        """
        layer_labels = dict(a='default contact', h='household', s='school', w='workplace', c='community')
        dates = {
            'date_critical'       : 'became critically ill and needed ICU care',
            'date_dead'           : 'died ☹',
            'date_diagnosed'      : 'was diagnosed with COVID',
            'date_end_quarantine' : 'ended quarantine',
            'date_infectious'     : 'became infectious',
            'date_pos_test'       : 'received their positive test result',
            'date_quarantined'    : 'entered quarantine',
            'date_recovered'      : 'recovered',
            'date_severe'         : 'developed severe symptoms and needed hospitalization',
            'date_symptomatic'    : 'became symptomatic',
            'date_tested'         : 'was tested for COVID',
            'date_vaccinated'     : 'was vaccinated against COVID',
        }
        infection_log = self.infection_log
        n_secondary = {} # The number of people each person infected
        for infection in infection_log:
            source = infection['source']
            n_secondary[source] = n_secondary.get(source, 0) + 1

        for uid in sc.tolist(uid) + list(args):
            sex = 'female' if self.female[uid] else 'male'
            intro = f'\nThis is the story of {uid}, a {self.age[uid]:.0f} year old {sex}'
            if not self.susceptible[uid] or self.recovered[uid]:
                print(f'{intro}, who had {"asymptomatic" if np.isnan(self.date_symptomatic[uid]) else "symptomatic"} COVID.')
            else:
                print(f'{intro}, who did not contract COVID.')

            total_contacts = 0
            no_contacts = []
            for lkey, net in self.sim.networks.items():
                llabel = layer_labels.get(lkey, f'"{lkey}"')
                n_contacts = np.count_nonzero(net.edges.p1 == uid) + np.count_nonzero(net.edges.p2 == uid)
                total_contacts += n_contacts
                if n_contacts:
                    print(f'{uid} is connected to {n_contacts} people in the {llabel} layer')
                else:
                    no_contacts.append(llabel)
            if len(no_contacts):
                print(f'{uid} has no contacts in the {", ".join(no_contacts)} layer(s)')
            print(f'{uid} has {total_contacts} contacts in total')

            events = []
            for attr, message in dates.items():
                date = getattr(self, attr)[uid]
                if not np.isnan(date):
                    events.append((date, message))
            for infection in infection_log: # As in v3, who infected them, and whom they infected
                lkey = infection['layer']
                llabel = layer_labels.get(lkey, f'"{lkey}"')
                if infection['target'] == uid:
                    if lkey:
                        events.append((infection['date'], f'was infected with COVID by {infection["source"]} via the {llabel} layer'))
                    else:
                        events.append((infection['date'], 'was infected with COVID as a seed infection'))
                if infection['source'] == uid:
                    target = infection['target']
                    events.append((infection['date'], f'gave COVID to {target} via the {llabel} layer ({n_secondary.get(target, 0)} secondary infections)'))
            if len(events):
                for day, event in sorted(events, key=lambda x: x[0]):
                    print(f'On day {day:.0f}, {uid} {event}')
            else:
                print(f'Nothing happened to {uid} during the simulation.')
        return


class V3MultiSim:
    """
    Mixin for cv.MultiSim that supports the v3 display and export methods, which differ from
    the Starsim ones (e.g. v3's ``msim.summarize()`` printed a description, whereas Starsim's
    computes summary statistics).
    """

    def result_keys(self):
        """ v3: the result keys of the base sim """
        try:
            keys = self.base_sim.result_keys()
        except Exception as E:
            errormsg = f'Could not retrieve result keys since base sim not accessible: {str(E)}'
            raise ValueError(errormsg)
        return keys

    def summarize(self, output=False):
        """ v3: print a moderate length description of the MultiSim (use show() in v4) """
        return self.show(output=output)

    def brief(self, output=False):
        """ v3: print (or return) a one-line description of the MultiSim """
        string = repr(self)
        if not output:
            print(string)
        else:
            return string

    def disp(self, output=False):
        """ v3: print (or return) a detailed description of the MultiSim """
        string = sc.prepr(self)
        if not output:
            print(string)
        else:
            return string

    def to_json(self, *args, **kwargs):
        """ v3: shortcut for base_sim.to_json(); only available after reduce() or combine() """
        if self.which is None:
            errormsg = 'JSON export only available for reduced sim; please run msim.mean() or msim.median() first'
            raise RuntimeError(errormsg)
        return self.base_sim.to_json(*args, **kwargs)

    def to_excel(self, *args, **kwargs):
        """ v3: shortcut for base_sim.to_excel(); only available after reduce() or combine() """
        if self.which is None:
            errormsg = 'Excel export only available for reduced sim; please run msim.mean() or msim.median() first'
            raise RuntimeError(errormsg)
        return self.base_sim.to_excel(*args, **kwargs)


class V3Scenarios:
    """ Mixin for cv.Scenarios that supports the v3 dict-style access to the metaparameters, e.g. ``scens['n_runs']`` """

    def __getitem__(self, key):
        """ v3: scens['key'] returns the metaparameter """
        return self.pars[key]

    def __setitem__(self, key, value):
        """ v3: scens['key'] = value sets the metaparameter """
        self.pars[key] = value
        return

    def update_pars(self, pars=None, **kwargs):
        """ v3: update the metaparameters from a dict and/or keyword arguments """
        self.pars.update(sc.mergedicts(pars, kwargs))
        return
