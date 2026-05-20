'''
Defines the Sim class, Covasim's core class.
'''

#%% Imports
import numpy as np
import pandas as pd
import sciris as sc
from . import utils as cvu
from . import misc as cvm
from . import base as cvb
from . import defaults as cvd
from . import parameters as cvpar
from . import population as cvpop
from . import plotting as cvplt
from . import interventions as cvi
from . import immunity as cvimm
from . import analysis as cva
from .settings import options as cvo

# Almost everything in this file is contained in the Sim class
__all__ = ['Sim', 'diff_sims', 'demo', 'AlreadyRunError']


class Sim(cvb.BaseSim):
    '''
    The Sim class handles the running of the simulation: the creation of the
    population and the dynamics of the epidemic. This class handles the mechanics
    of the actual simulation, while BaseSim takes care of housekeeping (saving,
    loading, exporting, etc.). Please see the BaseSim class for additional methods.

    Args:
        pars     (dict):   parameters to modify from their default values
        datafile (str/df): filename of (Excel, CSV) data file to load, or a pandas dataframe of the data
        datacols (list):   list of column names of the data to load
        label    (str):    the name of the simulation (useful to distinguish in batch runs)
        simfile  (str):    the filename for this simulation, if it's saved
        popfile  (str):    if supplied, load the population from this file
        people   (varies): if supplied, use these pre-generated people (as a dict, SynthPop, or People object) instead of loading or generating new ones
        version  (str):    if supplied, use default parameters from this version of Covasim instead of the latest
        kwargs   (dict):   additional parameters; passed to ``cv.make_pars()``

    **Examples**::

        sim = cv.Sim()
        sim = cv.Sim(pop_size=10e3, datafile='my_data.xlsx', label='Sim with data')
    '''

    def __init__(self, pars=None, datafile=None, label=None, simfile=None,
                 popfile=None, people=None, version=None, **kwargs):

        # Set attributes
        self.label         = label    # The label/name of the simulation
        self.created       = None     # The datetime the sim was created
        self.simfile       = simfile  # The filename of the sim
        self.datafile      = datafile # The name of the data file
        self.popfile       = popfile  # The population file
        self.data          = None     # The actual data
        self.popdict       = people   # The population dictionary
        self.people        = None     # Initialize these here so methods that check their length can see they're empty
        self.t             = None     # The current time in the simulation (during execution); outside of sim.step(), its value corresponds to next timestep to be computed
        self.results       = {}       # For storing results
        self.summary       = None     # For storing a summary of the results
        self.initialized   = False    # Whether or not initialization is complete
        self.complete      = False    # Whether a simulation has completed running
        self.results_ready = False    # Whether or not results are ready
        self._default_ver  = version  # Default version of parameters used
        self._legacy_trans = None     # Whether to use the legacy transmission calculation method (slower; for reproducing earlier results)
        self._orig_pars    = None     # Store original parameters to optionally restore at the end of the simulation

        # Make default parameters (using values from parameters.py)
        default_pars = cvpar.make_pars(version=version) # Start with default pars
        super().__init__(default_pars) # Initialize and set the parameters as attributes

        # Now update everything
        self.set_metadata(simfile)  # Set the simulation date and filename
        self.update_pars(pars, **kwargs)   # Update the parameters, if provided
        self.load_data(datafile) # Load the data, if provided

        return


    def load_data(self, datafile=None, verbose=None, **kwargs):
        ''' Load the data to calibrate against, if provided '''
        if verbose is None:
            verbose = self['verbose']
        self.datafile = datafile # Store this
        if datafile is not None: # If a data file is provided, load it
            self.data = cvm.load_data(datafile=datafile, verbose=verbose, start_day=self['start_day'], **kwargs)

        return


    def initialize(self, reset=False, init_infections=True, **kwargs):
        '''
        Perform all initializations on the sim.

        This includes validating the parameters, setting the random number seed,
        creating the results structure, initializing the people, validating the
        layer parameters (which requires the people), and initializing the interventions.

        Note: to create a population to save for later use, use ``init_infections=False``.
        This will then create a fresh People object which other sims can finish
        initializing.

        Args:
            reset (bool): whether or not to reset people even if they already exist
            init_infections (bool): whether to initialize infections (default true so sim is ready, but can't reuse people then)
            kwargs (dict): passed to init_people
        '''
        self.t = 0  # The current time index
        self.validate_pars() # Ensure parameters have valid values
        self.set_seed() # Reset the random seed before the population is created
        self.init_variants() # Initialize the variants
        self.init_immunity() # initialize information about immunity (if use_waning=True)
        self.init_results() # After initializing the variant, create the results structure
        self.init_people(reset=reset, init_infections=init_infections, **kwargs) # Create all the people (the heaviest step)
        self.init_interventions()  # Initialize the interventions...
        self.init_analyzers()  # ...and the analyzers...
        self.validate_layer_pars() # Once the population is initialized, validate the layer parameters again
        self.set_seed() # Reset the random seed again so the random number stream is consistent
        self.initialized   = True
        self.complete      = False
        self.results_ready = False
        return self


    def layer_keys(self):
        '''
        Attempt to retrieve the current layer keys, in the following order: from
        the people object (for an initialized sim), from the popdict (for one in
        the process of being initialized), from the beta_layer parameter (for an
        uninitialized sim), or by assuming a default (if none of the above are
        available).
        '''
        try:
            keys = list(self['beta_layer'].keys()) # Get keys from beta_layer since the "most required" layer parameter
        except: # pragma: no cover
            keys = []
        return keys


    def reset_layer_pars(self, layer_keys=None, force=False):
        '''
        Reset the parameters to match the population.

        Args:
            layer_keys (list): override the default layer keys (use stored keys by default)
            force (bool): reset the parameters even if they already exist
        '''
        if layer_keys is None:
            if self.people is not None: # If people exist
                layer_keys = self.people.contacts.keys()
            elif self.popdict is not None:
                layer_keys = self.popdict['layer_keys']
        cvpar.reset_layer_pars(self.pars, layer_keys=layer_keys, force=force)
        return


    def validate_layer_pars(self):
        '''
        Handle layer parameters, since they need to be validated after the population
        creation, rather than before.
        '''

        # First, try to figure out what the layer keys should be and perform basic type checking
        layer_keys = self.layer_keys()
        layer_pars = cvpar.layer_pars # The names of the parameters that are specified by layer
        for lp in layer_pars:
            val = self[lp]
            if sc.isnumber(val): # It's a scalar instead of a dict, assume it's all contacts
                self[lp] = {k:val for k in layer_keys}

        # Handle key mismatches
        for lp in layer_pars:
            lp_keys = set(self.pars[lp].keys())
            if not lp_keys == set(layer_keys):
                errormsg = 'At least one layer parameter is inconsistent with the layer keys; all parameters must have the same keys:'
                errormsg += f'\nsim.layer_keys() = {layer_keys}'
                for lp2 in layer_pars: # Fail on first error, but re-loop to list all of them
                    errormsg += f'\n{lp2} = ' + ', '.join(self.pars[lp2].keys())
                raise sc.KeyNotFoundError(errormsg)

        # Handle mismatches with the population
        if self.people is not None:
            pop_keys = set(self.people.contacts.keys())
            if pop_keys != set(layer_keys): # pragma: no cover
                if not len(pop_keys):
                    errormsg = f'Your population does not have any layer keys, but your simulation does {layer_keys}. If you called cv.People() directly, you probably need cv.make_people() instead.'
                    raise sc.KeyNotFoundError(errormsg)
                else:
                    errormsg = f'Please update your parameter keys {layer_keys} to match population keys {pop_keys}. You may find sim.reset_layer_pars() helpful.'
                    raise sc.KeyNotFoundError(errormsg)

        return


    def validate_pars(self, validate_layers=True):
        '''
        Some parameters can take multiple types; this makes them consistent.

        Args:
            validate_layers (bool): whether to validate layer parameters as well via validate_layer_pars() -- usually yes, except during initialization
        '''

        # Handle population size
        pop_size   = self.pars.get('pop_size')
        scaled_pop = self.pars.get('scaled_pop')
        pop_scale  = self.pars.get('pop_scale')
        if scaled_pop is not None: # If scaled_pop is supplied, try to use it
            if pop_scale in [None, 1.0]: # Normal case, recalculate population scale
                self['pop_scale'] = scaled_pop/pop_size
            else: # Special case, recalculate number of agents
                self['pop_size'] = int(scaled_pop/pop_scale)

        # Handle types
        for key in ['pop_size', 'pop_infected']:
            try:
                self[key] = int(self[key])
            except Exception as E:
                errormsg = f'Could not convert {key}={self[key]} of {type(self[key])} to integer'
                raise ValueError(errormsg) from E

        # Handle start day
        start_day = self['start_day'] # Shorten
        if start_day in [None, 0]: # Use default start day
            start_day = '2020-03-01'
        self['start_day'] = sc.date(start_day)

        # Handle end day and n_days
        end_day = self['end_day']
        n_days = self['n_days']
        if end_day:
            self['end_day'] = sc.date(end_day)
            n_days = sc.daydiff(self['start_day'], self['end_day'])
            if n_days <= 0:
                errormsg = f"Number of days must be >0, but you supplied start={str(self['start_day'])} and end={str(self['end_day'])}, which gives n_days={n_days}"
                raise ValueError(errormsg)
            else:
                self['n_days'] = int(n_days)
        else:
            if n_days:
                self['n_days'] = int(n_days)
                self['end_day'] = self.date(n_days) # Convert from the number of days to the end day
            else:
                errormsg = f'You must supply one of n_days and end_day, not "{n_days}" and "{end_day}"'
                raise ValueError(errormsg)

        # Handle population data
        popdata_choices = ['random', 'hybrid', 'clustered', 'synthpops']
        choice = self['pop_type']
        if choice and choice not in popdata_choices: # pragma: no cover
            choicestr = ', '.join(popdata_choices)
            errormsg = f'Population type "{choice}" not available; choices are: {choicestr}'
            raise ValueError(errormsg)

        # Handle interventions, analyzers, and variants
        for key in ['interventions', 'analyzers', 'variants']: # Ensure all of them are lists
            self[key] = sc.dcp(sc.tolist(self[key], keepnone=False)) # All of these have initialize functions that run into issues if they're reused
        for i,interv in enumerate(self['interventions']):
            if isinstance(interv, dict): # It's a dictionary representation of an intervention
                self['interventions'][i] = cvi.InterventionDict(**interv)
        self['variant_map'] = {int(k):v for k,v in self['variant_map'].items()} # Ensure keys are ints, not strings of ints if loaded from JSON

        # Optionally handle layer parameters
        if validate_layers:
            self.validate_layer_pars()

        # Handle versioning
        if self._legacy_trans is None:
            default_ver = self._default_ver if self._default_ver else self.version
            self._legacy_trans = sc.compareversions(default_ver, '<3.1.1') # Handle regression

        # Handle verbose
        if self['verbose'] == 'brief':
            self['verbose'] = -1
        if not sc.isnumber(self['verbose']): # pragma: no cover
            errormsg = f'Verbose argument should be either "brief", -1, or a float, not {type(self["verbose"])} "{self["verbose"]}"'
            raise ValueError(errormsg)

        return


    def init_results(self):
        '''
        Create the main results structure.
        We differentiate between flows, stocks, and cumulative results
        The prefix "new" is used for flow variables, i.e. counting new events (infections/deaths/recoveries) on each timestep
        The prefix "n" is used for stock variables, i.e. counting the total number in any given state (sus/inf/rec/etc) on any particular timestep
        The prefix "cum" is used for cumulative variables, i.e. counting the total number that have ever been in a given state at some point in the sim
        Note that, by definition, n_dead is the same as cum_deaths and n_recovered is the same as cum_recoveries, so we only define the cumulative versions
        '''

        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = cvb.Result(*args, **kwargs, npts=self.npts)
            return output

        dcols = cvd.get_default_colors() # Get default colors

        # Flows and cumulative flows
        for key,label in cvd.result_flows.items():
            self.results[f'cum_{key}'] = init_res(f'Cumulative {label}', color=dcols[key])  # Cumulative variables -- e.g. "Cumulative infections"

        for key,label in cvd.result_flows.items(): # Repeat to keep all the cumulative keys together
            self.results[f'new_{key}'] = init_res(f'Number of new {label}', color=dcols[key]) # Flow variables -- e.g. "Number of new infections"

        # Stock variables
        for key,label in cvd.result_stocks.items():
            self.results[f'n_{key}'] = init_res(label, color=dcols[key])

        # Other variables
        self.results['n_imports']           = init_res('Number of imported infections', scale=True)
        self.results['n_alive']             = init_res('Number alive', scale=True)
        self.results['n_naive']             = init_res('Number never infected', scale=True)
        self.results['n_preinfectious']     = init_res('Number preinfectious', scale=True, color=dcols.exposed)
        self.results['n_removed']           = init_res('Number removed', scale=True, color=dcols.recovered)
        self.results['prevalence']          = init_res('Prevalence', scale=False)
        self.results['incidence']           = init_res('Incidence', scale=False)
        self.results['r_eff']               = init_res('Effective reproduction number', scale=False)
        self.results['doubling_time']       = init_res('Doubling time', scale=False)
        self.results['test_yield']          = init_res('Testing yield', scale=False)
        self.results['rel_test_yield']      = init_res('Relative testing yield', scale=False)
        self.results['frac_vaccinated']     = init_res('Proportion vaccinated', scale=False)
        self.results['pop_nabs']            = init_res('Population nab levels', scale=False, color=dcols.pop_nabs)
        self.results['pop_protection']      = init_res('Population immunity protection', scale=False, color=dcols.pop_protection)
        self.results['pop_symp_protection'] = init_res('Population symptomatic protection', scale=False, color=dcols.pop_symp_protection)

        # Handle variants
        nv = self['n_variants']
        self.results['variant'] = {}
        self.results['variant']['prevalence_by_variant'] = init_res('Prevalence by variant', scale=False, n_variants=nv)
        self.results['variant']['incidence_by_variant']  = init_res('Incidence by variant', scale=False, n_variants=nv)
        for key,label in cvd.result_flows_by_variant.items():
            self.results['variant'][f'cum_{key}'] = init_res(f'Cumulative {label}', color=dcols[key], n_variants=nv)  # Cumulative variables -- e.g. "Cumulative infections"
        for key,label in cvd.result_flows_by_variant.items():
            self.results['variant'][f'new_{key}'] = init_res(f'Number of new {label}', color=dcols[key], n_variants=nv) # Flow variables -- e.g. "Number of new infections"
        for key,label in cvd.result_stocks_by_variant.items():
            self.results['variant'][f'n_{key}'] = init_res(label, color=dcols[key], n_variants=nv)

        # Populate the rest of the results
        if self['rescale']:
            scale = 1
        else:
            scale = self['pop_scale']
        self.rescale_vec   = scale*np.ones(self.npts) # Not included in the results, but used to scale them
        self.results['date'] = self.datevec
        self.results['t']    = self.tvec
        self.results_ready   = False

        return


    def load_population(self, popfile=None, init_people=True, **kwargs):
        '''
        Load the population dictionary from file -- typically done automatically
        as part of ``sim.initialize()``.

        Supports loading either saved population dictionaries (popdicts, file ending
        .pop by convention), or ready-to-go People objects (file ending .ppl by
        convention). Either object an also be supplied directly. Once a population
        file is loaded, it is removed from the Sim object.

        Args:
            popfile (str or obj): if a string, name of the file; otherwise, the popdict or People object to load
            init_people (bool): whether to immediately convert the loaded popdict into an initialized People object
            kwargs (dict): passed to ``sim.init_people()``
        '''
        # Set the file path if not is provided
        if popfile is None and self.popfile is not None:
            popfile = self.popfile

        # Load the population into the popdict
        self.popdict = cvm.load(popfile)
        if self['verbose']:
            print(f'Loading population from {popfile}')

        if init_people:
            self.init_people(**kwargs)

        return


    def init_people(self, popdict=None, init_infections=False, reset=False, verbose=None, **kwargs):
        '''
        Create the people.

        Use ``init_infections=False`` for creating a fresh People object for use
        in future simulations

        Args:
            popdict         (any):  pre-generated people of various formats
            init_infections (bool): whether to initialize infections (default false when called directly)
            reset           (bool): whether to regenerate the people even if they already exist
            verbose         (int):  detail to print
            kwargs          (dict): passed to cv.make_people()
        '''

        # Handle inputs
        if verbose is None:
            verbose = self['verbose']
        if popdict is not None:
            self.popdict = popdict
        if verbose > 0:
            resetstr= ''
            if self.people:
                resetstr = ' (resetting people)' if reset else ' (warning: not resetting sim.people)'
            print(f'Initializing sim{resetstr} with {self["pop_size"]:0n} people for {self["n_days"]} days')
        if self.popfile and self.popdict is None: # If there's a popdict, we initialize it via cvpop.make_people()
            self.load_population(init_people=False)

        # Actually make the people
        self.people = cvpop.make_people(self, reset=reset, verbose=verbose, **kwargs)
        self.people.initialize(sim_pars=self.pars) # Fully initialize the people
        self.reset_layer_pars(force=False) # Ensure that layer keys match the loaded population
        if init_infections:
            self.init_infections(verbose=verbose)

        return self


    def init_interventions(self):
        ''' Initialize and validate the interventions '''

        # Initialization
        if self._orig_pars and 'interventions' in self._orig_pars:
            self['interventions'] = self._orig_pars.pop('interventions') # Restore

        for i,intervention in enumerate(self['interventions']):
            if isinstance(intervention, cvi.Intervention):
                intervention.initialize(self)

        # Validation
        trace_ind = np.nan # Index of the tracing intervention(s)
        test_ind = np.nan # Index of the tracing intervention(s)
        for i,intervention in enumerate(self['interventions']):
            if isinstance(intervention, (cvi.contact_tracing)):
                trace_ind = np.fmin(trace_ind, i) # Find the earliest-scheduled tracing intervention
            elif isinstance(intervention, (cvi.test_num, cvi.test_prob)):
                test_ind = np.fmax(test_ind, i) # Find the latest-scheduled testing intervention

        if not np.isnan(trace_ind): # pragma: no cover
            warnmsg = ''
            if np.isnan(test_ind):
                warnmsg = 'Note: you have defined a contact tracing intervention but no testing intervention was found. Unless this is intentional, please define at least one testing intervention.'
            elif trace_ind < test_ind:
                warnmsg = f'Note: contact tracing (index {trace_ind:.0f}) is scheduled before testing ({test_ind:.0f}); this creates a 1-day delay. Unless this is intentional, please reorder the interentions.'
            if warnmsg:
                cvm.warn(warnmsg)

        return


    def finalize_interventions(self):
        for intervention in self['interventions']:
            if isinstance(intervention, cvi.Intervention):
                intervention.finalize(self)


    def init_analyzers(self):
        ''' Initialize the analyzers '''
        if self._orig_pars and 'analyzers' in self._orig_pars:
            self['analyzers'] = self._orig_pars.pop('analyzers') # Restore

        for analyzer in self['analyzers']:
            if isinstance(analyzer, cva.Analyzer):
                analyzer.initialize(self)
        return


    def finalize_analyzers(self):
        for analyzer in self['analyzers']:
            if isinstance(analyzer, cva.Analyzer):
                analyzer.finalize(self)


    def init_variants(self):
        ''' Initialize the variants '''
        if self._orig_pars and 'variants' in self._orig_pars:
            self['variants'] = self._orig_pars.pop('variants') # Restore

        for i,variant in enumerate(self['variants']):
            if isinstance(variant, cvimm.variant):
                if not variant.initialized:
                    variant.initialize(self)
            else: # pragma: no cover
                errormsg = f'Variant {i} ({variant}) is not a cv.variant object; please create using cv.variant()'
                raise TypeError(errormsg)

        len_pars = len(self['variant_pars'])
        len_map = len(self['variant_map'])
        assert len_pars == len_map, f"variant_pars and variant_map must be the same length, but they're not: {len_pars} ≠ {len_map}"
        self['n_variants'] = len_pars # Each variant has an entry in variant_pars

        return


    def init_immunity(self, create=False):
        ''' Initialize immunity matrices and precompute nab waning for each variant '''
        if self['use_waning']:
            cvimm.init_immunity(self, create=create)
        return


    def init_infections(self, force=False, verbose=None):
        '''
        Initialize prior immunity and seed infections.

        Args:
            force (bool): initialize prior infections even if already initialized
        '''

        if verbose is None:
            verbose = self['verbose']

        # If anyone is non-naive, don't re-initialize
        if self.people.count_not('naive') == 0 or force: # Everyone is naive

            # Handle anyone who isn't susceptible
            if self['frac_susceptible'] < 1:
                inds = cvu.choose(self['pop_size'], np.round((1-self['frac_susceptible'])*self['pop_size']))
                self.people.make_nonnaive(inds=inds)

            # Create the seed infections
            if self['pop_infected']:
                inds = cvu.choose(self['pop_size'], self['pop_infected'])
                self.people.infect(inds=inds, layer='seed_infection') # Not counted by results since flows are re-initialized during the step

        elif verbose:
            print(f'People already initialized with {self.people.count_not("naive")} people non-naive and {self.people.count("exposed")} exposed; not reinitializing')

        return


    def rescale(self):
        ''' Dynamically rescale the population -- used during step() '''
        if self['rescale']:
            pop_scale = self['pop_scale']
            current_scale = self.rescale_vec[self.t]
            if current_scale < pop_scale: # We have room to rescale
                not_naive_inds = self.people.false('naive') # Find everyone not naive
                n_not_naive = len(not_naive_inds) # Number of people who are not naive
                n_people = self['pop_size'] # Number of people overall
                current_ratio = n_not_naive/n_people # Current proportion not naive
                threshold = self['rescale_threshold'] # Threshold to trigger rescaling
                if current_ratio > threshold: # Check if we've reached point when we want to rescale
                    max_ratio = pop_scale/current_scale # We don't want to exceed the total population size
                    proposed_ratio = max(current_ratio/threshold, self['rescale_factor']) # The proposed ratio to rescale: the rescale factor, unless we've exceeded it
                    scaling_ratio = min(proposed_ratio, max_ratio) # We don't want to scale by more than the maximum ratio
                    self.rescale_vec[self.t:] *= scaling_ratio # Update the rescaling factor from here on
                    n = int(round(n_not_naive*(1.0-1.0/scaling_ratio))) # For example, rescaling by 2 gives n = 0.5*not_naive_inds
                    choices = cvu.choose(max_n=n_not_naive, n=n) # Choose who to make naive again
                    new_naive_inds = not_naive_inds[choices] # Convert these back into indices for people
                    self.people.make_naive(new_naive_inds) # Make people naive again
        return


    def step(self):
        '''
        Step the simulation forward in time. Usually, the user would use sim.run()
        rather than calling sim.step() directly.
        '''

        # Set the time and if we have reached the end of the simulation, then do nothing
        if self.complete:
            raise AlreadyRunError('Simulation already complete (call sim.initialize() to re-run)')

        t = self.t

        # If it's the first timestep, infect people
        if t == 0:
            self.init_infections(verbose=False)

        # Perform initial operations
        self.rescale() # Check if we need to rescale
        people   = self.people # Shorten this for later use
        people.update_states_pre(t=t) # Update the state of everyone and count the flows
        contacts = people.update_contacts() # Compute new contacts
        hosp_max = people.count('severe')   > self['n_beds_hosp'] if self['n_beds_hosp'] is not None else False # Check for acute bed constraint
        icu_max  = people.count('critical') > self['n_beds_icu']  if self['n_beds_icu']  is not None else False # Check for ICU bed constraint

        # Randomly infect some people (imported infections)
        if self['n_imports']:
            n_imports = cvu.poisson(self['n_imports']/self.rescale_vec[self.t]) # Imported cases
            if n_imports>0:
                importation_inds = cvu.choose(max_n=self['pop_size'], n=n_imports)
                people.infect(inds=importation_inds, hosp_max=hosp_max, icu_max=icu_max, layer='importation')
                self.results['n_imports'][t] += n_imports

        # Add variants
        for variant in self['variants']:
            if isinstance(variant, cvimm.variant):
                variant.apply(self)

        # Apply interventions
        for i,intervention in enumerate(self['interventions']):
            intervention(self) # If it's a function, call it directly

        people.update_states_post() # Check for state changes after interventions

        # Compute viral loads
        frac_time = cvd.default_float(self['viral_dist']['frac_time'])
        load_ratio = cvd.default_float(self['viral_dist']['load_ratio'])
        high_cap = cvd.default_float(self['viral_dist']['high_cap'])
        date_inf = people.date_infectious
        date_rec = people.date_recovered
        date_dead = people.date_dead
        viral_load = cvu.compute_viral_load(t, date_inf, date_rec, date_dead, frac_time, load_ratio, high_cap)

        # Shorten useful parameters
        nv = self['n_variants'] # Shorten number of variants
        sus = people.susceptible
        symp = people.symptomatic
        diag = people.diagnosed
        quar = people.quarantined
        iso  = people.isolated

        prel_trans = people.rel_trans
        prel_sus   = people.rel_sus

        # Iterate through n_variants to calculate infections
        for variant in range(nv):

            # Deal with variant parameters
            asymp_factor = self['asymp_factor']
            variant_label = self.pars['variant_map'][variant]
            beta = cvd.default_float(self['beta'] * self['rel_beta'] * self['variant_pars'][variant_label]['rel_beta'])

            inf_variant = people.infectious * (people.infectious_variant == variant)
            if ~inf_variant.any():
                continue

            for lkey, layer in contacts.items():
                p1 = layer['p1']
                p2 = layer['p2']
                betas = layer['beta']

                # Compute relative transmission and susceptibility
                sus_imm = people.sus_imm[variant,:]
                iso_factor  = cvd.default_float(self['iso_factor'][lkey])
                quar_factor = cvd.default_float(self['quar_factor'][lkey])
                beta_layer  = cvd.default_float(self['beta_layer'][lkey])
                rel_trans, rel_sus = cvu.compute_trans_sus(prel_trans, prel_sus, inf_variant, sus, beta_layer, viral_load, symp, iso, quar, asymp_factor, iso_factor, quar_factor, sus_imm)

                # Calculate actual transmission
                pairs = [[p1,p2]] if not self._legacy_trans else [[p1,p2], [p2,p1]] # Support slower legacy method of calculation, but by default skip this loop
                for p1,p2 in pairs:
                    source_inds, target_inds = cvu.compute_infections(beta, p1, p2, betas, rel_trans, rel_sus, legacy=self._legacy_trans)  # Calculate transmission!
                    people.infect(inds=target_inds, hosp_max=hosp_max, icu_max=icu_max, source=source_inds, layer=lkey, variant=variant)  # Actually infect people

        # Update counts for this time step: stocks
        for key in cvd.result_stocks.keys():
            self.results[f'n_{key}'][t] = people.count(key)
        for key in cvd.result_stocks_by_variant.keys():
            for variant in range(nv):
                self.results['variant'][f'n_{key}'][variant, t] = people.count_by_variant(key, variant)

        # Update counts for this time step: flows
        for key,count in people.flows.items():
            self.results[key][t] += count
        for key,count in people.flows_variant.items():
            for variant in range(nv):
                self.results['variant'][key][variant][t] += count[variant]

        # Update nab and immunity for this time step
        if self['use_waning']:
            has_nabs = cvu.true(people.peak_nab)
            if len(has_nabs):
                cvimm.update_nab(people, inds=has_nabs)

        inds_alive = cvu.false(people.dead)
        self.results['pop_nabs'][t]            = np.sum(people.nab[inds_alive[cvu.true(people.nab[inds_alive])]])/len(inds_alive)
        self.results['pop_protection'][t]      = np.nanmean(people.sus_imm)
        self.results['pop_symp_protection'][t] = np.nanmean(people.symp_imm)

        # Apply analyzers -- same syntax as interventions
        for i,analyzer in enumerate(self['analyzers']):
            analyzer(self)

        # Tidy up
        self.t += 1
        if self.t == self.npts:
            self.complete = True

        return


    def run(self, do_plot=False, until=None, restore_pars=True, reset_seed=True, verbose=None):
        '''
        Run the simulation.

        Args:
            do_plot (bool): whether to plot
            until (int/str): day or date to run until
            restore_pars (bool): whether to make a copy of the parameters before the run and restore it after, so runs are repeatable
            reset_seed (bool): whether to reset the random number stream immediately before run
            verbose (float): level of detail to print, e.g. -1 = one-line output, 0 = no output, 0.1 = print every 10th day, 1 = print every day

        Returns:
            A pointer to the sim object (with results modified in-place)
        '''

        # Initialization steps -- start the timer, initialize the sim and the seed, and check that the sim hasn't been run
        T = sc.timer()

        if not self.initialized:
            self.initialize()
            self._orig_pars = sc.dcp(self.pars) # Create a copy of the parameters, to restore after the run, in case they are dynamically modified

        if verbose is None:
            verbose = self['verbose']

        if reset_seed:
            # Reset the RNG. If the simulation is newly created, then the RNG will be reset by sim.initialize() so the use case
            # for resetting the seed here is if the simulation has been partially run, and changing the seed is required
            self.set_seed()

        # Check for AlreadyRun errors
        errormsg = None
        until = self.npts if until is None else self.day(until)
        if until > self.npts:
            errormsg = f'Requested to run until t={until} but the simulation end is t={self.npts}'
        if self.t >= until: # NB. At the start, self.t is None so this check must occur after initialization
            errormsg = f'Simulation is currently at t={self.t}, requested to run until t={until} which has already been reached'
        if self.complete:
            errormsg = 'Simulation is already complete (call sim.initialize() to re-run)'
        if self.people.t not in [self.t, self.t-1]: # Depending on how the sim stopped, either of these states are possible
            errormsg = f'The simulation has been run independently from the people (t={self.t}, people.t={self.people.t}): if this is intentional, manually set sim.people.t = sim.t. Remember to save the people object before running the sim.'
        if errormsg:
            raise AlreadyRunError(errormsg)

        # Main simulation loop
        while self.t < until:

            # Check if we were asked to stop
            elapsed = T.toc(output=True)
            if self['timelimit'] and elapsed > self['timelimit']:
                sc.printv(f"Time limit ({self['timelimit']} s) exceeded; call sim.finalize() to compute results if desired", 1, verbose)
                return
            elif self['stopping_func'] and self['stopping_func'](self):
                sc.printv("Stopping function terminated the simulation; call sim.finalize() to compute results if desired", 1, verbose)
                return

            # Print progress
            if verbose:
                simlabel = f'"{self.label}": ' if self.label else ''
                string = f'  Running {simlabel}{self.datevec[self.t]} ({self.t:2.0f}/{self.pars["n_days"]}) ({elapsed:0.2f} s) '
                if verbose >= 2:
                    sc.heading(string)
                elif verbose>0:
                    if not (self.t % int(1.0/verbose)):
                        sc.progressbar(self.t+1, self.npts, label=string, length=20, newline=True)

            # Do the heavy lifting -- actually run the model!
            self.step()

        # If simulation reached the end, finalize the results
        if self.complete:
            self.finalize(verbose=verbose, restore_pars=restore_pars)
            sc.printv(f'Run finished after {elapsed:0.2f} s.\n', 1, verbose)
        return self


    def finalize(self, verbose=None, restore_pars=True):
        ''' Compute final results '''

        if self.results_ready:
            # Because the results are rescaled in-place, finalizing the sim cannot be run more than once or
            # otherwise the scale factor will be applied multiple times
            raise AlreadyRunError('Simulation has already been finalized')

        # Scale the results
        for reskey in self.result_keys():
            if self.results[reskey].scale: # Scale the result dynamically
                self.results[reskey].values *= self.rescale_vec
        for reskey in self.result_keys('variant'):
            if self.results['variant'][reskey].scale: # Scale the result dynamically
                self.results['variant'][reskey].values = np.einsum('ij,j->ij', self.results['variant'][reskey].values, self.rescale_vec)

        # Calculate cumulative results
        for key in cvd.result_flows.keys():
            self.results[f'cum_{key}'][:] = np.cumsum(self.results[f'new_{key}'][:], axis=0)
        for key in cvd.result_flows_by_variant.keys():
            for variant in range(self['n_variants']):
                self.results['variant'][f'cum_{key}'][variant, :] = np.cumsum(self.results['variant'][f'new_{key}'][variant, :], axis=0)
        for res in [self.results['cum_infections'], self.results['variant']['cum_infections_by_variant']]: # Include initially infected people
            res.values += self['pop_infected']*self.rescale_vec[0]

        # Finalize interventions and analyzers
        self.finalize_interventions()
        self.finalize_analyzers()

        # Final settings
        self.results_ready = True # Set this first so self.summary() knows to print the results
        self.t -= 1 # During the run, this keeps track of the next step; restore this be the final day of the sim

        # Perform calculations on results
        self.compute_results(verbose=verbose) # Calculate the rest of the results
        self.results = sc.objdict(self.results) # Convert results to a odicts/objdict to allow e.g. sim.results.diagnoses

        if restore_pars and self._orig_pars:
            preserved = ['analyzers', 'interventions']
            orig_pars_keys = list(self._orig_pars.keys()) # Get a list of keys so we can iterate over them
            for key in orig_pars_keys:
                if key not in preserved:
                    self.pars[key] = self._orig_pars.pop(key) # Restore everything except for the analyzers and interventions

        # Optionally print summary output
        if verbose: # Verbose is any non-zero value
            if verbose>0: # Verbose is any positive number
                self.summarize() # Print medium-length summary of the sim
            else:
                self.brief() # Print brief summary of the sim

        return


    def compute_results(self, verbose=None):
        ''' Perform final calculations on the results '''
        self.compute_states()
        self.compute_yield()
        self.compute_doubling()
        self.compute_r_eff()
        self.compute_summary()
        return


    def compute_states(self):
        '''
        Compute prevalence, incidence, and other states. Prevalence is the current
        number of infected people divided by the number of people who are alive.
        Incidence is the number of new infections per day divided by the susceptible
        population. Also calculates the number of people alive, the number preinfectious,
        the number removed, and recalculates susceptibles to handle scaling.
        '''
        res = self.results
        count_recov = 1-self['use_waning'] # If waning is on, don't count recovered people as removed
        self.results['n_alive'][:]         = self.scaled_pop_size - res['cum_deaths'][:] # Number of people still alive
        self.results['n_naive'][:]         = self.scaled_pop_size - res['cum_deaths'][:] - res['n_recovered'][:] - res['n_exposed'][:] # Number of people naive
        self.results['n_susceptible'][:]   = res['n_alive'][:] - res['n_exposed'][:] - count_recov*res['cum_recoveries'][:] # Recalculate the number of susceptible people, not agents
        self.results['n_preinfectious'][:] = res['n_exposed'][:] - res['n_infectious'][:] # Calculate the number not yet infectious: exposed minus infectious
        self.results['n_removed'][:]       = count_recov*res['cum_recoveries'][:] + res['cum_deaths'][:] # Calculate the number removed: recovered + dead
        self.results['prevalence'][:]      = res['n_exposed'][:]/res['n_alive'][:] # Calculate the prevalence
        self.results['incidence'][:]       = res['new_infections'][:]/res['n_susceptible'][:] # Calculate the incidence
        self.results['frac_vaccinated'][:] = res['n_vaccinated'][:]/res['n_alive'][:] # Calculate the fraction vaccinated

        self.results['variant']['incidence_by_variant'][:] = np.einsum('ji,i->ji',res['variant']['new_infections_by_variant'][:], 1/res['n_susceptible'][:]) # Calculate the incidence
        self.results['variant']['prevalence_by_variant'][:] = np.einsum('ji,i->ji',res['variant']['new_infections_by_variant'][:], 1/res['n_alive'][:])  # Calculate the prevalence

        return


    def compute_yield(self):
        '''
        Compute test yield -- number of positive tests divided by the total number
        of tests, also called test positivity rate. Relative yield is with respect
        to prevalence: i.e., how the yield compares to what the yield would be from
        choosing a person at random from the population.
        '''
        # Absolute yield
        res = self.results
        inds = cvu.true(res['new_tests'][:]) # Pull out non-zero numbers of tests
        self.results['test_yield'][inds] = res['new_diagnoses'][inds]/res['new_tests'][inds] # Calculate the yield

        # Relative yield
        inds = cvu.true(res['n_infectious'][:]) # To avoid divide by zero if no one is infectious
        denom = res['n_infectious'][inds] / (res['n_alive'][inds] - res['cum_diagnoses'][inds]) # Alive + undiagnosed people might test; infectious people will test positive
        self.results['rel_test_yield'][inds] = self.results['test_yield'][inds]/denom # Calculate the relative yield
        return


    def compute_doubling(self, window=3, max_doubling_time=30):
        '''
        Calculate doubling time using exponential approximation -- a more detailed
        approach is in utils.py. Compares infections at time t to infections at time
        t-window, and uses that to compute the doubling time. For example, if there are
        100 cumulative infections on day 12 and 200 infections on day 19, doubling
        time is 7 days.

        Args:
            window (float): the size of the window used (larger values are more accurate but less precise)
            max_doubling_time (float): doubling time could be infinite, so this places a bound on it

        Returns:
            doubling_time (array): the doubling time results array
        '''

        cum_infections = self.results['cum_infections'].values
        infections_now = cum_infections[window:]
        infections_prev = cum_infections[:-window]
        use = (infections_prev > 0) & (infections_now > infections_prev)
        doubling_time = window * np.log(2) / np.log(infections_now[use] / infections_prev[use])
        self.results['doubling_time'][:] = np.nan
        self.results['doubling_time'][window:][use] = np.minimum(doubling_time, max_doubling_time)
        return self.results['doubling_time'].values


    def compute_r_eff(self, method='daily', smoothing=2, window=7):
        '''
        Effective reproduction number based on number of people each person infected.

        Args:
            method (str): 'daily' uses daily infections, 'infectious' counts from the date infectious, 'outcome' counts from the date recovered/dead
            smoothing (int): the number of steps to smooth over for the 'daily' method
            window (int): the size of the window used for 'infectious' and 'outcome' calculations (larger values are more accurate but less precise)

        Returns:
            r_eff (array): the r_eff results array
        '''

        # Initialize arrays to hold sources and targets infected each day
        sources = np.zeros(self.npts)
        targets = np.zeros(self.npts)
        window = int(window)

        # Default method -- calculate the daily infections
        if method == 'daily':

            # Find the dates that everyone became infectious and recovered, and hence calculate infectious duration
            recov_inds   = self.people.defined('date_recovered')
            dead_inds    = self.people.defined('date_dead')
            date_recov   = self.people.date_recovered[recov_inds]
            date_dead    = self.people.date_dead[dead_inds]
            date_outcome = np.concatenate((date_recov, date_dead))
            inds         = np.concatenate((recov_inds, dead_inds))
            date_inf     = self.people.date_infectious[inds]
            if len(date_outcome):
                mean_inf     = date_outcome.mean() - date_inf.mean()
            else:
                warnmsg ='There were no infections during the simulation'
                cvm.warn(warnmsg)
                mean_inf = 0 # Doesn't matter since r_eff is 0

            # Calculate R_eff as the mean infectious duration times the number of new infectious divided by the number of infectious people on a given day
            new_infections = self.results['new_infections'].values - self.results['n_imports'].values
            n_infectious = self.results['n_infectious'].values
            raw_values = mean_inf*np.divide(new_infections, n_infectious, out=np.zeros(self.npts), where=n_infectious>0)

            # Handle smoothing, including with too-short arrays
            len_raw = len(raw_values) # Calculate the number of raw values
            dur_pars = self['dur'][0] if isinstance(self['dur'], list) else self['dur'] # Note: does not take variants into account
            if len_raw >= 3: # Can't smooth arrays shorter than this since the default smoothing kernel has length 3
                initial_period = dur_pars['exp2inf']['par1'] + dur_pars['asym2rec']['par1'] # Approximate the duration of the seed infections for averaging
                initial_period = int(min(len_raw, initial_period)) # Ensure we don't have too many points
                for ind in range(initial_period): # Loop over each of the initial inds
                    raw_values[ind] = raw_values[ind:initial_period].mean() # Replace these values with their average
                values = sc.smooth(raw_values, smoothing)
                values[:smoothing] = raw_values[:smoothing] # To avoid numerical effects, replace the beginning and end with the original
                values[-smoothing:] = raw_values[-smoothing:]
            else:
                values = raw_values

        # Alternate (traditional) method -- count from the date of infection or outcome
        elif method in ['infectious', 'outcome']:

            # Store a mapping from each source to their date
            source_dates = {}

            for t in self.tvec:

                # Sources are easy -- count up the arrays for all the people who became infections on that day
                if method == 'infectious':
                    inds = cvu.true(t == self.people.date_infectious) # Find people who became infectious on this timestep
                elif method == 'outcome':
                    recov_inds = cvu.true(t == self.people.date_recovered) # Find people who recovered on this timestep
                    dead_inds  = cvu.true(t == self.people.date_dead)  # Find people who died on this timestep
                    inds       = np.concatenate((recov_inds, dead_inds))
                sources[t] = len(inds)

                # Create the mapping from sources to dates
                for ind in inds:
                    source_dates[ind] = t

            # Targets are hard -- loop over the transmission tree
            for transdict in self.people.infection_log:
                source = transdict['source']
                if source is not None and source in source_dates: # Skip seed infections and people with e.g. recovery after the end of the sim
                    source_date = source_dates[source]
                    targets[source_date] += 1

                # for ind in inds:
                #     targets[t] += len(self.people.transtree.targets[ind])

            # Populate the array -- to avoid divide-by-zero, skip indices that are 0
            r_eff = np.divide(targets, sources, out=np.full(self.npts, np.nan), where=sources > 0)

            # Use stored weights calculate the moving average over the window of timesteps, n
            num = np.nancumsum(r_eff * sources)
            num[window:] = num[window:] - num[:-window]
            den = np.cumsum(sources)
            den[window:] = den[window:] - den[:-window]
            values = np.divide(num, den, out=np.full(self.npts, np.nan), where=den > 0)

        # Method not recognized
        else: # pragma: no cover
            errormsg = f'Method must be "daily", "infectious", or "outcome", not "{method}"'
            raise ValueError(errormsg)

        # Set the values and return
        self.results['r_eff'].values[:] = values

        return self.results['r_eff'].values


    def compute_gen_time(self):
        '''
        Calculate the generation time (or serial interval). There are two
        ways to do this calculation. The 'true' interval (exposure time to
        exposure time) or 'clinical' (symptom onset to symptom onset).

        Returns:
            gen_time (dict): the generation time results
        '''

        intervals1 = np.zeros(len(self.people))
        intervals2 = np.zeros(len(self.people))
        pos1 = 0
        pos2 = 0
        date_exposed = self.people.date_exposed
        date_symptomatic = self.people.date_symptomatic

        for infection in self.people.infection_log:
            if infection['source'] is not None:
                source_ind = infection['source']
                target_ind = infection['target']
                intervals1[pos1] = date_exposed[target_ind] - date_exposed[source_ind]
                pos1 += 1
                if np.isfinite(date_symptomatic[source_ind]) and np.isfinite(date_symptomatic[target_ind]):
                    intervals2[pos2] = date_symptomatic[target_ind] - date_symptomatic[source_ind]
                    pos2 += 1

        self.results['gen_time'] = {
                'true':         np.mean(intervals1[:pos1]),
                'true_std':     np.std(intervals1[:pos1]),
                'clinical':     np.mean(intervals2[:pos2]),
                'clinical_std': np.std(intervals2[:pos2])}
        return self.results['gen_time']


    def compute_summary(self, full=None, t=None, update=True, output=False, require_run=False):
        '''
        Compute the summary dict and string for the sim. Used internally; see
        sim.summarize() for the user version.

        Args:
            full (bool): whether or not to print all results (by default, only cumulative)
            t (int/str): day or date to compute summary for (by default, the last point)
            update (bool): whether to update the stored sim.summary
            output (bool): whether to return the summary
            require_run (bool): whether to raise an exception if simulations have not been run yet
        '''
        if t is None:
            t = self.day(self.t)

        # Compute the summary
        if require_run and not self.results_ready:
            errormsg = 'Simulation not yet run'
            raise RuntimeError(errormsg)

        summary = sc.objdict()
        for key in self.result_keys():
            summary[key] = self.results[key][t]

        # Update the stored state
        if update:
            self.summary = summary

        # Optionally return
        if output:
            return summary
        else:
            return


    def summarize(self, full=False, t=None, sep=None, output=False):
        '''
        Print a medium-length summary of the simulation, drawing from the last time
        point in the simulation by default. Called by default at the end of a sim run.
        See also sim.disp() (detailed output) and sim.brief() (short output).

        Args:
            full   (bool):    whether or not to print all results (by default, only cumulative)
            t      (int/str): day or date to compute summary for (by default, the last point)
            sep    (str):     thousands separator (default ',')
            output (bool):    whether to return the summary instead of printing it

        **Examples**::

            sim = cv.Sim(label='Example sim', verbose=0) # Set to run silently
            sim.run() # Run the sim
            sim.summarize() # Print medium-length summary of the sim
            sim.summarize(t=24, full=True) # Print a "slice" of all sim results on day 24
        '''
        # Compute the summary
        summary = self.compute_summary(full=full, t=t, update=False, output=True)

        # Construct the output string
        if sep is None: sep = cvo.sep # Default separator
        labelstr = f' "{self.label}"' if self.label else ''
        string = f'Simulation{labelstr} summary:\n'
        for key in self.result_keys():
            if full or key.startswith('cum_'):
                val = np.round(summary[key])
                string += f'   {val:10,.0f} {self.results[key].name.lower()}\n'.replace(',', sep) # Use replace since it's more flexible

        # Print or return string
        if not output:
            print(string)
        else:
            return string


    def disp(self, output=False):
        '''
        Display a verbose description of a sim. See also sim.summarize() (medium
        length output) and sim.brief() (short output).

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            sim = cv.Sim(label='Example sim', verbose=0) # Set to run silently
            sim.run() # Run the sim
            sim.disp() # Displays detailed output
        '''
        string = self._disp()
        if not output:
            print(string)
        else:
            return string


    def brief(self, output=False):
        '''
        Print a one-line description of a sim. See also sim.disp() (detailed output)
        and sim.summarize() (medium length output). The symbol "⚙" is used to show
        infections, and "☠" is used to show deaths.

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            sim = cv.Sim(label='Example sim', verbose=0) # Set to run silently
            sim.run() # Run the sim
            sim.brief() # Prints one-line output
        '''
        string = self._brief()
        if not output:
            print(string)
        else:
            return string


    def compute_fit(self, *args, **kwargs):
        '''
        Compute the fit between the model and the data. See cv.Fit() for more
        information.

        Args:
            args   (list): passed to cv.Fit()
            kwargs (dict): passed to cv.Fit()

        Returns:
            A Fit object

        **Example**::

            sim = cv.Sim(datafile='data.csv')
            sim.run()
            fit = sim.compute_fit()
            fit.plot()
        '''
        if not self.results_ready:
            errormsg = 'Cannot compute fit since results are not ready yet -- did you run the sim?'
            raise RuntimeError(errormsg)
        self.fit = cva.Fit(self, *args, **kwargs)
        return self.fit


    def calibrate(self, calib_pars, **kwargs):
        '''
        Automatically calibrate the simulation, returning a Calibration object
        (a type of analyzer). See the documentation on that class for more information.

        Args:
            calib_pars (dict): a dictionary of the parameters to calibrate of the format dict(key1=[best, low, high])
            kwargs (dict): passed to cv.Calibration()

        Returns:
            A Calibration object

        **Example**::

            sim = cv.Sim(datafile='data.csv')
            calib_pars = dict(beta=[0.015, 0.010, 0.020])
            calib = sim.calibrate(calib_pars, n_trials=50)
            calib.plot()
        '''
        calib = cva.Calibration(sim=self, calib_pars=calib_pars, **kwargs)
        calib.calibrate()
        return calib


    def make_age_histogram(self, *args, output=True, **kwargs):
        '''
        Calculate the age histograms of infections, deaths, diagnoses, etc. See
        cv.age_histogram() for more information. This can be used alternatively
        to supplying the age histogram as an analyzer to the sim. If used this
        way, it can only record the final time point since the states of each
        person are not saved during the sim.

        Args:
            output (bool): whether or not to return the age histogram; if not, store in sim.results
            args   (list): passed to cv.age_histogram()
            kwargs (dict): passed to cv.age_histogram()

        **Example**::

            sim = cv.Sim()
            sim.run()
            agehist = sim.make_age_histogram()
            agehist.plot()
        '''
        if not self.results_ready:
            errormsg = 'Cannot make age histogram since results are not ready yet -- did you run the sim?'
            raise RuntimeError(errormsg)
        agehist = cva.age_histogram(sim=self, *args, **kwargs)
        if output:
            return agehist
        else: # pragma: no cover
            self.results.agehist = agehist
            return


    def make_transtree(self, *args, output=True, **kwargs):
        '''
        Create a TransTree (transmission tree) object, for analyzing the pattern
        of transmissions in the simulation. See cv.TransTree() for more information.

        Args:
            output (bool): whether or not to return the TransTree; if not, store in sim.results
            args   (list): passed to cv.TransTree()
            kwargs (dict): passed to cv.TransTree()

        **Example**::

            sim = cv.Sim()
            sim.run()
            tt = sim.make_transtree()
        '''
        if not self.results_ready:
            errormsg = 'Cannot compute transmission tree since results are not ready yet -- did you run the sim?'
            raise RuntimeError(errormsg)
        tt = cva.TransTree(self, *args, **kwargs)
        if output:
            return tt
        else: # pragma: no cover
            self.results.transtree = tt
            return


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
            - ``outer``:      only show the date labels on the outer (bottom) plots

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


def diff_sims(sim1, sim2, skip_key_diffs=False, skip=None, output=False, die=False):
    '''
    Compute the difference of the summaries of two simulations, and print any
    values which differ.

    Args:
        sim1 (sim/dict): either a simulation object or the sim.summary dictionary
        sim2 (sim/dict): ditto
        skip_key_diffs (bool): whether to skip keys that don't match between sims
        skip (list): a list of values to skip
        output (bool): whether to return the output as a string (otherwise print)
        die (bool): whether to raise an exception if the sims don't match
        require_run (bool): require that the simulations have been run

    **Example**::

        s1 = cv.Sim(beta=0.01)
        s2 = cv.Sim(beta=0.02)
        s1.run()
        s2.run()
        cv.diff_sims(s1, s2)
    '''

    if isinstance(sim1, Sim):
        sim1 = sim1.compute_summary(update=False, output=True, require_run=True)
    if isinstance(sim2, Sim):
        sim2 = sim2.compute_summary(update=False, output=True, require_run=True)
    for sim in [sim1, sim2]:
        if not isinstance(sim, dict): # pragma: no cover
            errormsg = f'Cannot compare object of type {type(sim)}, must be a sim or a sim.summary dict'
            raise TypeError(errormsg)

    # Compare keys
    keymatchmsg = ''
    sim1_keys = set(sim1.keys())
    sim2_keys = set(sim2.keys())
    if sim1_keys != sim2_keys and not skip_key_diffs: # pragma: no cover
        keymatchmsg = "Keys don't match!\n"
        missing = list(sim1_keys - sim2_keys)
        extra   = list(sim2_keys - sim1_keys)
        if missing:
            keymatchmsg += f'  Missing sim1 keys: {missing}\ns'
        if extra:
            keymatchmsg += f'  Extra sim2 keys: {extra}\n'

    # Compare values
    valmatchmsg = ''
    mismatches = {}
    skip = sc.tolist(skip)
    for key in sim2.keys(): # To ensure order
        if key in sim1_keys and key not in skip: # If a key is missing, don't count it as a mismatch
            sim1_val = sim1[key]
            sim2_val = sim2[key]
            both_nan = sc.isnumber(sim1_val, isnan=True) and sc.isnumber(sim2_val, isnan=True)
            if not np.isclose(sim1_val,sim2_val) and not both_nan:
                mismatches[key] = {'sim1': sim1_val, 'sim2': sim2_val}

    if len(mismatches):
        valmatchmsg = '\nThe following values differ between the two simulations:\n'
        df = pd.DataFrame.from_dict(mismatches).transpose()
        diff   = []
        ratio  = []
        change = []
        small_change = 1e-3 # Define a small change, e.g. a rounding error
        for mdict in mismatches.values():
            old = mdict['sim1']
            new = mdict['sim2']
            numeric = sc.isnumber(sim1_val) and sc.isnumber(sim2_val)
            if numeric and old>0:
                this_diff  = new - old
                this_ratio = new/old
                abs_ratio  = max(this_ratio, 1.0/this_ratio)

                # Set the character to use
                if abs_ratio<small_change:
                    change_char = '≈'
                elif new > old:
                    change_char = '↑'
                elif new < old:
                    change_char = '↓'
                else:
                    errormsg = f'Could not determine relationship between sim1={old} and sim2={new}'
                    raise ValueError(errormsg)

                # Set how many repeats it should have
                repeats = 1
                if abs_ratio >= 1.1:
                    repeats = 2
                if abs_ratio >= 2:
                    repeats = 3
                if abs_ratio >= 10:
                    repeats = 4

                this_change = change_char*repeats
            else: # pragma: no cover
                this_diff   = np.nan
                this_ratio  = np.nan
                this_change = 'N/A'

            diff.append(this_diff)
            ratio.append(this_ratio)
            change.append(this_change)

        df['diff'] = diff
        df['ratio'] = ratio
        for col in ['sim1', 'sim2', 'diff', 'ratio']:
            df[col] = df[col].round(decimals=3)
        df['change'] = change
        valmatchmsg += str(df)

    # Raise an error if mismatches were found
    mismatchmsg = keymatchmsg + valmatchmsg
    if mismatchmsg: # pragma: no cover
        if die:
            raise ValueError(mismatchmsg)
        elif output:
            return mismatchmsg
        else:
            print(mismatchmsg)
    else:
        if not output:
            print('Sims match')
    return


def demo(preset=None, to_plot=None, scens=None, run_args=None, plot_args=None, **kwargs):
    '''
    Shortcut for ``cv.Sim().run().plot()``.

    Args:
        preset (str): use a preset run configuration; currently the only option is "full"
        to_plot (str): what to plot
        scens (dict): dictionary of scenarios to run as a multisim, if preset='full'
        kwargs (dict): passed to Sim()
        run_args (dict): passed to sim.run()
        plot_args (dict): passed to sim.plot()

    **Examples**::

        cv.demo() # Simplest example
        cv.demo('full') # Full example
        cv.demo('full', overview=True) # Plot all results
        cv.demo(beta=0.020, run_args={'verbose':0}, plot_args={'to_plot':'overview'}) # Pass in custom values
    '''
    from . import interventions as cvi # To avoid circular imports
    from . import run as cvr

    run_args = sc.mergedicts(run_args)
    plot_args = sc.mergedicts(plot_args)
    if to_plot:
        plot_args = sc.mergedicts(plot_args, {'to_plot':to_plot})

    if not preset:
        sim = Sim(**kwargs)
        sim.run(**run_args)
        sim.plot(**plot_args)
        return sim

    elif preset == 'full':

            # Define interventions
            cb = cvi.change_beta(days=40, changes=0.5)
            tp = cvi.test_prob(start_day=20, symp_prob=0.1, asymp_prob=0.01)
            ct = cvi.contact_tracing(trace_probs=0.3, start_day=50)

            # Define the parameters
            pars = dict(
                pop_size      = 20e3,         # Population size
                pop_infected  = 100,          # Number of initial infections -- use more for increased robustness
                pop_type      = 'hybrid',     # Population to use -- "hybrid" is random with household, school,and work structure
                n_days        = 60,           # Number of days to simulate
                verbose       = 0,            # Don't print details of the run
                rand_seed     = 2,            # Set a non-default seed
                interventions = [cb, tp, ct], # Include the most common interventions
            )
            pars = sc.mergedicts(pars, kwargs)
            if scens is None:
                scens = ('beta', {'Low beta':0.012, 'Medium beta':0.016, 'High beta':0.020})
            scenpar = scens[0]
            scenval = scens[1]

            # Run the simulations
            sims = [Sim(pars, **{scenpar:val}, label=label) for label,val in scenval.items()]
            msim = cvr.MultiSim(sims)
            msim.run(**run_args)
            msim.plot(**plot_args)
            msim.median()
            msim.plot(**plot_args)
            return msim

    else:
        errormsg = f'Could not understand preset argument "{preset}"; must be None or "full"'
        raise NotImplementedError(errormsg)


class AlreadyRunError(RuntimeError):
    '''
    This error is raised if a simulation is run in such a way that no timesteps
    will be taken. This error is a distinct type so that it can be safely caught
    and ignored if required, but it is anticipated that most of the time, calling
    sim.run() and not taking any timesteps, would be an inadvertent error.
    '''
    pass
