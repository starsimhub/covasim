'''
Defines classes and methods for calculating immunity
'''

import numpy as np
import sciris as sc
from . import utils as cvu
from . import defaults as cvd
from . import parameters as cvpar
from . import interventions as cvi


# %% Define variant class -- all other functions are for internal use only

__all__ = ['variant']


class variant(sc.prettyobj):
    '''
    Add a new variant to the sim

    Args:
        variant (str/dict): name of variant, or dictionary of parameters specifying information about the variant
        days   (int/list): day(s) on which new variant is introduced
        label       (str): if variant is supplied as a dict, the name of the variant
        n_imports   (int): the number of imports of the variant to be added
        rescale    (bool): whether the number of imports should be rescaled with the population

    **Example**::

        alpha    = cv.variant('alpha', days=10) # Make the alpha variant B117 active from day 10
        p1      = cv.variant('p1', days=15) # Make variant P1 active from day 15
        my_var  = cv.variant(variant={'rel_beta': 2.5}, label='My variant', days=20)
        sim     = cv.Sim(variants=[alpha, p1, my_var]).run() # Add them all to the sim
        sim2    = cv.Sim(variants=cv.variant('alpha', days=0, n_imports=20), pop_infected=0).run() # Replace default variant with alpha
    '''

    def __init__(self, variant, days, label=None, n_imports=1, rescale=True):
        self.days = days # Handle inputs
        self.n_imports = int(n_imports)
        self.rescale   = rescale
        self.index     = None # Index of the variant in the sim; set later
        self.label     = None # Variant label (used as a dict key)
        self.p         = None # This is where the parameters will be stored
        self.parse(variant=variant, label=label) # Variants can be defined in different ways: process these here
        self.initialized = False
        return


    def parse(self, variant=None, label=None):
        ''' Unpack variant information, which may be given as either a string or a dict '''

        # Option 1: variants can be chosen from a list of pre-defined variants
        if isinstance(variant, str):

            choices, mapping = cvpar.get_variant_choices()
            known_variant_pars = cvpar.get_variant_pars()

            label = variant.lower()
            for txt in ['.', ' ', 'variant', 'variant', 'voc']:
                label = label.replace(txt, '')

            if label in mapping:
                label = mapping[label]
                variant_pars = known_variant_pars[label]
            else:
                errormsg = f'The selected variant "{variant}" is not implemented; choices are:\n{sc.pp(choices, doprint=False)}'
                raise NotImplementedError(errormsg)

        # Option 2: variants can be specified as a dict of pars
        elif isinstance(variant, dict):

            default_variant_pars = cvpar.get_variant_pars(default=True)
            default_keys = list(default_variant_pars.keys())

            # Parse label
            variant_pars = variant
            label = variant_pars.pop('label', label) # Allow including the label in the parameters
            if label is None:
                label = 'custom'

            # Check that valid keys have been supplied...
            invalid = []
            for key in variant_pars.keys():
                if key not in default_keys:
                    invalid.append(key)
            if len(invalid):
                errormsg = f'Could not parse variant keys "{sc.strjoin(invalid)}"; valid keys are: "{sc.strjoin(cvd.variant_pars)}"'
                raise sc.KeyNotFoundError(errormsg)

            # ...and populate any that are missing
            for key in default_keys:
                if key not in variant_pars:
                    variant_pars[key] = default_variant_pars[key]

        else:
            errormsg = f'Could not understand {type(variant)}, please specify as a dict or a predefined variant:\n{sc.pp(choices, doprint=False)}'
            raise ValueError(errormsg)

        # Set label and parameters
        self.label = label
        self.p = variant_pars

        return


    def initialize(self, sim):
        ''' Update variant info in sim '''
        self.days = cvi.process_days(sim, self.days) # Convert days into correct format
        sim['variant_pars'][self.label] = self.p  # Store the parameters
        self.index = list(sim['variant_pars'].keys()).index(self.label) # Find where we are in the list
        sim['variant_map'][self.index]  = self.label # Use that to populate the reverse mapping
        self.initialized = True
        return


    def apply(self, sim):
        ''' Introduce new infections with this variant '''
        for ind in cvi.find_day(self.days, sim.t, interv=self, sim=sim): # Time to introduce variant
            susceptible_inds = cvu.true(sim.people.susceptible)
            rescale_factor = sim.rescale_vec[sim.t] if self.rescale else 1.0
            scaled_imports = self.n_imports/rescale_factor
            n_imports = sc.randround(scaled_imports) # Round stochastically to the nearest number of imports
            if self.n_imports > 0 and n_imports == 0 and sim['verbose']:
                msg = f'Warning: {self.n_imports:n} imported infections of {self.label} were specified on day {sim.t}, but given the rescale factor of {rescale_factor:n}, no agents were infected. Increase the number of imports or use more agents.'
                print(msg)
            importation_inds = np.random.choice(susceptible_inds, n_imports, replace=False) # Can't use cvu.choice() since sampling from indices
            sim.people.infect(inds=importation_inds, layer='importation', variant=self.index)
            sim.results['n_imports'][sim.t] += n_imports
        return




#%% Neutralizing antibody methods


def update_peak_nab(people, inds, nab_pars, symp=None):
    '''
    Update peak NAb level

    This function updates the peak NAb level for individuals when a NAb event occurs.
        - individuals that already have NAbs from a previous vaccination/infection have their NAb level boosted;
        - individuals without prior NAbs are assigned an initial level drawn from a distribution. This level
            depends on whether the NAbs are from a natural infection (and if so, on the infection's severity)
            or from a vaccination (and if so, on the type of vaccine).

    Args:
        people: A people object
        inds: Array of people indices
        nab_pars: Parameters from which to draw values for quantities like ['nab_init'] - either sim pars (for natural immunity) or vaccine pars
        symp: either None (if NAbs are vaccine-derived), or a dictionary keyed by 'asymp', 'mild', and 'sev' giving the indices of people with each of those symptoms

    Returns: None
    '''

    # Extract parameters and indices
    pars = people.pars
    has_nabs = people.nab[inds] > 0
    no_prior_nab_inds = inds[~has_nabs]
    prior_nab_inds = inds[has_nabs]

    # 1) Individuals that already have NAbs from a previous vaccination/infection have their NAb level boosted
    if len(prior_nab_inds):
        boost_factor = nab_pars['nab_boost']
        people.peak_nab[prior_nab_inds] *= boost_factor


    # 2) Individuals without prior NAbs are assigned an initial level drawn from a distribution.
    if len(no_prior_nab_inds):

        # Firstly, ensure that we don't try to apply a booster effect to people without NAbs
        if nab_pars['nab_init'] is None:
            errormsg = f'Attempt to administer a vaccine without an initial NAb distribution to {len(no_prior_nab_inds)} unvaccinated people failed.'
            raise ValueError(errormsg)

        # Now draw the initial NAb levels
        init_nab = cvu.sample(**nab_pars['nab_init'], size=len(no_prior_nab_inds))
        no_prior_nab = (2 ** init_nab)

        # Next, these initial NAb levels are normalized to be equivalent to "vaccine NAbs".
        # This is done so that when we check immunity, we can calculate immune protection
        # using a single curve and account for multiple sources of immunity (vaccine and natural).
        if symp is not None:
            # Setting up for symptom scaling
            prior_symp = np.full(pars['pop_size'], np.nan)
            prior_symp[symp['asymp']] = pars['rel_imm_symp']['asymp']
            prior_symp[symp['mild']] = pars['rel_imm_symp']['mild']
            prior_symp[symp['sev']] = pars['rel_imm_symp']['severe']
            prior_symp[prior_nab_inds] = np.nan
            prior_symp = prior_symp[~np.isnan(prior_symp)]
            # Applying symptom scaling and a normalization factor to the NAbs
            norm_factor = 1 + nab_pars['nab_eff']['alpha_inf_diff']
            no_prior_nab = no_prior_nab * prior_symp * norm_factor

        # Update people's peak NAbs
        people.peak_nab[no_prior_nab_inds] = no_prior_nab

    # Update time of nab event
    people.t_nab_event[inds] = people.t

    return


def update_nab(people, inds):
    '''
    Step NAb levels forward in time
    '''
    t_since_boost = people.t - people.t_nab_event[inds]
    people.nab[inds] += people.pars['nab_kin'][t_since_boost]*people.peak_nab[inds]
    people.nab[inds] = np.where(people.nab[inds]<0, 0, people.nab[inds]) # Make sure nabs don't drop below 0
    people.nab[inds] = np.where([people.nab[inds] > people.peak_nab[inds]], people.peak_nab[inds], people.nab[inds]) # Make sure nabs don't exceed peak_nab
    return


def calc_VE(nab, ax, pars):
    '''
        Convert NAb levels to immunity protection factors, using the functional form
        given in this paper: https://doi.org/10.1101/2021.03.09.21252641

        Args:
            nab  (arr)  : an array of effective NAb levels (i.e. actual NAb levels, scaled by cross-immunity)
            ax   (str)  : axis of protection; can be 'sus', 'symp' or 'sev', corresponding to the efficacy of protection against infection, symptoms, and severe disease respectively
            pars (dict) : dictionary of parameters for the vaccine efficacy

        Returns:
            an array the same size as NAb, containing the immunity protection factors for the specified axis
         '''

    choices = ['sus', 'symp', 'sev']
    if ax not in choices:
        errormsg = f'Choice {ax} not in list of choices: {sc.strjoin(choices)}'
        raise ValueError(errormsg)

    if ax == 'sus':
        alpha = pars['alpha_inf']
        beta = pars['beta_inf']
    elif ax == 'symp':
        alpha = pars['alpha_symp_inf']
        beta = pars['beta_symp_inf']
    else:
        alpha = pars['alpha_sev_symp']
        beta = pars['beta_sev_symp']

    exp_lo = np.exp(alpha) * nab**beta
    output = exp_lo/(1+exp_lo) # Inverse logit function
    return output


def calc_VE_symp(nab, pars):
    '''
    Converts NAbs to marginal VE against symptomatic disease
    '''

    exp_lo_inf = np.exp(pars['alpha_inf']) * nab**pars['beta_inf']
    inv_lo_inf = exp_lo_inf / (1 + exp_lo_inf)

    exp_lo_symp_inf = np.exp(pars['alpha_symp_inf']) * nab**pars['beta_symp_inf']
    inv_lo_symp_inf = exp_lo_symp_inf / (1 + exp_lo_symp_inf)

    VE_symp = 1 - ((1 - inv_lo_inf)*(1 - inv_lo_symp_inf))
    return VE_symp




#%% Immunity methods

def init_immunity(sim, create=False):
    ''' Initialize immunity matrices with all variants that will eventually be in the sim'''

    # Don't use this function if immunity is turned off
    if not sim['use_waning']:
        return

    # Pull out all of the circulating variants for cross-immunity
    nv = sim['n_variants']

    # If immunity values have been provided, process them
    if sim['immunity'] is None or create:

        # Firstly, initialize immunity matrix with defaults. These are then overwitten with variant-specific values below
        # Susceptibility matrix is of size sim['n_variants']*sim['n_variants']
        immunity = np.ones((nv, nv), dtype=cvd.default_float)  # Fill with defaults

        # Next, overwrite these defaults with any known immunity values about specific variants
        default_cross_immunity = cvpar.get_cross_immunity()
        for i in range(nv):
            label_i = sim['variant_map'][i]
            for j in range(nv):
                label_j = sim['variant_map'][j]
                if label_i in default_cross_immunity and label_j in default_cross_immunity:
                    immunity[j][i] = default_cross_immunity[label_j][label_i]

        sim['immunity'] = immunity

    # Next, precompute the NAb kinetics and store these for access during the sim
    sim['nab_kin'] = precompute_waning(length=sim.npts, pars=sim['nab_decay'])

    return


def check_immunity(people, variants=None):
    '''
    Calculate people's immunity on this timestep from prior infections + vaccination. Calculates effective NAbs by
    weighting individuals NAbs by source and then calculating efficacy.

    There are two fundamental sources of immunity:

        (1) prior exposure: degree of protection depends on variant, prior symptoms, and time since recovery
        (2) vaccination: degree of protection depends on variant, vaccine, and time since vaccination
    '''

    # Handle parameters and indices
    pars = people.pars
    nab_eff = pars['nab_eff']
    if variants is None:
        variants = range(pars['n_variants'])

    # Update immunity for each variant
    for variant in variants:
        natural_imm = np.zeros(len(people))
        vaccine_imm = np.zeros(len(people))

        # Natural immunity weighting
        was_inf = cvu.true(people.t >= people.date_recovered)  # Had a previous exposure, now recovered
        recovered_variant = people.recovered_variant[was_inf]
        immunity = pars['immunity'][variant, :]  # retrieve cross immunity factors for natural infection
        natural_imm[was_inf] = immunity[recovered_variant.astype(int)]

        # Vaccine immunity weighting
        is_vacc = cvu.true(people.vaccinated)  # Vaccinated
        vacc_source = people.vaccine_source[is_vacc]
        if len(is_vacc) and len(pars['vaccine_pars']):  # if using simple_vaccine, do not apply
            vx_pars = pars['vaccine_pars']
            vx_map = pars['vaccine_map']
            var_key = pars['variant_map'][variant]
            imm_arr = np.zeros(max(vx_map.keys()) + 1)
            for num, key in vx_map.items():
                imm_arr[num] = vx_pars[key][var_key]
            vaccine_imm[is_vacc] = imm_arr[vacc_source]

        # Calculate overall immunity
        imm = np.maximum(natural_imm, vaccine_imm)  # Use the larger of natural immunity or vaccine immunity
        effective_nabs = people.nab * imm
        people.sus_imm[variant, :]  = calc_VE(effective_nabs, 'sus', nab_eff)
        people.symp_imm[variant, :] = calc_VE(effective_nabs, 'symp', nab_eff)
        people.sev_imm[variant, :]  = calc_VE(effective_nabs, 'sev', nab_eff)

    return



#%% Methods for computing waning

def precompute_waning(length, pars=None):
    '''
    Process functional form and parameters into values:

        - 'nab_growth_decay' : based on Khoury et al. (https://www.nature.com/articles/s41591-021-01377-8)
        - 'nab_decay'   : specific decay function taken from https://doi.org/10.1101/2021.03.09.21252641
        - 'exp_decay'   : exponential decay. Parameters should be init_val and half_life (half_life can be None/nan)
        - 'linear_decay': linear decay

    A custom function can also be supplied.

    Args:
        length (float): length of array to return, i.e., for how long waning is calculated
        pars (dict): passed to individual immunity functions

    Returns:
        array of length 'length' of values
    '''

    pars = sc.dcp(pars)
    form = pars.pop('form')
    choices = [
        'nab_growth_decay', # Default if no form is provided
        'nab_decay',
        'exp_decay',
    ]

    # Process inputs
    if form is None or form == 'nab_growth_decay':
        output = nab_growth_decay(length, **pars)

    elif form == 'nab_decay':
        output = nab_decay(length, **pars)

    elif form == 'exp_decay':
        if pars['half_life'] is None: pars['half_life'] = np.nan
        output = exp_decay(length, **pars)

    elif callable(form):
        output = form(length, **pars)

    else:
        errormsg = f'The selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}'
        raise NotImplementedError(errormsg)

    return output


def nab_growth_decay(length, growth_time, decay_rate1, decay_time1, decay_rate2, decay_time2):
    '''
    Returns an array of length 'length' containing the evaluated function nab growth/decay
    function at each point.

    Uses linear growth + exponential decay, with the rate of exponential decay also set to
    decay linearly until it reaches a 10-year half life.

    Args:
        length (int): number of points
        growth_time (int): length of time NAbs grow (used to determine slope)
        decay_rate1 (float): initial rate of exponential decay
        decay_time1 (float): time of the first exponential decay
        decay_rate2 (float): the rate of exponential decay in late period
        decay_time2 (float): total time until late decay period (must be greater than decay_time1)
    '''


    def f1(t, growth_time):
        '''Simple linear growth'''
        return (1 / growth_time) * t

    def f2(t, decay_time1, decay_time2, decay_rate1, decay_rate2):
        decayRate = np.full(len(t), fill_value=decay_rate1)
        decayRate[cvu.true(t>decay_time2)] = decay_rate2
        slowing = (1 / (decay_time2 - decay_time1)) * (decay_rate1 - decay_rate2)
        decayRate[cvu.true((t>decay_time1)*(t<=decay_time2))] = decay_rate1 - slowing * (np.arange(len(cvu.true((t>decay_time1)*(t<=decay_time2))), dtype=cvd.default_int))
        titre = np.zeros(len(t))
        for i in range(1, len(t)):
            titre[i] = titre[i-1]+decayRate[i]
        return np.exp(-titre)

    if decay_time2 < decay_time1:
        errormsg = f'Decay time 2 must be larger than decay time 1, but you supplied {decay_time2} which is smaller than {decay_time1}.'
        raise ValueError(errormsg)

    length = length + 1
    t1 = np.arange(growth_time, dtype=cvd.default_int)
    t2 = np.arange(length - growth_time, dtype=cvd.default_int)
    y1 = f1(t1, growth_time)
    y2 = f2(t2, decay_time1, decay_time2, decay_rate1, decay_rate2)
    y  = np.concatenate([y1,y2])
    y = np.diff(y)[0:length]

    return y


def nab_decay(length, decay_rate1, decay_time1, decay_rate2):
    '''
    Returns an array of length 'length' containing the evaluated function nab decay
    function at each point.

    Uses exponential decay, with the rate of exponential decay also set to exponentially
    decay (!) after 250 days.

    Args:
        length (int): number of points
        decay_rate1 (float): initial rate of exponential decay
        decay_time1 (float): time on the first exponential decay
        decay_rate2 (float): the rate at which the decay decays
    '''
    def f1(t, decay_rate1):
        ''' Simple exponential decay '''
        return np.exp(-t*decay_rate1)

    def f2(t, decay_rate1, decay_time1, decay_rate2):
        ''' Complex exponential decay '''
        return np.exp(-t*(decay_rate1*np.exp(-(t-decay_time1)*decay_rate2)))

    t  = np.arange(length, dtype=cvd.default_int)
    y1 = f1(cvu.true(t<=decay_time1), decay_rate1)
    y2 = f2(cvu.true(t>decay_time1), decay_rate1, decay_time1, decay_rate2)
    y  = np.concatenate([[-np.inf],y1,y2])
    y = np.diff(y)[0:length]
    y[0] = 1
    return y


def exp_decay(length, init_val, half_life, delay=None):
    '''
    Returns an array of length t with values for the immunity at each time step after recovery
    '''
    length = length+1
    decay_rate = np.log(2) / half_life if ~np.isnan(half_life) else 0.
    if delay is not None:
        t = np.arange(length-delay, dtype=cvd.default_int)
        growth = linear_growth(delay, init_val/delay)
        decay = init_val * np.exp(-decay_rate * t)
        result = np.concatenate([growth, decay], axis=None)
    else:
        t = np.arange(length, dtype=cvd.default_int)
        result = init_val * np.exp(-decay_rate * t)
    return np.diff(result)


def linear_decay(length, init_val, slope):
    ''' Calculate linear decay '''
    result = -slope*np.ones(length)
    result[0] = init_val
    return result


def linear_growth(length, slope):
    ''' Calculate linear growth '''
    return slope*np.ones(length)
