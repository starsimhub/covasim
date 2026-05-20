'''
Numerical utilities for running Covasim.

These include the viral load, transmissibility, and infection calculations
at the heart of the integration loop.
'''

#%% Housekeeping

import numba as nb # For faster computations
import numpy as np # For numerics
import random # Used only for resetting the seed
import sciris as sc # For additional utilities
from .settings import options as cvo # To set options
from . import defaults as cvd # To set default types


# What functions are externally visible -- note, this gets populated in each section below
__all__ = []

# Set dtypes -- note, these cannot be changed after import since Numba functions are precompiled
nbbool  = nb.bool_
nbint   = cvd.nbint
nbfloat = cvd.nbfloat

# Specify whether to allow parallel Numba calculation -- 10% faster for safe and 20% faster for random, but the random number stream becomes nondeterministic for the latter
safe_opts = [1, '1', 'safe']
full_opts = [2, '2', 'full']
safe_parallel = cvo.numba_parallel in safe_opts + full_opts
rand_parallel = cvo.numba_parallel in full_opts
if cvo.numba_parallel not in [0, 1, 2, '0', '1', '2', 'none', 'safe', 'full']:
    errormsg = f'Numba parallel must be "none", "safe", or "full", not "{cvo.numba_parallel}"'
    raise ValueError(errormsg)
cache = cvo.numba_cache # Turning this off can help switching parallelization options


#%% The core Covasim functions -- compute the infections

@nb.njit(             (nbint, nbfloat[:], nbfloat[:],     nbfloat[:], nbfloat,   nbfloat,    nbfloat), cache=cache, parallel=safe_parallel)
def compute_viral_load(t,     time_start, time_recovered, time_dead,  frac_time, load_ratio, high_cap): # pragma: no cover
    '''
    Calculate relative transmissibility for time t. Includes time varying
    viral load, pre/asymptomatic factor, diagnosis factor, etc.

    Args:
        t: (int) timestep
        time_start: (float[]) individuals' infectious date
        time_recovered: (float[]) individuals' recovered date
        time_dead: (float[]) individuals' death date
        frac_time: (float) fraction of time in high load
        load_ratio: (float) ratio for high to low viral load
        high_cap: (float) cap on the number of days with high viral load

    Returns:
        load (float): viral load
    '''

    # Get the end date from recover or death
    n = len(time_dead)
    time_stop = np.ones(n, dtype=cvd.default_float)*time_recovered # This is needed to make a copy
    inds = ~np.isnan(time_dead)
    time_stop[inds] = time_dead[inds]

    # Calculate which individuals with be high past the cap and when it should happen
    infect_days_total = time_stop-time_start
    trans_day = frac_time*infect_days_total
    inds = trans_day > high_cap
    cap_frac = high_cap/infect_days_total[inds]

    # Get corrected time to switch from high to low
    trans_point = np.ones(n,dtype=cvd.default_float)*frac_time
    trans_point[inds] = cap_frac

    # Calculate load
    load = np.ones(n, dtype=cvd.default_float) # allocate an array of ones with the correct dtype
    early = (t-time_start)/infect_days_total < trans_point # are we in the early or late phase
    load = (load_ratio * early + load * ~early)/(load+frac_time*(load_ratio-load)) # calculate load

    return load


@nb.njit(            (nbfloat[:], nbfloat[:], nbbool[:], nbbool[:], nbfloat,    nbfloat[:], nbbool[:], nbbool[:], nbbool[:], nbfloat,      nbfloat,    nbfloat,     nbfloat[:]), cache=cache, parallel=safe_parallel)
def compute_trans_sus(rel_trans,  rel_sus,    inf,       sus,       beta_layer, viral_load, symp,      iso,      quar,      asymp_factor, iso_factor, quar_factor, immunity_factors): # pragma: no cover
    ''' Calculate relative transmissibility and susceptibility '''
    f_asymp   =  symp + ~symp * asymp_factor # Asymptomatic factor, changes e.g. [0,1] with a factor of 0.8 to [0.8,1.0]
    f_iso     = ~iso  +  iso  * iso_factor # Isolation factor, changes e.g. [0,1] with a factor of 0.2 to [1,0.2]
    f_quar    = ~quar +  quar * quar_factor # Quarantine, changes e.g. [0,1] with a factor of 0.5 to [1,0.5]
    rel_trans = rel_trans * inf * f_quar * f_asymp * f_iso * beta_layer * viral_load # Recalculate transmissibility
    rel_sus   = rel_sus * sus * f_quar * (1-immunity_factors) # Recalculate susceptibility
    return rel_trans, rel_sus


@nb.njit(             (nbfloat,  nbint[:],  nbint[:], nbfloat[:],   nbfloat[:], nbfloat[:], nbbool), cache=cache, parallel=rand_parallel)
def compute_infections(beta,     p1,        p2,       layer_betas,  rel_trans,  rel_sus,    legacy=False): # pragma: no cover
    '''
    Compute who infects whom

    The heaviest step of the model -- figure out who gets infected on this timestep.
    Cannot be easily parallelized since random numbers are used. Loops over contacts
    in both directions (i.e., targets become sources).

    Args:
        beta: overall transmissibility
        p1: person 1
        p2: person 2
        layer_betas: per-contact transmissibilities
        rel_trans: the source's relative transmissibility
        rel_sus: the target's relative susceptibility
        legacy: whether to use the slower legacy (pre 3.1.1) calculation method
    '''
    slist = np.empty(0, dtype=nbint)
    tlist = np.empty(0, dtype=nbint)
    pairs = [[p1,p2], [p2,p1]] if not legacy else [[p1,p2]]
    for sources,targets in pairs:
        source_trans     = rel_trans[sources] # Pull out the transmissibility of the sources (0 for non-infectious people)
        inf_inds         = source_trans.nonzero()[0] # Infectious indices -- remove noninfectious people
        betas            = beta * layer_betas[inf_inds] * source_trans[inf_inds] * rel_sus[targets[inf_inds]] # Calculate the raw transmission probabilities
        nonzero_inds     = betas.nonzero()[0] # Find nonzero entries
        nonzero_inf_inds = inf_inds[nonzero_inds] # Map onto original indices
        nonzero_betas    = betas[nonzero_inds] # Remove zero entries from beta
        nonzero_sources  = sources[nonzero_inf_inds] # Remove zero entries from the sources
        nonzero_targets  = targets[nonzero_inf_inds] # Remove zero entries from the targets
        transmissions    = (np.random.random(len(nonzero_betas)) < nonzero_betas).nonzero()[0] # Compute the actual infections!
        source_inds      = nonzero_sources[transmissions]
        target_inds      = nonzero_targets[transmissions] # Filter the targets on the actual infections
        slist = np.concatenate((slist, source_inds), axis=0)
        tlist = np.concatenate((tlist, target_inds), axis=0)
    return slist, tlist


@nb.njit((nbint[:], nbint[:], nb.int64[:]), cache=cache)
def find_contacts(p1, p2, inds): # pragma: no cover
    """
    Numba for Layer.find_contacts()

    A set is returned here rather than a sorted array so that custom tracing interventions can efficiently
    add extra people. For a version with sorting by default, see Layer.find_contacts(). Indices must be
    an int64 array since this is what's returned by true() etc. functions by default.
    """
    pairing_partners = set()
    inds = set(inds)
    for i in range(len(p1)):
        if p1[i] in inds:
            pairing_partners.add(p2[i])
        if p2[i] in inds:
            pairing_partners.add(p1[i])
    return pairing_partners



#%% Sampling and seed methods

__all__ += ['sample', 'get_pdf', 'set_seed']


def sample(dist=None, par1=None, par2=None, size=None, **kwargs):
    '''
    Draw a sample from the distribution specified by the input. The available
    distributions are:

    - 'uniform'       : uniform distribution from low=par1 to high=par2; mean is equal to (par1+par2)/2
    - 'normal'        : normal distribution with mean=par1 and std=par2
    - 'lognormal'     : lognormal distribution with mean=par1 and std=par2 (parameters are for the lognormal distribution, *not* the underlying normal distribution)
    - 'normal_pos'    : right-sided normal distribution (i.e. only positive values), with mean=par1 and std=par2 *of the underlying normal distribution*
    - 'normal_int'    : normal distribution with mean=par1 and std=par2, returns only integer values
    - 'lognormal_int' : lognormal distribution with mean=par1 and std=par2, returns only integer values
    - 'poisson'       : Poisson distribution with rate=par1 (par2 is not used); mean and variance are equal to par1
    - 'neg_binomial'  : negative binomial distribution with mean=par1 and k=par2; converges to Poisson with k=∞

    Args:
        dist (str):   the distribution to sample from
        par1 (float): the "main" distribution parameter (e.g. mean)
        par2 (float): the "secondary" distribution parameter (e.g. std)
        size (int):   the number of samples (default=1)
        kwargs (dict): passed to individual sampling functions

    Returns:
        A length N array of samples

    **Examples**::

        cv.sample() # returns Unif(0,1)
        cv.sample(dist='normal', par1=3, par2=0.5) # returns Normal(μ=3, σ=0.5)
        cv.sample(dist='lognormal_int', par1=5, par2=3) # returns a lognormally distributed set of values with mean 5 and std 3

    Notes:
        Lognormal distributions are parameterized with reference to the underlying normal distribution (see:
        https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.lognormal.html), but this
        function assumes the user wants to specify the mean and std of the lognormal distribution.

        Negative binomial distributions are parameterized with reference to the mean and dispersion parameter k
        (see: https://en.wikipedia.org/wiki/Negative_binomial_distribution). The r parameter of the underlying
        distribution is then calculated from the desired mean and k. For a small mean (~1), a dispersion parameter
        of ∞ corresponds to the variance and standard deviation being equal to the mean (i.e., Poisson). For a
        large mean (e.g. >100), a dispersion parameter of 1 corresponds to the standard deviation being equal to
        the mean.
    '''

    # Some of these have aliases, but these are the "official" names
    choices = [
        'uniform',
        'normal',
        'normal_pos',
        'normal_int',
        'lognormal',
        'lognormal_int',
        'poisson',
        'neg_binomial',
    ]

    # Ensure it's an integer
    if size is not None:
        size = int(size)

    # Compute distribution parameters and draw samples
    # NB, if adding a new distribution, also add to choices above
    if   dist in ['unif', 'uniform']: samples = np.random.uniform(low=par1, high=par2, size=size, **kwargs)
    elif dist in ['norm', 'normal']:  samples = np.random.normal(loc=par1, scale=par2, size=size, **kwargs)
    elif dist == 'normal_pos':        samples = np.abs(np.random.normal(loc=par1, scale=par2, size=size, **kwargs))
    elif dist == 'normal_int':        samples = np.round(np.abs(np.random.normal(loc=par1, scale=par2, size=size, **kwargs)))
    elif dist == 'poisson':           samples = n_poisson(rate=par1, n=size, **kwargs) # Use Numba version below for speed
    elif dist == 'neg_binomial':      samples = n_neg_binomial(rate=par1, dispersion=par2, n=size, **kwargs) # Use custom version below
    elif dist in ['lognorm', 'lognormal', 'lognorm_int', 'lognormal_int']:
        if par1>0:
            mean  = np.log(par1**2 / np.sqrt(par2**2 + par1**2)) # Computes the mean of the underlying normal distribution
            sigma = np.sqrt(np.log(par2**2/par1**2 + 1)) # Computes sigma for the underlying normal distribution
            samples = np.random.lognormal(mean=mean, sigma=sigma, size=size, **kwargs)
        else:
            samples = np.zeros(size)
        if '_int' in dist:
            samples = np.round(samples)
    else:
        errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {sc.newlinejoin(choices)}'
        raise NotImplementedError(errormsg)

    return samples



def get_pdf(dist=None, par1=None, par2=None):
    '''
    Return a probability density function for the specified distribution. This
    is used for example by test_num to retrieve the distribution of times from
    symptom-to-swab for testing. For example, for Washington State, these values
    are dist='lognormal', par1=10, par2=170.
    '''
    import scipy.stats as sps # Import here since slow

    choices = [
        'none',
        'uniform',
        'lognormal',
    ]

    if dist in ['None', 'none', None]:
        return None
    elif dist == 'uniform':
        pdf = sps.uniform(loc=par1, scale=par2)
    elif dist == 'lognormal':
        mean  = np.log(par1**2 / np.sqrt(par2 + par1**2)) # Computes the mean of the underlying normal distribution
        sigma = np.sqrt(np.log(par2/par1**2 + 1)) # Computes sigma for the underlying normal distribution
        pdf   = sps.lognorm(sigma, loc=-0.5, scale=np.exp(mean))
    else:
        choicestr = '\n'.join(choices)
        errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {choicestr}'
        raise NotImplementedError(errormsg)

    return pdf


def set_seed(seed=None):
    '''
    Reset the random seed -- complicated because of Numba, which requires special
    syntax to reset the seed. This function also resets Python's built-in random
    number generated.

    Args:
        seed (int): the random seed
    '''

    @nb.njit((nbint,), cache=cache)
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    # Dies if a float is given
    if seed is not None:
        seed = int(seed)

    set_seed_regular(seed) # If None, reinitializes it
    if seed is None: # Numba can't accept a None seed, so use our just-reinitialized Numpy stream to generate one
        seed = np.random.randint(1e9)
    set_seed_numba(seed)
    random.seed(seed) # Finally, reset Python's built-in random number generator, just in case (used by SynthPops)

    return


#%% Probabilities -- mostly not jitted since performance gain is minimal

__all__ += ['n_binomial', 'binomial_filter', 'binomial_arr', 'n_multinomial',
            'poisson', 'n_poisson', 'n_neg_binomial', 'choose', 'choose_r', 'choose_w']

def n_binomial(prob, n):
    '''
    Perform multiple binomial (Bernolli) trials

    Args:
        prob (float): probability of each trial succeeding
        n (int): number of trials (size of array)

    Returns:
        Boolean array of which trials succeeded

    **Example**::

        outcomes = cv.n_binomial(0.5, 100) # Perform 100 coin-flips
    '''
    return np.random.random(n) < prob


def binomial_filter(prob, arr): # No speed gain from Numba
    '''
    Binomial "filter" -- the same as n_binomial, except return
    the elements of arr that succeeded.

    Args:
        prob (float): probability of each trial succeeding
        arr (array): the array to be filtered

    Returns:
        Subset of array for which trials succeeded

    **Example**::

        inds = cv.binomial_filter(0.5, np.arange(20)**2) # Return which values out of the (arbitrary) array passed the coin flip
    '''
    return arr[(np.random.random(len(arr)) < prob).nonzero()[0]]


def binomial_arr(prob_arr):
    '''
    Binomial (Bernoulli) trials each with different probabilities.

    Args:
        prob_arr (array): array of probabilities

    Returns:
         Boolean array of which trials on the input array succeeded

    **Example**::

        outcomes = cv.binomial_arr([0.1, 0.1, 0.2, 0.2, 0.8, 0.8]) # Perform 6 trials with different probabilities
    '''
    return np.random.random(len(prob_arr)) < prob_arr


def n_multinomial(probs, n): # No speed gain from Numba
    '''
    An array of multinomial trials.

    Args:
        probs (array): probability of each outcome, which usually should sum to 1
        n (int): number of trials

    Returns:
        Array of integer outcomes

    **Example**::

        outcomes = cv.multinomial(np.ones(6)/6.0, 50)+1 # Return 50 die-rolls
    '''
    return np.searchsorted(np.cumsum(probs), np.random.random(n))


@nb.njit((nbfloat,), cache=cache, parallel=rand_parallel) # Numba hugely increases performance
def poisson(rate):
    '''
    A Poisson trial.

    Args:
        rate (float): the rate of the Poisson process

    **Example**::

        outcome = cv.poisson(100) # Single Poisson trial with mean 100
    '''
    return np.random.poisson(rate, 1)[0]


@nb.njit((nbfloat, nbint), cache=cache, parallel=rand_parallel) # Numba hugely increases performance
def n_poisson(rate, n):
    '''
    An array of Poisson trials.

    Args:
        rate (float): the rate of the Poisson process (mean)
        n (int): number of trials

    **Example**::

        outcomes = cv.n_poisson(100, 20) # 20 Poisson trials with mean 100
    '''
    return np.random.poisson(rate, n)


def n_neg_binomial(rate, dispersion, n, step=1): # Numba not used due to incompatible implementation
    '''
    An array of negative binomial trials. See cv.sample() for more explanation.

    Args:
        rate (float): the rate of the process (mean, same as Poisson)
        dispersion (float):  dispersion parameter; lower is more dispersion, i.e. 0 = infinite, ∞ = Poisson
        n (int): number of trials
        step (float): the step size to use if non-integer outputs are desired

    **Example**::

        outcomes = cv.n_neg_binomial(100, 1, 50) # 50 negative binomial trials with mean 100 and dispersion roughly equal to mean (large-mean limit)
        outcomes = cv.n_neg_binomial(1, 100, 20) # 20 negative binomial trials with mean 1 and dispersion still roughly equal to mean (approximately Poisson)
    '''
    nbn_n = dispersion
    nbn_p = dispersion/(rate/step + dispersion)
    samples = np.random.negative_binomial(n=nbn_n, p=nbn_p, size=n)*step
    return samples


@nb.njit((nbint, nbint), cache=cache) # Numba hugely increases performance
def choose(max_n, n):
    '''
    Choose a subset of items (e.g., people) without replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    **Example**::

        choices = cv.choose(5, 2) # choose 2 out of 5 people with equal probability (without repeats)
    '''
    return np.random.choice(max_n, n, replace=False)


@nb.njit((nbint, nbint), cache=cache) # Numba hugely increases performance
def choose_r(max_n, n):
    '''
    Choose a subset of items (e.g., people), with replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    **Example**::

        choices = cv.choose_r(5, 10) # choose 10 out of 5 people with equal probability (with repeats)
    '''
    return np.random.choice(max_n, n, replace=True)


def choose_w(probs, n, unique=True): # No performance gain from Numba
    '''
    Choose n items (e.g. people), each with a probability from the distribution probs.

    Args:
        probs (array): list of probabilities, should sum to 1
        n (int): number of samples to choose
        unique (bool): whether or not to ensure unique indices

    **Example**::

        choices = cv.choose_w([0.2, 0.5, 0.1, 0.1, 0.1], 2) # choose 2 out of 5 people with nonequal probability.
    '''
    probs = np.array(probs)
    n_choices = len(probs)
    n_samples = int(n)
    probs_sum = probs.sum()
    if probs_sum: # Weight is nonzero, rescale
        probs = probs/probs_sum
    else: # Weights are all zero, choose uniformly
        probs = np.ones(n_choices)/n_choices
    return np.random.choice(n_choices, n_samples, p=probs, replace=not(unique))



#%% Simple array operations

__all__ += ['true',   'false',   'defined',   'undefined',
            'itrue',  'ifalse',  'idefined',  'iundefined',
            'itruei', 'ifalsei', 'idefinedi', 'iundefinedi']


def true(arr):
    '''
    Returns the indices of the values of the array that are true: just an alias
    for arr.nonzero()[0].

    Args:
        arr (array): any array

    **Example**::

        inds = cv.true(np.array([1,0,0,1,1,0,1])) # Returns array([0, 3, 4, 6])
    '''
    return arr.nonzero()[0]


def false(arr):
    '''
    Returns the indices of the values of the array that are false.

    Args:
        arr (array): any array

    **Example**::

        inds = cv.false(np.array([1,0,0,1,1,0,1]))
    '''
    return np.logical_not(arr).nonzero()[0]


def defined(arr):
    '''
    Returns the indices of the values of the array that are not-nan.

    Args:
        arr (array): any array

    **Example**::

        inds = cv.defined(np.array([1,np.nan,0,np.nan,1,0,1]))
    '''
    return (~np.isnan(arr)).nonzero()[0]


def undefined(arr):
    '''
    Returns the indices of the values of the array that are not-nan.

    Args:
        arr (array): any array

    **Example**::

        inds = cv.defined(np.array([1,np.nan,0,np.nan,1,0,1]))
    '''
    return np.isnan(arr).nonzero()[0]


def itrue(arr, inds):
    '''
    Returns the indices that are true in the array -- name is short for indices[true]

    Args:
        arr (array): a Boolean array, used as a filter
        inds (array): any other array (usually, an array of indices) of the same size

    **Example**::

        inds = cv.itrue(np.array([True,False,True,True]), inds=np.array([5,22,47,93]))
    '''
    return inds[arr]


def ifalse(arr, inds):
    '''
    Returns the indices that are true in the array -- name is short for indices[false]

    Args:
        arr (array): a Boolean array, used as a filter
        inds (array): any other array (usually, an array of indices) of the same size

    **Example**::

        inds = cv.ifalse(np.array([True,False,True,True]), inds=np.array([5,22,47,93]))
    '''
    return inds[np.logical_not(arr)]


def idefined(arr, inds):
    '''
    Returns the indices that are defined in the array -- name is short for indices[defined]

    Args:
        arr (array): any array, used as a filter
        inds (array): any other array (usually, an array of indices) of the same size

    **Example**::

        inds = cv.idefined(np.array([3,np.nan,np.nan,4]), inds=np.array([5,22,47,93]))
    '''
    return inds[~np.isnan(arr)]


def iundefined(arr, inds):
    '''
    Returns the indices that are undefined in the array -- name is short for indices[undefined]

    Args:
        arr (array): any array, used as a filter
        inds (array): any other array (usually, an array of indices) of the same size

    **Example**::

        inds = cv.iundefined(np.array([3,np.nan,np.nan,4]), inds=np.array([5,22,47,93]))
    '''
    return inds[np.isnan(arr)]



def itruei(arr, inds):
    '''
    Returns the indices that are true in the array -- name is short for indices[true[indices]]

    Args:
        arr (array): a Boolean array, used as a filter
        inds (array): an array of indices for the original array

    **Example**::

        inds = cv.itruei(np.array([True,False,True,True,False,False,True,False]), inds=np.array([0,1,3,5]))
    '''
    return inds[arr[inds]]


def ifalsei(arr, inds):
    '''
    Returns the indices that are false in the array -- name is short for indices[false[indices]]

    Args:
        arr (array): a Boolean array, used as a filter
        inds (array): an array of indices for the original array

    **Example**::

        inds = cv.ifalsei(np.array([True,False,True,True,False,False,True,False]), inds=np.array([0,1,3,5]))
    '''
    return inds[np.logical_not(arr[inds])]


def idefinedi(arr, inds):
    '''
    Returns the indices that are defined in the array -- name is short for indices[defined[indices]]

    Args:
        arr (array): any array, used as a filter
        inds (array): an array of indices for the original array

    **Example**::

        inds = cv.idefinedi(np.array([4,np.nan,0,np.nan,np.nan,4,7,4,np.nan]), inds=np.array([0,1,3,5]))
    '''
    return inds[~np.isnan(arr[inds])]


def iundefinedi(arr, inds):
    '''
    Returns the indices that are undefined in the array -- name is short for indices[defined[indices]]

    Args:
        arr (array): any array, used as a filter
        inds (array): an array of indices for the original array

    **Example**::

        inds = cv.iundefinedi(np.array([4,np.nan,0,np.nan,np.nan,4,7,4,np.nan]), inds=np.array([0,1,3,5]))
    '''
    return inds[np.isnan(arr[inds])]
