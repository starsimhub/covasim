"""
Load data
"""

#%% Housekeeping
import numpy as np
import sciris as sc
from . import country_age_data    as cad
from . import state_age_data      as sad
from . import household_size_data as hsd

__all__ = ['get_country_aliases', 'map_entries', 'show_locations', 'get_age_distribution', 'get_household_size']


def get_country_aliases():
    """
    Define aliases for countries with odd names in the data. Each alias maps to
    the name used in the age data; the household size data use some different
    spellings, which are also listed here as aliases.
    """
    country_mappings = {
       'Bolivia':        'Bolivia (Plurinational State of)',
       'Burkina':        'Burkina Faso',
       'Cape Verde':     'Cabo Verde',
       'Hong Kong':      'China, Hong Kong Special Administrative Region',
       'Macao':          'China, Macao Special Administrative Region',
       "Cote d'Ivore":   'Côte d’Ivoire',
       "Cote d'Ivoire":  'Côte d’Ivoire',
       "Ivory Coast":    'Côte d’Ivoire',
       'DRC':            'Democratic Republic of the Congo',
       'Iran':           'Iran (Islamic Republic of)',
       'Laos':           "Lao People's Democratic Republic",
       'Micronesia':     'Micronesia (Federated States of)',
       'Korea':          'Republic of Korea',
       'South Korea':    'Republic of Korea',
       'Moldova':        'Republic of Moldova',
       'Russia':         'Russian Federation',
       'Palestine':      'State of Palestine',
       'Syria':          'Syrian Arab Republic',
       'Taiwan':         'Taiwan Province of China',
       'Macedonia':      'The former Yugoslav Republic of Macedonia',
       'UK':             'United Kingdom of Great Britain and Northern Ireland',
       'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',
       'Tanzania':       'United Republic of Tanzania',
       'USA':            'United States of America',
       'United States':  'United States of America',
       'Venezuela':      'Venezuela (Bolivarian Republic of)',
       'Vietnam':        'Viet Nam',

       # Spellings used in the household size data
       'China, Hong Kong SAR':        'China, Hong Kong Special Administrative Region',
       'China, Macao SAR':            'China, Macao Special Administrative Region',
       "Côte d'Ivoire":               'Côte d’Ivoire',
       "Dem. People's Rep. of Korea": "Democratic People's Republic of Korea",
       'Dem. Republic of the Congo':  'Democratic Republic of the Congo',
       "Lao People's Dem. Republic":  "Lao People's Democratic Republic",
       'Swaziland':                   'Eswatini',
        }

    return country_mappings


def map_entries(data, location):
    """
    Find a match between the data and the provided location(s). Names are
    case-insensitive, and any alias from get_country_aliases() can be used for
    any spelling of the same country.

    Args:
        data (dict): the data being loaded
        location (list or str): the list of locations to pull from
    """
    countries = [key.lower() for key in data.keys()]
    values = list(data.values())

    # Set parameters
    if location is None:
        location = countries
    else:
        location = sc.promotetolist(location)

    # Define a mapping for common mistakes
    mapping = get_country_aliases()
    mapping = {key.lower(): val.lower() for key, val in mapping.items()}

    entries = {}
    for loc in location:

        # Find all the equivalent names: the name itself, the name it is an alias for, and the other aliases of that name
        lloc = loc.lower()
        target = mapping.get(lloc, lloc)
        names = [lloc, target]
        for key,val in mapping.items():
            if val == target:
                names.append(key)

        # Use the first name that is in the data
        match = None
        for name in names:
            if name in countries:
                match = name
                break
        if match is None:
            suggestions = sc.suggest(loc, countries, n=4)
            if suggestions:
                errormsg = f'Location "{loc}" not recognized, did you mean {suggestions}?'
            else:
                errormsg = f'Location "{loc}" not recognized'
            raise ValueError(errormsg)
        entries[loc] = values[countries.index(match)]

    return entries


def show_locations(location=None, output=False):
    """
    Print a list of available locations.

    Args:
        location (str): if provided, only check if this location is in the list
        output (bool): whether to return the list (else print); with a location, return whether the age and household size data are available

    **Examples**::

        cv.data.show_locations() # Print a list of valid locations
        cv.data.show_locations('lithuania') # Check if Lithuania is a valid location
        cv.data.show_locations('Viet-Nam') # Check if Viet-Nam is a valid location
    """
    age_data = sc.mergedicts(sad.data, cad.data) # Countries will overwrite states, e.g. Georgia
    aliases  = get_country_aliases()

    loclist = sc.objdict()
    loclist.age_distributions = sorted(list(age_data.keys()) + list(aliases.keys()))
    loclist.household_size_distributions = sorted(list(hsd.data.keys()))

    if location is not None:

        # Check availability using the same lookup as loading the data
        age_available = True
        try:
            map_entries(age_data, location)
        except ValueError:
            age_available = False
        hh_available = True
        try:
            map_entries(hsd.data, location)
        except ValueError:
            hh_available = False

        # Suggest alternatives if not available
        age_sugg = ''
        if not age_available:
            age_sugg = f'(closest match: {sc.suggest(location, loclist.age_distributions)})'
        hh_sugg = ''
        if not hh_available:
            hh_sugg = f'(closest match: {sc.suggest(location, loclist.household_size_distributions)})'

        if output:
            return age_available, hh_available
        print(f'For location "{location}":')
        print(f'  Population age distribution is available: {age_available} {age_sugg}')
        print(f'  Household size distribution is available: {hh_available} {hh_sugg}')
        return

    if output:
        return loclist
    else:
        print(f'There are {len(loclist.age_distributions)} age distributions and {len(loclist.household_size_distributions)} household size distributions.')
        print('\nList of available locations (case insensitive):\n')
        sc.pp(loclist)
        return


def get_age_distribution(location=None):
    """
    Load age distribution for a given country or countries.

    Args:
        location (str or list): name of the country or countries to load the age distribution for

    Returns:
        age_data (array): Numpy array of age distributions, or dict if multiple locations
    """

    # Load the raw data
    data = sc.mergedicts(sad.data, cad.data) # Countries will overwrite states, e.g. Georgia
    entries = map_entries(data, location)

    max_age = 99
    result = {}
    for loc,age_distribution in entries.items():
        total_pop = sum(list(age_distribution.values()))
        local_pop = []

        for age, age_pop in age_distribution.items():
            if age[-1] == '+':
                val = [int(age[:-1]), max_age, age_pop/total_pop]
            else:
                ages = age.split('-')
                val = [int(ages[0]), int(ages[1]), age_pop/total_pop]
            local_pop.append(val)
        result[loc] = np.array(local_pop)

    if len(result) == 1:
        result = list(result.values())[0]

    return result


def get_household_size(location=None):
    """
    Load household size distribution for a given country or countries.

    Args:
        location (str or list): name of the country or countries to load the household size distribution for

    Returns:
        house_size (float): Size of household, or dict if multiple locations
    """
    # Load the raw data
    result = map_entries(hsd.data, location)
    if len(result) == 1:
        result = list(result.values())[0]

    return result
