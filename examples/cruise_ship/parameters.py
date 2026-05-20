'''
Set the parameters for COVID-ABM.
'''

import os
import pylab as pl
import pandas as pd
from datetime import datetime


__all__ = ['make_pars', 'get_age_sex', 'load_data']


def make_pars():
    ''' Set parameters for the simulation '''
    pars = {}

    # Simulation parameters
    pars['n_guests']   = 2666 # From https://www.princess.com/news/notices_and_advisories/notices/diamond-princess-update.html
    pars['n_crew']     = 1045 # Ditto
    pars['day_0']      = datetime(2020, 1, 22) # Start day of the epidemic
    pars['n_days']     = 32 # How many days to simulate -- 31 days is until 2020-Feb-20
    pars['rand_seed']  = 1 # Random seed, if None, don't reset
    pars['verbose']    = 1 # Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)

    # Epidemic parameters
    pars['r_contact']      = 0.05 # Probability of infection per contact, estimated
    pars['contacts_guest'] = 30 # Number of contacts per guest per day, estimated
    pars['contacts_crew']  = 30 # Number of contacts per crew member per day
    pars['incub']          = 4.0 # Using Mike's Snohomish number
    pars['incub_std']      = 1.0 # Standard deviation of the serial interval, estimated
    pars['dur']            = 8 # Using Mike's Snohomish number
    pars['dur_std']        = 2 # Variance in duration
    pars['sensitivity']    = 1.0 # Probability of a true positive, estimated
    pars['symptomatic']    = 5 # Increased probability of testing someone symptomatic, estimated

    # Events
    pars['quarantine']       = 15  # Day on which quarantine took effect
    pars['quarantine_eff']   = 0.10 # Change in transmissibility for guests due to quarantine, estimated
    # pars['quarantine_eff_g'] = 0.10 # Change in transmissibility for guests due to quarantine, estimated
    # pars['quarantine_eff_c'] = 0.15 # Change in transmissibility for crew due to quarantine, estimated
    pars['testing_change']   = 23  # Day that testing protocol changed (TODO: double-check), from https://hopkinsidd.github.io/nCoV-Sandbox/Diamond-Princess.html
    pars['testing_symptoms'] = 0.1 # Relative change in symptomatic testing after the protocol change
    pars['evac_positives']   = 1  # If people who test positive are removed from the ship (Boolean)

    return pars


def get_age_sex(is_crew=False, min_age=18, max_age=99, crew_age=35, crew_std=5, guest_age=68, guest_std=8):
    '''
    Define age-sex distributions. Passenger age distribution based on:

    https://www.nytimes.com/reuters/2020/02/12/world/asia/12reuters-china-health-japan.html

    "About 80% of the passengers were aged 60 or over [=2130], with 215 in their 80s and 11 in the 90s,
    the English-language Japan Times newspaper reported."
    '''

    # Define female (0) or male (1) -- evenly distributed
    sex = pl.randint(2)

    # Define age distribution for the crew and guests
    if is_crew:
        age = pl.normal(crew_age, crew_std)
    else:
        age = pl.normal(guest_age, guest_std)

    # Normalize
    age = pl.median([min_age, age, max_age])

    return age, sex


def load_data(filename=None):
    ''' Load data for comparing to the model output '''

    default_datafile = 'reported_infections.xlsx'

    # Handle default filename
    if filename is None:
        cwd = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(cwd, default_datafile)

    # Load data
    raw_data = pd.read_excel(filename)

    # Confirm data integrity and simplify
    cols = ['day', 'date', 'new_tests', 'new_positives', 'confirmed_crew', 'confirmed_guests', 'evacuated', 'evacuated_positives']
    data = pd.DataFrame()
    for col in cols:
        assert col in raw_data.columns, f'Column "{col}" is missing from the loaded data'
    data = raw_data[cols]

    return data



