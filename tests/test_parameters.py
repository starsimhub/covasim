'''
Test that the parameters and data files are being created correctly.
'''

#%% Imports
import os
import pytest
import sciris as sc
import covasim as cv


#%% Define the tests
def test_parameters():
    sc.heading('Model parameters')
    pars = cv.make_pars()
    sc.pp(pars)
    return pars


def test_data():
    sc.heading('Data loading')
    data = cv.load_data(os.path.join(sc.thisdir(__file__), 'example_data.csv'))
    sc.pp(data)

    # Check that it is looking for the right file
    with pytest.raises(FileNotFoundError):
        data = cv.load_data(datafile='file_not_found.csv')

    return data


def test_location():
    sc.heading('Population settings')

    pars = dict(
        pop_size = 1000,
        pop_type = 'hybrid',
        location = 'nigeria',
        )
    sim = cv.Sim(pars)
    sim.initialize()

    return sim


def test_age_structure():
    sc.heading('Age structures')

    available     = 'Lithuania'
    not_available = 'Ruritania'

    age_data = cv.data.loaders.get_age_distribution(available)

    with pytest.raises(ValueError):
        cv.data.loaders.get_age_distribution(not_available)

    return age_data


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    pars = test_parameters()
    data = test_data()
    sim  = test_location()
    age  = test_age_structure()

    sc.toc(T)
    print('Done.')
