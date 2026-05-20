'''
Test that the parameters and data files are being created correctly.
'''

#%% Imports
import pytest
import pylab as pl
import sciris as sc
import cruise_ship as cova # NOTE: this is the only tests script that doesn't use base

do_plot = False


#%% Define the tests
def test_parameters():
    sc.heading('Model parameters')
    pars = cova.make_pars()
    sc.pp(pars)
    return pars


def test_age_sex(do_plot=False):
    sc.heading('Guest/crew ages')

    # Set up
    pars = cova.make_pars()
    ages = {}
    keys = ['guests', 'crew']
    for key in keys:
        ages[key] = []

    # Create the distributions
    for i in range(pars['n_guests']):
        age,sex = cova.get_age_sex(is_crew=False)
        ages['guests'].append(age)

    for i in range(pars['n_crew']):
        age,sex = cova.get_age_sex(is_crew=True)
        ages['crew'].append(age)

    for key in keys:
        ages[key] = pl.array(ages[key])

    age_bins = [60, 80, 90]
    reported = [2130, 226, 11]

    simulated = []
    for age_bin in age_bins:
        simulated.append(sum(ages['guests']>=age_bin))

    print(f'''Reported ages:
    >{age_bins[0]} = {reported[0]}
    >{age_bins[1]} = {reported[1]}
    >{age_bins[2]} = {reported[2]}''')

    print(f'''Simulated ages:
    >{age_bins[0]} = {simulated[0]}
    >{age_bins[1]} = {simulated[1]}
    >{age_bins[2]} = {simulated[2]}''')

    if do_plot:
        pl.figure(figsize=(18,12))
        for i,key in enumerate(keys):
            pl.subplot(2,1,i+1)
            pl.hist(ages[key], bins=30)
            pl.xlim([0,100])
            pl.title(f'Age distribution for {key}')
            pl.xlabel('Age')
            pl.ylabel('Count')

    return ages


def test_data():
    sc.heading('Data loading')
    data = cova.load_data()
    sc.pp(data)

    # Check that it is looking for the right file
    with pytest.raises(FileNotFoundError):
        data = cova.load_data(filename='file_not_found.csv')

    return data


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    pars = test_parameters()
    data = test_data()
    ages = test_age_sex(do_plot=do_plot)

    sc.toc()


print('Done.')
