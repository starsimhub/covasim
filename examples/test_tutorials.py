#!/usr/bin/env python3
'''
Run the tutorial examples
'''

import os
import covasim as cv
import sciris as sc
import pickle
import test_examples as tex

def test_all_tutorials():

    # Get and run tests
    filenames = sc.getfilelist(tex.examples_dir, pattern='t*.py', nopath=True)
    for filename in filenames:
        if filename[1] in '0123456789': # Should have format e.g. t05_foo.py, not test_foo.py
            sc.heading(f'Running {filename}...')
            try:
                tex.run_example(filename)
            except (pickle.PicklingError, NameError) as E: # Ignore these: issue with how the modules are loaded in the run_example function
                print(f'Skipping {filename} due to known pickling error: {E}')
        else:
            print(f'[Skipping "{filename}" since does not match pattern]')

    # Tidy up
    testfiles = sc.getfilelist(tex.examples_dir, pattern='my-*.*')

    sc.heading('Tidying...')
    print(f'Deleting:')
    for filename in testfiles:
        print(f'  {filename}')
    print('in 3 seconds...')
    sc.timedsleep(3)
    for filename in testfiles:
        os.remove(filename)
        print(f'  Deleted {filename}')

    return


#%% Run as a script
if __name__ == '__main__':

    cv.options.set(verbose=0)
    T = sc.tic()

    test_all_tutorials()

    sc.toc(T)
    print('Done.')
