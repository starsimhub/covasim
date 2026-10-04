'''
Check that correct versions of each library are installed, and print warnings
or errors if not.
'''

#%% Housekeeping

__all__ = ['min_versions', 'check_sciris']

min_versions = {'sciris':'2.0.1'} # Should match requirements.txt


#%% Check dependencies

def check_sciris():
    ''' Check that Sciris is available and the right version '''
    try:
        import sciris as sc
    except ModuleNotFoundError: # pragma: no cover
        errormsg = 'Sciris is a required dependency but is not found; please install via "pip install sciris"'
        raise ModuleNotFoundError(errormsg)
    ver = sc.__version__
    minver = min_versions['sciris']
    if sc.compareversions(ver, minver) < 0:
        errormsg = f'You have Sciris {ver} but {minver} is required; please upgrade via "pip install --upgrade sciris"'
        raise ImportError(errormsg)
    return


# Perform the version checks on import
check_sciris()