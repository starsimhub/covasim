"""
Check that correct versions of each library are installed, and print warnings
or errors if not.
"""

#%% Housekeeping

import importlib
import sciris as sc # Its version is checked below

min_versions = {'sciris':'3.4.0', 'starsim':'4.0.0'} # Should match pyproject.toml


#%% Check dependencies

def check_requirements():
    """ Check that each required library is available and the right version """
    for name,minver in min_versions.items():
        try:
            lib = importlib.import_module(name)
        except ModuleNotFoundError as E: # pragma: no cover
            errormsg = f'{name} is a required dependency but is not found; please install via "pip install {name}"'
            raise ModuleNotFoundError(errormsg) from E
        ver = lib.__version__
        if sc.compareversions(ver, minver) < 0: # pragma: no cover
            errormsg = f'You have {name} {ver} but {minver} is required; please upgrade via "pip install --upgrade {name}"'
            raise ImportError(errormsg)
    return


# Perform the version checks on import
check_requirements()