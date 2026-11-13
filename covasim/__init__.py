"""
Initialize Covasim by importing all the modules

Convention is to use "import covasim as cv", and then to use all functions and
classes directly, e.g. cv.Sim() rather than cv.sim.Sim().
"""
# Last verified on: Starsim 4.0.0rc4 (009927777)

# Set options
from .settings import *

# Import the version and print the license unless verbosity is disabled, via e.g. os.environ['COVASIM_VERBOSE'] = 0
from .version import __version__, __versiondate__, __license__
if settings.options.verbose:
    print(__license__)

# Import the actual model
from .defaults      import * # Depends on settings
from .misc          import * # Depends on settings, version
from .parameters    import * # Depends on settings, misc, defaults
from .utils         import * # Depends on settings, defaults
from .plotting      import * # Depends on settings, misc, defaults
from .base          import * # No dependencies
from .population    import * # Depends on utils, defaults
from .network       import * # Depends on parameters
from .immunity      import * # Depends on parameters
from .connectors    import * # Depends on immunity
from .covid         import * # Depends on parameters, immunity
from .interventions import * # Depends on compat, utils, parameters, immunity, misc, covid
from .people        import * # Depends on defaults, compat, plotting
from .sim           import * # Depends on almost everything, including analysis
from .analysis      import * # Depends on misc, compat, plotting, settings, interventions, run (imported by sim, which it does not depend on)
from .run           import * # Depends on misc, compat, plotting, settings, sim
# compat (v3 compatibility shims; depends on parameters, network, base, utils, misc) is imported by the modules that need it, and not with *, since its classes are base classes used internally
from .              import data # The demographic data
from .regression    import migrate3to4 # The script for migrating v3 code to v4
