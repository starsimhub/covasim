'''
Initialize Covasim by importing all the modules

Convention is to use "import covasim as cv", and then to use all functions and
classes directly, e.g. cv.Sim() rather than cv.sim.Sim().
'''

# Check that requirements are met and set options
from . import requirements
from .settings import *

# Import the version and print the license unless verbosity is disabled, via e.g. os.environ['COVASIM_VERBOSE'] = 0
from .version import __version__, __versiondate__, __license__
if settings.options.verbose:
    print(__license__)

# Import the actual model
from .defaults      import * # Depends on settings
from .misc          import * # Depends on version
from .parameters    import * # Depends on settings, misc
from .utils         import * # Depends on defaults
from .plotting      import * # Depends on defaults, misc
from .base          import * # Depends on utils, defaults
from .population    import * # Depends on utils, defaults
from .network       import * # Depends on parameters
from .covid         import * # Depends on parameters, immunity
from .immunity      import * # Depends on parameters
from .connectors    import * # Depends on immunity
from .interventions import * # Depends on covid, parameters
from .people        import * # Depends on defaults
from .sim           import * # Depends on almost everything
from .analysis      import * # Depends on sim
from .run           import * # Depends on sim
from .              import data # The demographic data
from .regression    import migrate3to4 # The script for migrating v3 code to v4
