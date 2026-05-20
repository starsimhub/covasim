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
from .base          import * # Depends on version, misc, defaults, parameters, utils
from .people        import * # Depends on utils, defaults, base, plotting
from .population    import * # Depends on people et al.
from .interventions import * # Depends on defaults, utils, base
from .immunity      import * # Depends on utils, parameters, defaults
from .analysis      import * # Depends on utils, misc, interventions
from .sim           import * # Depends on almost everything
from .run           import * # Depends on sim


