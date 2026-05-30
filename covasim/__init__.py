'''
Initialize Covasim by importing all the modules

Convention is to use "import covasim as cv", and then to use all functions and
classes directly, e.g. cv.Sim() rather than cv.sim.Sim().

v4.0 Starsim port -- M1 (basic transmission) surface. Active modules: settings,
version, defaults, misc, parameters, utils, base (cv.Layer / cv.Contacts /
cv.Result / cv.ParsObj kept; BaseSim / BasePeople dormant), the population
builders, and the new Starsim-based cv.Network, cv.COVID, cv.People, cv.Sim.
Quarantined under covasim/_v2_legacy/ and restored over M2-M10: analysis,
immunity, interventions, run, plotting, and the v3 sim / people engines.
'''

# Check that requirements are met and set options
from . import requirements
from .settings import *

# Import the version and print the license unless verbosity is disabled, via e.g. os.environ['COVASIM_VERBOSE'] = 0
from .version import __version__, __versiondate__, __license__
if settings.options.verbose:
    print(__license__)

# Import the actual model (M1 surface; see module docstring)
from .defaults      import * # Depends on settings
from .misc          import * # Depends on version
from .parameters    import * # Depends on settings, misc
from .utils         import * # Depends on defaults
from .base          import * # cv.Layer / cv.Contacts / cv.Result / cv.ParsObj (BaseSim/BasePeople dormant)
from .population    import * # cv.make_randpop / cv.make_hybrid_contacts / ... (ported in M1 Task 1)
from .network       import * # cv.Network(ss.Network)  -- M1
from .covid         import * # cv.COVID(ss.Infection)  -- M1
from .immunity      import * # cv.variant / build_immunity_matrix -- M3
from .connectors    import * # cv.CrossImmunity(ss.Connector) -- M3
from .people        import * # cv.People(ss.People)    -- M1
from .sim           import * # cv.Sim(ss.Sim)          -- M1
