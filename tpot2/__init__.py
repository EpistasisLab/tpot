
#TODO: are all the imports in the init files done correctly?
#TODO clean up import organization

from .individual import BaseIndividual

from .graphsklearn import GraphPipeline
from .population import Population

from . import builtin_modules
from . import config
from . import search_spaces
from . import utils
from . import evolvers
from . import objectives
from . import selectors
from . import tpot_estimator
from . import old_config_utils

from .tpot_estimator import TPOTClassifier, TPOTRegressor, TPOTEstimator, TPOTEstimatorSteadyState

from update_checker import update_check
from ._version import __version__
update_check("tpot2",__version__)