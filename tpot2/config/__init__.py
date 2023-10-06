#TODO: make configuration dictionaries optinally based on strings?
from .classifiers import make_classifier_config_dictionary
from .transformers import make_transformer_config_dictionary
from .regressors import make_regressor_config_dictionary
from .selectors import make_selector_config_dictionary
from .special_configs import make_arithmetic_transformer_config_dictionary, make_FSS_config_dictionary, make_passthrough_config_dictionary
from .autoqtl_builtins import make_FeatureEncodingFrequencySelector_config_dictionary, make_genetic_encoders_config_dictionary
from .hyperparametersuggestor import *

try:
    from .classifiers_sklearnex import make_sklearnex_classifier_config_dictionary
    from .regressors_sklearnex import make_sklearnex_regressor_config_dictionary
except ModuleNotFoundError: #if optional packages are not installed
    pass

try:
    from .mdr_configs import make_skrebate_config_dictionary, make_MDR_config_dictionary, make_ContinuousMDR_config_dictionary
except: #if optional packages are not installed
    pass

from .classifiers import *