#TODO: make configuration dictionaries optinally based on strings?
from .classifiers import classifier_config_dictionary, make_classifier_config_dictionary
from .transformers import transformer_config_dictionary, make_transformer_config_dictionary
from .regressors import regressor_config_dictionary, make_regressor_config_dictionary
from .selectors import selector_config_dictionary, make_selector_config_dictionary
from .special_configs import make_arithmetic_transformer_config_dictionary, make_FSS_config_dictionary, make_passthrough_config_dictionary
from .hyperparametersuggestor import *
try:
    from .mdr_configs import make_skrebate_config_dictionary, make_MDR_config_dictionary, make_ContinuousMDR_config_dictionary
except: #if optional packages are not installed
    pass