from tpot2.builtin_modules import genetic_encoders
from tpot2.builtin_modules import feature_encoding_frequency_selector
import sklearn
import numpy as np

def params_FeatureEncodingFrequencySelector(trial, name=None):
    return {
        'threshold': trial.suggest_float(f'threshold_{name}', 0, .35)
    }




def make_FeatureEncodingFrequencySelector_config_dictionary():
    return {feature_encoding_frequency_selector.FeatureEncodingFrequencySelector: params_FeatureEncodingFrequencySelector}

def make_genetic_encoders_config_dictionary():
    return {
                genetic_encoders.DominantEncoder : {},
                genetic_encoders.RecessiveEncoder : {},
                genetic_encoders.HeterosisEncoder : {},
                genetic_encoders.UnderDominanceEncoder : {},
                genetic_encoders.OverDominanceEncoder : {},
            }
