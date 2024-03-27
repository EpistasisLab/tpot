from tpot2.builtin_modules import genetic_encoders
from tpot2.builtin_modules import feature_encoding_frequency_selector
import sklearn
import numpy as np

from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal

FeatureEncodingFrequencySelector_ConfigurationSpace = ConfigurationSpace(
    space = {
        'threshold': Float("threshold", bounds=(0, .35))
    }
)


# genetic_encoders.DominantEncoder : {},
# genetic_encoders.RecessiveEncoder : {},
# genetic_encoders.HeterosisEncoder : {},
# genetic_encoders.UnderDominanceEncoder : {},
# genetic_encoders.OverDominanceEncoder : {},

