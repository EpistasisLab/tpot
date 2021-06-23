# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

# This configuration only includes the image extractors. 
# These are appended to any config selected if the input is indicated as an image.
# This config is NOT meant to be used by itself. 
# It is selected/appended to any config used if the argument
# input_type='image' is passed to the TPOT object on instantiation

config_imagefeatureextract = {

    #TODO: Add image feature extractor(s) here
    'tpot.builtins.DeepImageFeatureExtractor': {
        'network_name': ["resnet", "alexnet", "vgg", "densenet", "googlenet", "shufflenet", "mobilenet", "resnext", "wide_resnet", "mnasnet"]
    },

    
}
