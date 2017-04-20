# -*- coding: utf-8 -*-
"""
Copyright 2015-Present Randal S. Olson

This file is part of the TPOT library.

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

dictionary format (json-like format):
key:
    operator name
value:
    source: module source (e.g sklearn.tree)
    dependencies: depended module (e.g. SVC in selectors RFE); None for no dependency
    params: a dictionary of parameter names (keys) and parameter ranges (values); None for no dependency
"""

tpot_mdr_classifier_config_dict = {

    # Classifiers

    'mdr.MDR': {
        'tie_break': [0, 1],
        'default_label': [0, 1]
    },

    # Feature Selectors

    'skrebate.ReliefF': {
        'n_features_to_select': range(1, 6),
        'n_neighbors': [2, 10, 50, 100, 250, 500]
    },

    'skrebate.SURF': {
        'n_features_to_select': range(1, 6)
    },

    'skrebate.SURFstar': {
        'n_features_to_select': range(1, 6)
    },

    'skrebate.MultiSURF': {
        'n_features_to_select': range(1, 6)
    }

}
