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

from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing

import numpy as np

from .base import TPOTBase
from .config.classifier import classifier_config_dict
from .config.regressor import regressor_config_dict


class TPOTClassifier(TPOTBase):
    """TPOT estimator for classification problems."""

    scoring_function = 'accuracy'  # Classification scoring
    default_config_dict = classifier_config_dict  # Classification dictionary
    classification = True
    regression = False

    def _init_pretest(self, features, target):
        """Set the sample of data used to verify pipelines work
        with the passed data set.

        This is not intend for anything other than perfunctory dataset
        pipeline compatibility testing
        """

        self.pretest_X, _, self.pretest_y, _ = \
                train_test_split(
                                features,
                                target,
                                random_state=self.random_state,
                                test_size=None,
                                train_size=min(50,int(0.9*features.shape[0])),
                                stratify=target
                                )
        #Make sure there is a least one example from each class
        #for this evaluative test sample
        if not np.array_equal(np.unique(target),np.unique(self.pretest_y)):
            unique_target_idx = np.unique(target,return_index=True)[1]
            self.pretest_y[0:unique_target_idx.shape[0]] = \
                    _safe_indexing(target, unique_target_idx)


class TPOTRegressor(TPOTBase):
    """TPOT estimator for regression problems."""

    scoring_function = 'neg_mean_squared_error'  # Regression scoring
    default_config_dict = regressor_config_dict  # Regression dictionary
    classification = False
    regression = True

    def _init_pretest(self, features, target):
        """Set the sample of data used to verify pipelines work with the passed data set.

        """
        self.pretest_X, _, self.pretest_y, _ = \
                train_test_split(
                                features,
                                target,
                                random_state=self.random_state,
                                test_size=None,
                                train_size=min(50,int(0.9*features.shape[0]))
                                )
