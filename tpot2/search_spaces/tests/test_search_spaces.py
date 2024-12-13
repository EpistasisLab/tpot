"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

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
# Test all nodes have all dictionaries
import pytest
import tpot2

import tpot2
from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


def test_EstimatorNodeCrossover():
    knn_configspace = {}
    standard_scaler_configspace = {}

    knn_node = tpot2.search_spaces.nodes.EstimatorNode(
        method = KNeighborsClassifier,
        space = knn_configspace,
    )

    knnind1 = knn_node.generate()
    knnind2 = knn_node.generate()

    for i in range(0,10):
        knnind1.mutate()
        knnind2.mutate()
        knnind1.crossover(knnind2)


def test_ValueError_different_types():
    knn_node = tpot2.config.get_search_space(["KNeighborsClassifier"])
    sfm_wrapper_node = tpot2.config.get_search_space(["SelectFromModel_classification"])

    for i in range(10):
        ind1 = knn_node.generate()
        ind2 = sfm_wrapper_node.generate()
        assert not ind1.crossover(ind2)
        assert not ind2.crossover(ind1)

if __name__ == "__main__":
    test_EstimatorNodeCrossover()
    test_ValueError_different_types()